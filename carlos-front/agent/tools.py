import os
import warnings
import base64
from io import BytesIO
from urllib.parse import urlparse
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from transformers import XLMRobertaTokenizer

from typing import Optional

# Importações ADK para acessar contexto de ferramenta
try:
    from google.adk.tools import ToolContext
    TOOL_CONTEXT_AVAILABLE = True
except ImportError:
    TOOL_CONTEXT_AVAILABLE = False
    ToolContext = None

# Importações e disponibilidade do MUSK
from musk import utils, modeling
from timm.models import create_model
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

MUSK_AVAILABLE = True
MUSK_STATUS = "✅ MUSK disponível"

# ======================================
# CONFIGURAÇÕES DO MODELO
# ======================================
TOP_K = 5
VECTORSTORE_DIR = "./streamlit_chroma_vectorstore_precomputed"
cuda_device = os.environ.get("NVIDIA_VISIBLE_DEVICES", "0")
DEVICE = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() and cuda_device.isdigit() else "cuda:0" if torch.cuda.is_available() else "cpu")

# Cache global para imagens processadas (contorna truncamento do ADK)
_IMAGE_CACHE = {}

# Singletons do modelo e vectorstore
_MUSK_MODEL = None
_MUSK_TRANSFORM = None
_VECTORSTORE = None

# ======================================
# EMBEDDINGS PLACEHOLDER (apenas para query)
# ======================================
class QueryOnlyEmbeddings(Embeddings):
    """Embeddings placeholder - necessário para usar o Chroma."""
    
    def embed_documents(self, texts):
        """Não usado - embeddings já estão no Chroma."""
        raise NotImplementedError("Use apenas para queries")
    
    def embed_query(self, text):
        """Retorna embedding vazio - será substituído pelo vetor real."""
        return [0.0] * 768  # Dimensão placeholder


# ======================================
# FUNÇÕES DE CARREGAMENTO DE MODELO
# ======================================
def load_musk_model():
    """Carrega o modelo MUSK do seu projeto"""
    global _MUSK_MODEL, _MUSK_TRANSFORM

    if _MUSK_MODEL is not None and _MUSK_TRANSFORM is not None:
        return _MUSK_MODEL, _MUSK_TRANSFORM

    try:
        model = create_model("musk_large_patch16_384")
        utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, "model|module", "")
        model.to(device=DEVICE, dtype=torch.float16)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(384, interpolation=3, antialias=True),
            transforms.CenterCrop((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ])

        _MUSK_MODEL = model
        _MUSK_TRANSFORM = transform
        return _MUSK_MODEL, _MUSK_TRANSFORM

    except Exception as e:
        print(f"❌ Erro ao carregar modelo MUSK: {e}")
        return None, None


def load_vectorstore():
    """Carrega o vectorstore já persistido"""
    global _VECTORSTORE
    
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    try:
        embeddings = QueryOnlyEmbeddings()
        _VECTORSTORE = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=embeddings,
            collection_name="isic_images_precomputed"
        )
        return _VECTORSTORE
    except Exception as e:
        print(f"❌ Erro ao carregar vectorstore: {e}")
        return None


# ======================================
# FERRAMENTAS DE BUSCA ADK
# ======================================
def search_by_image_query(top_k: int = TOP_K, tool_context = None) -> str:
    """Busca imagens de lâminas histológicas semelhantes a partir de uma imagem de consulta.
    
    Esta ferramenta utiliza o modelo MUSK (Multimodal Universal Search with Knowledge) para
    gerar embeddings da imagem fornecida e buscar as imagens mais semelhantes no banco de dados
    de lâminas histológicas pré-indexadas.
    
    IMPORTANTE: A imagem é extraída automaticamente do contexto da mensagem do usuário.
    Não é necessário passar a imagem como parâmetro - ela será obtida do cache global
    que é preenchido quando o usuário envia uma imagem.
    
    Args:
        top_k: Número de resultados similares a retornar. Padrão é 5.
        tool_context: Contexto da ferramenta (fornecido automaticamente pelo ADK).
    
    Returns:
        String formatada contendo os resultados da busca, incluindo:
        - Posição do resultado
        - Percentual de similaridade
        - Identificador ou descrição da imagem encontrada
    
    Examples:
        >>> search_by_image_query(top_k=3)
        "Resultado #1: 85.23% de similaridade - ISIC_0053494.jpg\n..."
    """
    # Tentar recuperar imagem do cache ou contexto
    image = None
    
    # Estratégia 1: Recuperar do cache global (preenchido em before_model_modifier)
    if _IMAGE_CACHE:
        print(f"🔍 Cache disponível com {len(_IMAGE_CACHE)} imagem(ns)")
        cache_key = list(_IMAGE_CACHE.keys())[0]
        image = _IMAGE_CACHE[cache_key]
        print(f"✅ Imagem recuperada do cache (tamanho: {len(image)} chars)")
        print(f"📋 Formato da imagem: {image[:100]}...")
        del _IMAGE_CACHE[cache_key]
    
    # Estratégia 2: Se não estiver no cache, tentar acessar do tool_context
    elif tool_context is not None and TOOL_CONTEXT_AVAILABLE:
        try:
            if hasattr(tool_context, 'llm_request') and tool_context.llm_request:
                for content in tool_context.llm_request.contents or []:
                    if content.parts:
                        for part in content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                blob = part.inline_data
                                if hasattr(blob, 'data') and blob.data:
                                    mime_type = getattr(blob, 'mime_type', 'image/png')
                                    image_b64 = base64.b64encode(blob.data).decode('utf-8')
                                    image = f"data:{mime_type};base64,{image_b64}"
                                    break
                        if image and image.startswith("data:image/"):
                            break
        except Exception:
            pass
    
    # Se não houver imagem, retornar erro
    if not image:
        return "❌ Nenhuma imagem foi fornecida. Por favor, envie uma imagem junto com sua mensagem."
    
    # Corrigir padding base64 se necessário
    if image.startswith("data:image/"):
        if "," in image:
            header, encoded = image.split(",", 1)
            remainder = len(encoded) % 4
            if remainder != 0:
                encoded += "=" * (4 - remainder)
                image = f"{header},{encoded}"
    
    # Carregar modelo e vectorstore
    model, transform = load_musk_model()
    vectorstore = load_vectorstore()
    
    if not model or not vectorstore:
        return "❌ Falha ao inicializar modelo ou vectorstore."

    # Processar diferentes formatos de imagem
    try:
        pil_image = None
        
        # Detectar formato e carregar imagem
        if image.startswith("data:image/"):
            if "," in image:
                header, encoded = image.split(",", 1)
            else:
                header = "data:image/png;base64"
                encoded = image
                image = f"{header},{encoded}"
            
            # Corrigir padding base64
            remainder = len(encoded) % 4
            if remainder != 0:
                encoded += "=" * (4 - remainder)
                image = f"{header},{encoded}"
            
            print(f"🔧 Decodificando base64 (tamanho: {len(encoded)} chars)")
            try:
                image_bytes = base64.b64decode(encoded, validate=True)
                print(f"✅ Base64 decodificado com sucesso: {len(image_bytes)} bytes")
            except Exception as e1:
                print(f"⚠️ Falha na validação estrita: {e1}")
                try:
                    image_bytes = base64.b64decode(encoded, validate=False)
                    print(f"✅ Base64 decodificado sem validação: {len(image_bytes)} bytes")
                except Exception as e2:
                    print(f"❌ Falha total na decodificação: {e2}")
                    return f"❌ Erro ao decodificar base64: {e2}. A string pode estar truncada ou corrompida."
            
            print(f"🖼️ Abrindo imagem com PIL...")
            try:
                pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                print(f"✅ Imagem aberta: {pil_image.size}, modo: {pil_image.mode}")
            except Exception as e_pil:
                print(f"❌ ERRO ao abrir imagem com PIL: {type(e_pil).__name__}: {e_pil}")
                import traceback
                traceback.print_exc()
                raise
            
        elif image.startswith(("http://", "https://")):
            try:
                import requests
                response = requests.get(image, timeout=30)
                response.raise_for_status()
                pil_image = Image.open(BytesIO(response.content)).convert("RGB")
            except ImportError:
                return "❌ Biblioteca 'requests' não está instalada. Necessária para download de imagens HTTP/HTTPS."
            except Exception as e:
                return f"❌ Erro ao baixar imagem de {image}: {str(e)}"
                
        else:
            if not os.path.exists(image):
                return f"❌ Arquivo de imagem não encontrado: {image}"
            pil_image = Image.open(image).convert("RGB")
        
        if pil_image is None:
            return "❌ Não foi possível carregar a imagem."
            
    except FileNotFoundError:
        return f"❌ Arquivo de imagem não encontrado: {image}"
    except Exception as e:
        print(f"❌ ERRO GERAL ao processar imagem: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Erro ao processar imagem: {str(e)}"

    try:
        # Redimensionar imagem se muito grande (otimização)
        max_size = 2048
        if max(pil_image.size) > max_size:
            ratio = max_size / max(pil_image.size)
            new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE, dtype=torch.float16)

        with torch.inference_mode():
            features = model(
                image=image_tensor,
                with_head=True,
                out_norm=True,
                return_global=True,
            )[0]

        query_embedding = features.cpu().numpy().flatten()
        
        results = vectorstore.similarity_search_by_vector_with_relevance_scores(query_embedding, k=top_k)

        # Formatar resultados como string legível
        result_lines = [f"\n📊 Resultados da busca por imagem (Imagem → Imagens semelhantes):"]
        for i, (doc, score) in enumerate(results, start=1):
            similarity_percent = max(0, (1 - score/2) * 100)
            result_line = f"  #{i:02d} | {similarity_percent:.2f}% de similaridade | {doc.page_content}"
            result_lines.append(result_line)
        
        result_lines.append("—" * 60)
        
        return "\n".join(result_lines)
    
    except Exception as e:
        return f"❌ Erro ao processar imagem: {str(e)}"


def search_by_text_query(text_query: str, top_k: int = TOP_K) -> str:
    """Busca imagens de lâminas histológicas a partir de uma descrição textual.
    
    Esta ferramenta utiliza o modelo MUSK (Multimodal Universal Search with Knowledge) para
    gerar embeddings da descrição textual fornecida e buscar as imagens mais semelhantes no
    banco de dados de lâminas histológicas pré-indexadas. Ideal para buscar por características
    histológicas específicas, diagnósticos ou padrões morfológicos descritos em texto.
    
    Args:
        text_query: Descrição textual da lâmina histológica ou características a buscar.
                   Exemplos: "prostate adenocarcinoma with cribriform pattern",
                   "melanoma with dermal invasion", "benign nevus".
        top_k: Número de resultados similares a retornar. Padrão é 5.
    
    Returns:
        String formatada contendo os resultados da busca, incluindo:
        - Posição do resultado
        - Percentual de similaridade
        - Identificador ou descrição da imagem encontrada
        
    Example:
        >>> search_by_text_query("prostate adenocarcinoma with cribriform pattern", top_k=3)
        "Resultado #1: 92.15% de similaridade - ISIC_0053494.jpg\n..."
    """
    model, _ = load_musk_model()
    vectorstore = load_vectorstore()
    if not model or not vectorstore:
        return "❌ Falha ao inicializar modelo ou vectorstore."

    try:
        tokenizer = XLMRobertaTokenizer("./src/models/tokenizer.spm")
        txt_ids, pad = utils.xlm_tokenizer(text_query, tokenizer, max_len=100)
        txt_ids_tensor = torch.tensor(txt_ids, dtype=torch.long).unsqueeze(0)
        pad_tensor = torch.tensor(pad, dtype=torch.bool).unsqueeze(0)

        with torch.inference_mode():
            features = model(
                text_description=txt_ids_tensor.to(DEVICE),
                padding_mask=pad_tensor.to(DEVICE),
                with_head=True,
                out_norm=True,
                return_global=True,
            )[1]

        query_embedding = features.cpu().numpy().flatten().tolist()
        results = vectorstore.similarity_search_by_vector_with_relevance_scores(query_embedding, k=top_k)

        # Formatar resultados como string legível
        result_lines = [f"\n📊 Resultados da busca textual (Texto → Imagens correspondentes):"]
        for i, (doc, score) in enumerate(results, start=1):
            similarity_percent = max(0, (1 - score/2) * 100)
            result_line = f"  #{i:02d} | {similarity_percent:.2f}% de similaridade | {doc.page_content}"
            result_lines.append(result_line)
        
        result_lines.append("—" * 60)
        
        return "\n".join(result_lines)
    
    except Exception as e:
        return f"❌ Erro ao processar consulta textual: {str(e)}"


# # ======================================
# # EXECUÇÃO MANUAL (TESTES)
# # ======================================
# if __name__ == "__main__":
#     image_path = "ISIC_0053494.jpg"  # substitua pelo caminho real no container
#     if os.path.exists(image_path):
#         search_by_image_query(image_path, top_k=TOP_K)
#     else:
#         print(f"⚠️ Caminho da imagem de teste não encontrado: {image_path}")

#     # 🔹 Busca textual de teste
#     query = "prostate adenocarcinoma with cribriform pattern"
#     search_by_text_query(query, top_k=TOP_K)
    

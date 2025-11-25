import os
import warnings
import base64
import json
from io import BytesIO
from urllib.parse import urlparse
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from transformers import XLMRobertaTokenizer

from typing import Optional, Any

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
def _extract_image_from_context(tool_context: Optional[Any]):
    """Recupera bytes da última imagem enviada pelo usuário via ToolContext."""
    if tool_context is None or not TOOL_CONTEXT_AVAILABLE:
        return None, None
    try:
        llm_request = getattr(tool_context, 'llm_request', None)
        contents = getattr(llm_request, 'contents', None) or []
        for content in reversed(contents):
            if getattr(content, 'role', None) != 'user' or not getattr(content, 'parts', None):
                continue
            for part in content.parts:
                if hasattr(part, 'inline_data') and part.inline_data and getattr(part.inline_data, 'data', None):
                    mime = getattr(part.inline_data, 'mime_type', 'image/png')
                    return part.inline_data.data, mime
                if hasattr(part, 'text') and part.text:
                    try:
                        payload = json.loads(part.text)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if isinstance(payload, list):
                        for item in payload:
                            if isinstance(item, dict) and item.get('type') in {'binary', 'image_ref'}:
                                data_field = item.get('data')
                                if data_field:
                                    if isinstance(data_field, str):
                                        data_str = data_field.split(',', 1)[1] if ',' in data_field else data_field
                                        try:
                                            return base64.b64decode(data_str), item.get('mimeType', 'image/png')
                                        except Exception:
                                            continue
                                path = item.get('path')
                                if path and os.path.exists(path):
                                    with open(path, 'rb') as f:
                                        return f.read(), item.get('mimeType', 'image/png')
        return None, None
    except Exception:
        return None, None


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
    # Recuperar imagem diretamente do contexto
    image_bytes = None
    mime_type = 'image/png'
    if tool_context is not None:
        image_bytes, mime_type = _extract_image_from_context(tool_context)
    
    if not image_bytes:
        return "❌ Nenhuma imagem foi fornecida. Por favor, envie uma imagem junto com sua mensagem."
    
    # Carregar modelo e vectorstore
    model, transform = load_musk_model()
    vectorstore = load_vectorstore()
    
    if not model or not vectorstore:
        return "❌ Falha ao inicializar modelo ou vectorstore."

    # Processar diferentes formatos de imagem
    try:
        pil_image = None
        
        # Detectar formato e carregar imagem
        if image_bytes is not None:
            print(f"🖼️ Abrindo imagem com PIL ({len(image_bytes)} bytes)...")
            try:
                pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                print(f"✅ Imagem aberta: {pil_image.size}, modo: {pil_image.mode}")
            except Exception as e_pil:
                print(f"❌ ERRO ao abrir imagem com PIL: {type(e_pil).__name__}: {e_pil}")
                import traceback
                traceback.print_exc()
                raise
        else:
            return "❌ Não foi possível carregar a imagem fornecida."
        
        if pil_image is None:
            return "❌ Não foi possível carregar a imagem."
            
    except FileNotFoundError:
        return "❌ Arquivo de imagem não encontrado."
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
    

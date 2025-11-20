import os
import warnings
import base64
import time
from io import BytesIO
from urllib.parse import urlparse
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from transformers import XLMRobertaTokenizer

# MUSK imports e disponibilidade
from musk import utils, modeling
from timm.models import create_model
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

MUSK_AVAILABLE = True
MUSK_STATUS = "✅ MUSK disponível"

# ======================================
# CONFIGURAÇÕES
# ======================================
TOP_K = 5
VECTORSTORE_DIR = "./streamlit_chroma_vectorstore_precomputed"
cuda_device = os.environ.get("NVIDIA_VISIBLE_DEVICES", "0")
DEVICE = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() and cuda_device.isdigit() else "cuda:0" if torch.cuda.is_available() else "cpu")

# Singleton
_MUSK_MODEL = None
_MUSK_TRANSFORM = None
_VECTORSTORE = None

# ======================================
# EMBEDDINGS DUMMY (só para query)
# ======================================
class QueryOnlyEmbeddings(Embeddings):
    """Embeddings dummy - precisa disso pra usar o chroma"""
    
    def embed_documents(self, texts):
        """Não usado - embeddings já estão no Chroma"""
        raise NotImplementedError("Use apenas para queries")
    
    def embed_query(self, text):
        """Retorna embedding vazio - será substituído pelo vetor real"""
        return [0.0] * 768  # Dimensão placeholder


# ======================================
# FUNÇÕES DE SUPORTE
# ======================================
def load_musk_model():
    """Carrega o modelo MUSK do seu projeto"""
    global _MUSK_MODEL, _MUSK_TRANSFORM

    if _MUSK_MODEL is not None and _MUSK_TRANSFORM is not None:
        return _MUSK_MODEL, _MUSK_TRANSFORM
    
    print("🧩 Carregando modelo MUSK...")

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

        print("✅ Modelo carregado com sucesso!")
        return _MUSK_MODEL, _MUSK_TRANSFORM

    except Exception as e:
        print(f"❌ Erro ao carregar modelo MUSK: {e}")
        return None, None


def load_vectorstore():
    """Carrega o vectorstore já persistido"""
    global _VECTORSTORE
    
    if _VECTORSTORE is not None:
        return _VECTORSTORE
    print("📂 Carregando vectorstore pré-computado...")

    try:
        embeddings = QueryOnlyEmbeddings()
        _VECTORSTORE = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=embeddings,
            collection_name="isic_images_precomputed"
        )
        print("✅ Vectorstore carregado com sucesso!")
        return _VECTORSTORE
    except Exception as e:
        print(f"❌ Erro ao carregar vectorstore: {e}")
        return None


# ======================================
# FERRAMENTAS ADK
# ======================================
def search_by_image_query(image: str, top_k: int = TOP_K) -> str:
    """Busca imagens de lâminas histológicas semelhantes a partir de uma imagem de consulta.
    
    Esta ferramenta utiliza o modelo MUSK (Multimodal Universal Search with Knowledge) para
    gerar embeddings da imagem fornecida e buscar as imagens mais semelhantes no banco de dados
    de lâminas histológicas pré-indexadas.
    
    IMPORTANTE: Esta função processa a imagem internamente e retorna apenas os resultados da busca.
    Não inclua dados brutos da imagem (como strings base64) no output ou nas respostas ao usuário.
    Apenas apresente os resultados formatados retornados por esta função.
    
    Args:
        image: String representando a imagem em um dos seguintes formatos:
            - Caminho local de arquivo (ex: "/path/to/image.jpg")
            - URI do Google Cloud Storage (ex: "gs://bucket/image.jpg")
            - URI HTTP/HTTPS (ex: "https://example.com/image.jpg")
            - Data URI base64 (ex: "data:image/jpeg;base64,/9j/4AAQ...")
            Quando chamado pelo ADK com conteúdo multimodal, a imagem é passada como string
            (geralmente URI ou base64). Strings base64 muito longas (>10MB) podem causar lentidão.
        top_k: Número de resultados similares a retornar. Padrão é 5.
    
    Returns:
        String formatada contendo os resultados da busca, incluindo:
        - Posição do resultado
        - Percentual de similaridade
        - Identificador ou descrição da imagem encontrada
        
        NOTA: Retorne apenas este resultado formatado. Não inclua os dados brutos da imagem
        (como a string base64) na resposta ao usuário.
        
    Examples:
        >>> # Usando caminho local
        >>> search_by_image_query("/path/to/histology_slide.jpg", top_k=3)
        "Resultado #1: 85.23% de similaridade - ISIC_0053494.jpg\n..."
        
        >>> # Usando URI HTTP
        >>> search_by_image_query("https://example.com/image.jpg", top_k=3)
        "Resultado #1: 85.23% de similaridade - ISIC_0053494.jpg\n..."
        
        >>> # Usando base64 (quando chamado pelo ADK)
        >>> search_by_image_query("data:image/jpeg;base64,/9j/4AAQ...", top_k=3)
        "Resultado #1: 85.23% de similaridade - ISIC_0053494.jpg\n..."
    """
    start_time = time.time()
    
    # Validar tamanho da string de entrada (especialmente para base64)
    MAX_INPUT_SIZE = 20 * 1024 * 1024  # 20MB
    if len(image) > MAX_INPUT_SIZE:
        error_msg = f"❌ String de imagem muito grande ({len(image) / 1024 / 1024:.1f} MB). Máximo permitido: {MAX_INPUT_SIZE / 1024 / 1024:.1f} MB. Considere usar uma URI em vez de base64."
        print(error_msg)
        return error_msg
    
    # Carregar modelo e vectorstore
    t0 = time.time()
    model, transform = load_musk_model()
    vectorstore = load_vectorstore()
    load_time = time.time() - t0
    if load_time > 0.1:  # Só logar se demorar mais que 100ms
        print(f"⏱️  Tempo de carregamento de modelo/vectorstore: {load_time:.2f}s")
    
    if not model or not vectorstore:
        error_msg = "❌ Falha ao inicializar modelo ou vectorstore."
        print(error_msg)
        return error_msg

    # Processar diferentes formatos de string
    try:
        pil_image = None
        t1 = time.time()
        
        # Detectar formato e carregar imagem
        if image.startswith("data:image/"):
            # Data URI base64
            print(f"\n🔍 Realizando busca por imagem (base64)")
            
            # Validar tamanho do base64 antes de decodificar
            base64_size = len(image.split(",", 1)[1]) if "," in image else 0
            estimated_bytes = int(base64_size * 3 / 4)  # Aproximação: base64 é ~33% maior que bytes
            if estimated_bytes > 10 * 1024 * 1024:  # 10MB
                print(f"⚠️  Aviso: Imagem base64 muito grande (estimado: {estimated_bytes / 1024 / 1024:.1f} MB). Isso pode causar lentidão.")
            
            t_decode = time.time()
            header, encoded = image.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            decode_time = time.time() - t_decode
            print(f"⏱️  Tempo de decodificação base64: {decode_time:.2f}s (tamanho: {len(image_bytes) / 1024:.1f} KB)")
            
            t_open = time.time()
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            open_time = time.time() - t_open
            print(f"⏱️  Tempo de abertura de imagem: {open_time:.2f}s (dimensões: {pil_image.size})")
            
        elif image.startswith("gs://"):
            # URI GCS
            print(f"\n🔍 Realizando busca por imagem: {os.path.basename(image)}")
            # Por enquanto, URIs GCS precisam ser convertidas para caminho local primeiro
            raise NotImplementedError("URIs GCS precisam ser convertidas para caminho local primeiro")
            
        elif image.startswith(("http://", "https://")):
            # URI HTTP/HTTPS
            print(f"\n🔍 Realizando busca por imagem (HTTP/HTTPS): {os.path.basename(urlparse(image).path)}")
            try:
                import requests
                t_download = time.time()
                response = requests.get(image, timeout=30)
                response.raise_for_status()
                download_time = time.time() - t_download
                print(f"⏱️  Tempo de download: {download_time:.2f}s (tamanho: {len(response.content) / 1024:.1f} KB)")
                
                t_open = time.time()
                pil_image = Image.open(BytesIO(response.content)).convert("RGB")
                open_time = time.time() - t_open
                print(f"⏱️  Tempo de abertura de imagem: {open_time:.2f}s (dimensões: {pil_image.size})")
            except ImportError:
                error_msg = "❌ Biblioteca 'requests' não está instalada. Necessária para download de imagens HTTP/HTTPS."
                print(error_msg)
                return error_msg
            except Exception as e:
                error_msg = f"❌ Erro ao baixar imagem de {image}: {str(e)}"
                print(error_msg)
                return error_msg
                
        else:
            # Caminho local de arquivo
            print(f"\n🔍 Realizando busca por imagem: {os.path.basename(image)}")
            if not os.path.exists(image):
                error_msg = f"❌ Arquivo de imagem não encontrado: {image}"
                print(error_msg)
                return error_msg
            t_open = time.time()
            pil_image = Image.open(image).convert("RGB")
            open_time = time.time() - t_open
            print(f"⏱️  Tempo de abertura de imagem: {open_time:.2f}s (dimensões: {pil_image.size})")
        
        image_load_time = time.time() - t1
        if image_load_time > 0.5:  # Só logar se demorar mais que 500ms
            print(f"⏱️  Tempo total de carregamento de imagem: {image_load_time:.2f}s")
        
        if pil_image is None:
            error_msg = "❌ Não foi possível carregar a imagem."
            print(error_msg)
            return error_msg
            
    except FileNotFoundError:
        error_msg = f"❌ Arquivo de imagem não encontrado: {image}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Erro ao processar imagem: {str(e)}"
        print(error_msg)
        return error_msg

    try:
        # Redimensionar imagem se muito grande (otimização)
        original_size = pil_image.size
        max_size = 2048  # Limite de 2048px no maior lado
        if max(pil_image.size) > max_size:
            t_resize = time.time()
            ratio = max_size / max(pil_image.size)
            new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            resize_time = time.time() - t_resize
            print(f"⏱️  Imagem redimensionada de {original_size} para {new_size} em {resize_time:.2f}s")
        
        t_transform = time.time()
        image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE, dtype=torch.float16)
        transform_time = time.time() - t_transform
        if transform_time > 0.1:
            print(f"⏱️  Tempo de transformação: {transform_time:.2f}s")

        t_model = time.time()
        with torch.inference_mode():
            features = model(
                image=image_tensor,
                with_head=True,
                out_norm=True,
                return_global=True,
            )[0]
        model_time = time.time() - t_model
        print(f"⏱️  Tempo de processamento do modelo MUSK: {model_time:.2f}s")

        t_embedding = time.time()
        query_embedding = features.cpu().numpy().flatten()
        embedding_time = time.time() - t_embedding
        if embedding_time > 0.1:
            print(f"⏱️  Tempo de conversão para embedding: {embedding_time:.2f}s")
        
        t_search = time.time()
        results = vectorstore.similarity_search_by_vector_with_relevance_scores(query_embedding, k=top_k)
        search_time = time.time() - t_search
        print(f"⏱️  Tempo de busca no vectorstore: {search_time:.2f}s")

        # Formatar resultados como string legível
        result_lines = [f"\n📊 Resultados da busca por imagem (Imagem → Imagens semelhantes):"]
        for i, (doc, score) in enumerate(results, start=1):
            similarity_percent = max(0, (1 - score/2) * 100)
            result_line = f"  #{i:02d} | {similarity_percent:.2f}% de similaridade | {doc.page_content}"
            result_lines.append(result_line)
            print(result_line)
        
        result_lines.append("—" * 60)
        print("—" * 60)
        
        total_time = time.time() - start_time
        print(f"⏱️  Tempo total da função: {total_time:.2f}s")
        
        return "\n".join(result_lines)
    
    except Exception as e:
        error_msg = f"❌ Erro ao processar imagem: {str(e)}"
        print(error_msg)
        total_time = time.time() - start_time
        print(f"⏱️  Tempo total (com erro): {total_time:.2f}s")
        return error_msg


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
        error_msg = "❌ Falha ao inicializar modelo ou vectorstore."
        print(error_msg)
        return error_msg

    print(f"\n🔍 Realizando busca textual: \"{text_query}\"")

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
            print(result_line)
        
        result_lines.append("—" * 60)
        print("—" * 60)
        
        return "\n".join(result_lines)
    
    except Exception as e:
        error_msg = f"❌ Erro ao processar consulta textual: {str(e)}"
        print(error_msg)
        return error_msg


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
    

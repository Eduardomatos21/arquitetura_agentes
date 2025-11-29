import os
import warnings
import base64
import json
import logging
import math
from io import BytesIO
from urllib.parse import urlparse
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from transformers import XLMRobertaTokenizer

from typing import Optional, Any, Iterable, List, Tuple

# Logger configurado localmente para evitar configuração global acidental
logger = logging.getLogger("histopathology.tools")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[tools] %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

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
from session_media_store import load_image_from_state

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
# AJUDANTES DE FILTRO POR METADADOS
# ======================================
def _normalize_sex_filter(value: Optional[str]) -> Optional[str]:
    """Normaliza valores de sexo para 'male' ou 'female'."""
    if value is None:
        return None

    mapping = {
        "f": "female",
        "feminino": "female",
        "feminina": "female",
        "mulher": "female",
        "female": "female",
        "m": "male",
        "masculino": "male",
        "masculina": "male",
        "homem": "male",
        "male": "male",
    }

    normalized = value.strip().lower()
    sex = mapping.get(normalized)
    if sex is None and normalized in {"masculine", "feminine"}:
        sex = "male" if normalized.startswith("masc") else "female"
    if sex is None:
        logger.warning("Ignoring unsupported sex filter value: %s", value)
    return sex


def _coerce_age(value: Optional[Any]) -> Optional[int]:
    """Converte entrada arbitrária em inteiro, se possível."""
    if value is None:
        return None
    try:
        # Strings como "55 anos" ou "idade>60" são comuns; extrair dígitos
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            value = digits or value
        return int(value)
    except (ValueError, TypeError):
        logger.warning("Ignoring invalid age value: %s", value)
        return None


def _prepare_filters(sex: Optional[str], min_age: Optional[Any], max_age: Optional[Any]):
    """Normaliza filtros e garante consistência."""
    normalized_sex = _normalize_sex_filter(sex)
    normalized_min_age = _coerce_age(min_age)
    normalized_max_age = _coerce_age(max_age)

    if (
        normalized_min_age is not None
        and normalized_max_age is not None
        and normalized_min_age > normalized_max_age
    ):
        logger.info(
            "Swapping min/max age to maintain order (min=%s max=%s)",
            normalized_min_age,
            normalized_max_age,
        )
        normalized_min_age, normalized_max_age = normalized_max_age, normalized_min_age

    return normalized_sex, normalized_min_age, normalized_max_age


MetadataResult = Tuple[str, float, dict]


def _metadata_matches_filters(
    metadata: Optional[dict],
    sex: Optional[str],
    min_age: Optional[int],
    max_age: Optional[int],
) -> bool:
    """Retorna True se os metadados atendem aos filtros solicitados."""
    metadata = metadata or {}
    patient_sex = metadata.get("sex")
    patient_age = metadata.get("age_approx")
    if patient_age is None:
        patient_age = metadata.get("age")

    if patient_sex:
        patient_sex = str(patient_sex).strip().lower()
    if patient_age is not None:
        try:
            patient_age = int(float(patient_age))
        except (ValueError, TypeError):
            patient_age = None

    if sex is not None:
        if patient_sex is None:
            return False
        if patient_sex not in {"male", "female"}:
            patient_sex = _normalize_sex_filter(patient_sex)
        if patient_sex != sex:
            return False

    if min_age is not None:
        if patient_age is None or patient_age < min_age:
            return False

    if max_age is not None:
        if patient_age is None or patient_age > max_age:
            return False

    return True


def _filter_metadata_results(
    candidates: Iterable[MetadataResult],
    sex: Optional[str],
    min_age: Optional[int],
    max_age: Optional[int],
) -> Tuple[List[MetadataResult], List[MetadataResult]]:
    """Separa resultados que atendem aos filtros e os demais."""
    matched: List[MetadataResult] = []
    remainder: List[MetadataResult] = []

    for doc_path, distance, metadata in candidates:
        if _metadata_matches_filters(metadata, sex, min_age, max_age):
            matched.append((doc_path, distance, metadata))
        else:
            remainder.append((doc_path, distance, metadata))

    return matched, remainder


def _summarize_filters(sex: Optional[str], min_age: Optional[int], max_age: Optional[int]) -> Optional[str]:
    """Gera texto breve com os filtros aplicados."""
    parts = []
    if sex == "female":
        parts.append("sexo: feminino")
    elif sex == "male":
        parts.append("sexo: masculino")
    if min_age is not None and max_age is not None:
        parts.append(f"idade aproximada: {min_age}-{max_age} anos")
    elif min_age is not None:
        parts.append(f"idade mínima: {min_age} anos")
    elif max_age is not None:
        parts.append(f"idade máxima: {max_age} anos")
    return ", ".join(parts) if parts else None

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
        logger.info("Reusing cached MUSK model and transform on device %s", DEVICE)
        return _MUSK_MODEL, _MUSK_TRANSFORM

    try:
        logger.info("Loading MUSK model (musk_large_patch16_384) on device %s", DEVICE)
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
        logger.info("MUSK model loaded successfully")
        return _MUSK_MODEL, _MUSK_TRANSFORM

    except Exception as e:
        logger.exception("Failed to load MUSK model: %s", e)
        print(f"❌ Erro ao carregar modelo MUSK: {e}")
        return None, None


def load_vectorstore():
    """Carrega o vectorstore já persistido"""
    global _VECTORSTORE
    
    if _VECTORSTORE is not None:
        logger.info("Reusing cached Chroma vectorstore at %s", VECTORSTORE_DIR)
        return _VECTORSTORE

    try:
        logger.info("Loading Chroma vectorstore from %s", VECTORSTORE_DIR)
        embeddings = QueryOnlyEmbeddings()
        _VECTORSTORE = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=embeddings,
            collection_name="isic_images_precomputed"
        )
        logger.info("Vectorstore loaded successfully with collection 'isic_images_precomputed'")
        return _VECTORSTORE
    except Exception as e:
        logger.exception("Failed to load vectorstore: %s", e)
        print(f"❌ Erro ao carregar vectorstore: {e}")
        return None


# ======================================
# FERRAMENTAS DE BUSCA ADK
# ======================================
def _extract_image_from_context(tool_context: Optional[Any]):
    """Recupera bytes da última imagem enviada pelo usuário via ToolContext."""
    if tool_context is None or not TOOL_CONTEXT_AVAILABLE:
        logger.warning("ToolContext unavailable or not provided; cannot extract inline image")
        return None, None
    try:
        logger.info("Attempting to extract inline image from ToolContext payload")
        llm_request = getattr(tool_context, 'llm_request', None)
        contents = getattr(llm_request, 'contents', None) or []
        for content in reversed(contents):
            if getattr(content, 'role', None) != 'user' or not getattr(content, 'parts', None):
                continue
            for part in content.parts:
                if hasattr(part, 'inline_data') and part.inline_data and getattr(part.inline_data, 'data', None):
                    mime = getattr(part.inline_data, 'mime_type', 'image/png')
                    logger.info("Found inline_data directly in user part; mime=%s", mime)
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
                                            logger.info("Decoding base64 payload embedded in text message")
                                            return base64.b64decode(data_str), item.get('mimeType', 'image/png')
                                        except Exception:
                                            logger.exception("Base64 decode failed for embedded binary payload")
                                            continue
                                path = item.get('path')
                                if path and os.path.exists(path):
                                    logger.info("Loading image from referenced path: %s", path)
                                    with open(path, 'rb') as f:
                                        return f.read(), item.get('mimeType', 'image/png')
    except Exception:
        logger.exception("Unexpected error while extracting image from ToolContext")
        return None, None
    # Fallback para estado persistente da sessão
    state = getattr(tool_context, 'state', None)
    if state is not None:
        cached_bytes, cached_mime = load_image_from_state(state)
        if cached_bytes:
            logger.info("Using cached image from session state; mime=%s", cached_mime)
            return cached_bytes, cached_mime
    logger.warning("No image found in ToolContext or session state")
    return None, None


def search_by_image_query(
    top_k: int = TOP_K,
    sex: Optional[str] = None,
    min_age: Optional[Any] = None,
    max_age: Optional[Any] = None,
    tool_context = None,
) -> str:
    """Busca imagens de lâminas histológicas semelhantes a partir de uma imagem de consulta.
    
    Esta ferramenta utiliza o modelo MUSK (Multimodal Universal Search with Knowledge) para
    gerar embeddings da imagem fornecida e buscar as imagens mais semelhantes no banco de dados
    de lâminas histológicas pré-indexadas.
    
    IMPORTANTE: A imagem é extraída automaticamente do contexto da mensagem do usuário.
    Não é necessário passar a imagem como parâmetro - ela será obtida do cache global
    que é preenchido quando o usuário envia uma imagem.
    
    Args:
          top_k: Número de resultados similares a retornar. Padrão é 5.
          sex: Filtra por sexo biológico registrado no dataset ("female" ou "male").
              Aceita variantes em português como "feminino" ou "masculino".
          min_age: Idade mínima aproximada do paciente (inteiro).
          max_age: Idade máxima aproximada do paciente (inteiro).
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
    sex, min_age, max_age = _prepare_filters(sex, min_age, max_age)
    filters_applied = any(v is not None for v in (sex, min_age, max_age))
    logger.info(
        "search_by_image_query invoked top_k=%s sex=%s min_age=%s max_age=%s",
        top_k,
        sex,
        min_age,
        max_age,
    )
    if tool_context is not None:
        logger.info("ToolContext provided; attempting inline extraction")
        image_bytes, mime_type = _extract_image_from_context(tool_context)
    
    if not image_bytes:
        logger.error("Image bytes missing for search_by_image_query")
        return "❌ Nenhuma imagem foi fornecida. Por favor, envie uma imagem junto com sua mensagem."
    
    # Carregar modelo e vectorstore
    model, transform = load_musk_model()
    vectorstore = load_vectorstore()
    collection = getattr(vectorstore, "_collection", None) if vectorstore else None
    
    if not model or not vectorstore or not collection:
        logger.error("Model or vectorstore unavailable (model=%s, vectorstore=%s)", bool(model), bool(vectorstore))
        return "❌ Falha ao inicializar modelo ou vectorstore."

    # Processar diferentes formatos de imagem
    try:
        pil_image = None
        
        # Detectar formato e carregar imagem
        if image_bytes is not None:
            logger.info("Opening image with PIL; %s bytes; mime=%s", len(image_bytes), mime_type)
            try:
                pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                logger.info("PIL image opened successfully size=%s mode=%s", pil_image.size, pil_image.mode)
            except Exception as e_pil:
                logger.exception("Failed to open image with PIL: %s", e_pil)
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
            logger.info("Resizing large image from %s to %s", pil_image.size, new_size)
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE, dtype=torch.float16)

        with torch.inference_mode():
            features = model(
                image=image_tensor,
                with_head=True,
                out_norm=True,
                return_global=True,
            )[0]

        query_embedding_np = features.cpu().numpy().flatten()
        norm = math.sqrt(float((query_embedding_np * query_embedding_np).sum()))
        query_embedding = query_embedding_np.tolist()
        logger.info("Image embedding length=%s l2_norm=%.4f", len(query_embedding), norm)
        
        try:
            n_results = top_k if not filters_applied else 100
            raw = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["distances", "documents", "metadatas"],
            )
        except Exception as raw_error:
            logger.exception("Failed to query Chroma directly: %s", raw_error)
            return "❌ Erro ao consultar o banco vetorial."
        raw_documents = raw.get("documents", [[]])[0]
        raw_distances = raw.get("distances", [[]])[0]
        raw_metadatas = raw.get("metadatas", [[]])[0]
        logger.info("Chroma returned %s raw document(s) for image query", len(raw_documents))
        candidates = list(zip(raw_documents, raw_distances, raw_metadatas))

        fallback_used = False
        if filters_applied:
            matched, remainder = _filter_metadata_results(
                candidates,
                sex,
                min_age,
                max_age,
            )
            if len(matched) < top_k and remainder:
                needed = top_k - len(matched)
                matched.extend(remainder[:needed])
                fallback_used = True
            candidates = matched[:top_k]
            logger.info(
                "Image query after filters yielded %s match(es); fallback_used=%s",
                len(candidates),
                fallback_used,
            )
        else:
            candidates = candidates[:top_k]

        if not candidates:
            filter_msg = _summarize_filters(sex, min_age, max_age)
            if filter_msg:
                return (
                    "⚠️ Nenhum resultado encontrado com os filtros aplicados. "
                    f"Filtros: {filter_msg}. Tente ajustar ou remover os filtros."
                )
            logger.warning("No raw documents returned for image query")
            return "⚠️ Nenhuma imagem semelhante foi encontrada." 

        # Formatar resultados como string legível
        result_lines = [f"\n📊 Resultados da busca por imagem (Imagem → Imagens semelhantes):"]
        filter_summary = _summarize_filters(sex, min_age, max_age)
        if filter_summary:
            result_lines.append(f"  ↳ Filtros aplicados: {filter_summary}")
            if fallback_used:
                result_lines.append(
                    "  ↳ Alguns resultados adicionais não atendem totalmente aos filtros; listados para completar o top_k."
                )
        elif fallback_used:
            fallback_used = False  # Segurança; não deve ocorrer sem filtros.

        for i, (doc_path, distance, metadata) in enumerate(candidates, start=1):
            similarity_percent = max(0, (1 - distance / 2) * 100)
            doc_label = metadata.get("isic_id") if isinstance(metadata, dict) else None
            display = doc_label or doc_path
            patient_sex = metadata.get("sex") if isinstance(metadata, dict) else None
            patient_age = metadata.get("age_approx") if isinstance(metadata, dict) else None
            if patient_age is None and isinstance(metadata, dict):
                patient_age = metadata.get("age")
            try:
                patient_age = int(float(patient_age)) if patient_age is not None else None
            except (ValueError, TypeError):
                patient_age = None
            matches_filters = (
                not filters_applied or _metadata_matches_filters(metadata, sex, min_age, max_age)
            )
            extra_bits = []
            if patient_sex:
                extra_bits.append(f"sexo: {patient_sex}")
            if patient_age is not None:
                extra_bits.append(f"idade≈{patient_age}")
            if filters_applied and not matches_filters:
                extra_bits.append("⚠️ fora dos filtros")
            extras = f" ({', '.join(extra_bits)})" if extra_bits else ""
            result_line = (
                f"  #{i:02d} | {similarity_percent:.2f}% de similaridade | {display}{extras}"
            )
            result_lines.append(result_line)
            logger.info(
                "Image result #%s distance=%.4f similarity=%.2f doc=%s",
                i,
                distance,
                similarity_percent,
                display,
            )
        result_lines.append("—" * 60)
        logger.info("Formatted image search response with %s entries", len(result_lines) - 2)

        return "\n".join(result_lines)
    
    except Exception as e:
        logger.exception("Unhandled error during image search: %s", e)
        return f"❌ Erro ao processar imagem: {str(e)}"


def search_by_text_query(
    text_query: str,
    top_k: int = TOP_K,
    sex: Optional[str] = None,
    min_age: Optional[Any] = None,
    max_age: Optional[Any] = None,
) -> str:
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
          sex: Filtra por sexo biológico registrado no dataset ("female" ou "male").
              Aceita variantes em português como "feminino" ou "masculino".
          min_age: Idade mínima aproximada do paciente (inteiro).
          max_age: Idade máxima aproximada do paciente (inteiro).
    
    Returns:
        String formatada contendo os resultados da busca, incluindo:
        - Posição do resultado
        - Percentual de similaridade
        - Identificador ou descrição da imagem encontrada
        
    Example:
        >>> search_by_text_query("prostate adenocarcinoma with cribriform pattern", top_k=3)
        "Resultado #1: 92.15% de similaridade - ISIC_0053494.jpg\n..."
    """
    sex, min_age, max_age = _prepare_filters(sex, min_age, max_age)
    filters_applied = any(v is not None for v in (sex, min_age, max_age))
    logger.info(
        "search_by_text_query invoked top_k=%s sex=%s min_age=%s max_age=%s query='%s'",
        top_k,
        sex,
        min_age,
        max_age,
        text_query[:80],
    )
    model, _ = load_musk_model()
    vectorstore = load_vectorstore()
    collection = getattr(vectorstore, "_collection", None) if vectorstore else None
    if not model or not vectorstore or not collection:
        logger.error("Model or vectorstore unavailable for text query (model=%s, vectorstore=%s)", bool(model), bool(vectorstore))
        return "❌ Falha ao inicializar modelo ou vectorstore."

    try:
        tokenizer = XLMRobertaTokenizer("./tokenizer/tokenizer.spm")
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
        norm = math.sqrt(sum(x * x for x in query_embedding))
        logger.info("Text embedding length=%s l2_norm=%.4f", len(query_embedding), norm)
        try:
            n_results = top_k if not filters_applied else 100
            raw = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["distances", "documents", "metadatas"],
            )
        except Exception as raw_error:
            logger.exception("Failed to query Chroma directly (text): %s", raw_error)
            return "❌ Erro ao consultar o banco vetorial."
        raw_documents = raw.get("documents", [[]])[0]
        raw_distances = raw.get("distances", [[]])[0]
        raw_metadatas = raw.get("metadatas", [[]])[0]
        logger.info("Chroma returned %s raw document(s) for text query", len(raw_documents))
        candidates = list(zip(raw_documents, raw_distances, raw_metadatas))

        fallback_used = False
        if filters_applied:
            matched, remainder = _filter_metadata_results(
                candidates,
                sex,
                min_age,
                max_age,
            )
            if len(matched) < top_k and remainder:
                needed = top_k - len(matched)
                matched.extend(remainder[:needed])
                fallback_used = True
            candidates = matched[:top_k]
            logger.info(
                "Text query after filters yielded %s match(es); fallback_used=%s",
                len(candidates),
                fallback_used,
            )
        else:
            candidates = candidates[:top_k]

        if not candidates:
            filter_msg = _summarize_filters(sex, min_age, max_age)
            if filter_msg:
                return (
                    "⚠️ Nenhum resultado encontrado com os filtros aplicados. "
                    f"Filtros: {filter_msg}. Tente ajustar ou remover os filtros."
                )
            logger.warning("No raw documents returned for text query")
            return "⚠️ Nenhuma imagem correspondente foi encontrada." 

        # Formatar resultados como string legível
        result_lines = [f"\n📊 Resultados da busca textual (Texto → Imagens correspondentes):"]
        filter_summary = _summarize_filters(sex, min_age, max_age)
        if filter_summary:
            result_lines.append(f"  ↳ Filtros aplicados: {filter_summary}")
            if fallback_used:
                result_lines.append(
                    "  ↳ Alguns resultados adicionais não atendem totalmente aos filtros; listados para completar o top_k."
                )
        elif fallback_used:
            fallback_used = False

        for i, (doc_path, distance, metadata) in enumerate(candidates, start=1):
            similarity_percent = max(0, (1 - distance / 2) * 100)
            doc_label = metadata.get("isic_id") if isinstance(metadata, dict) else None
            display = doc_label or doc_path
            patient_sex = metadata.get("sex") if isinstance(metadata, dict) else None
            patient_age = metadata.get("age_approx") if isinstance(metadata, dict) else None
            if patient_age is None and isinstance(metadata, dict):
                patient_age = metadata.get("age")
            try:
                patient_age = int(float(patient_age)) if patient_age is not None else None
            except (ValueError, TypeError):
                patient_age = None
            matches_filters = (
                not filters_applied or _metadata_matches_filters(metadata, sex, min_age, max_age)
            )
            extra_bits = []
            if patient_sex:
                extra_bits.append(f"sexo: {patient_sex}")
            if patient_age is not None:
                extra_bits.append(f"idade≈{patient_age}")
            if filters_applied and not matches_filters:
                extra_bits.append("⚠️ fora dos filtros")
            extras = f" ({', '.join(extra_bits)})" if extra_bits else ""
            result_line = (
                f"  #{i:02d} | {similarity_percent:.2f}% de similaridade | {display}{extras}"
            )
            result_lines.append(result_line)
            logger.info(
                "Text result #%s distance=%.4f similarity=%.2f doc=%s",
                i,
                distance,
                similarity_percent,
                display,
            )
        result_lines.append("—" * 60)
        logger.info("Formatted text search response with %s entries", len(result_lines) - 2)

        return "\n".join(result_lines)
    
    except Exception as e:
        logger.exception("Unhandled error during text search: %s", e)
        return f"❌ Erro ao processar consulta textual: {str(e)}"
    
#TODO: Adicionar ferramenta de filtro por padrão morfológico
import os
import warnings
import base64
import json
import logging
import math
import time
import ast
from io import BytesIO
from urllib.parse import urlparse
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from transformers import XLMRobertaTokenizer

from typing import Optional, Any, Iterable, List, Tuple, Dict

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
DEFAULT_VECTORSTORE_DIR = "./vectorstore/chroma_vectorstore"
VECTORSTORE_DIR = os.environ.get("TCGA_VECTORSTORE_DIR", DEFAULT_VECTORSTORE_DIR)
DEFAULT_VECTORSTORE_COLLECTION = "tcga_images_precomputed"
VECTORSTORE_COLLECTION = os.environ.get("VECTORSTORE_COLLECTION", DEFAULT_VECTORSTORE_COLLECTION)
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


def _normalize_filter_sequence(value: Optional[Any]) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Converte entrada arbitrária em lista de strings normalizadas e exibição original."""
    if value is None:
        return None, None

    cleaned = _clean_metadata_value(value, allow_list=True)
    if cleaned is None:
        return None, None

    if isinstance(cleaned, list):
        items = cleaned
    else:
        items = [cleaned]

    normalized: List[str] = []
    display: List[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in normalized:
            continue
        normalized.append(lowered)
        display.append(text)

    if not normalized:
        return None, None
    return normalized, display


def _prepare_filters(
    sex: Optional[str],
    min_age: Optional[Any],
    max_age: Optional[Any],
    *,
    primary_site: Optional[Any] = None,
    tissue_origin: Optional[Any] = None,
    site_of_resection: Optional[Any] = None,
    tissue_type: Optional[Any] = None,
    specimen_type: Optional[Any] = None,
    disease_type: Optional[Any] = None,
    pathologic_stage: Optional[Any] = None,
    ajcc_t: Optional[Any] = None,
    ajcc_n: Optional[Any] = None,
    ajcc_m: Optional[Any] = None,
):
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

    margin = 3
    if normalized_min_age is not None:
        normalized_min_age = max(normalized_min_age - margin, 0)
    if normalized_max_age is not None:
        normalized_max_age = normalized_max_age + margin

    if (
        normalized_min_age is not None
        and normalized_max_age is not None
        and normalized_min_age > normalized_max_age
    ):
        logger.info(
            "Adjusted margin caused min to exceed max; collapsing to single value (min=%s max=%s)",
            normalized_min_age,
            normalized_max_age,
        )
        midpoint = normalized_min_age
        normalized_min_age = midpoint
        normalized_max_age = midpoint

    primary_site_norm, primary_site_display = _normalize_filter_sequence(primary_site)
    tissue_origin_norm, tissue_origin_display = _normalize_filter_sequence(tissue_origin)
    site_resection_norm, site_resection_display = _normalize_filter_sequence(site_of_resection)
    tissue_type_norm, tissue_type_display = _normalize_filter_sequence(tissue_type)
    specimen_type_norm, specimen_type_display = _normalize_filter_sequence(specimen_type)
    disease_type_norm, disease_type_display = _normalize_filter_sequence(disease_type)
    pathologic_stage_norm, pathologic_stage_display = _normalize_filter_sequence(pathologic_stage)
    ajcc_t_norm, ajcc_t_display = _normalize_filter_sequence(ajcc_t)
    ajcc_n_norm, ajcc_n_display = _normalize_filter_sequence(ajcc_n)
    ajcc_m_norm, ajcc_m_display = _normalize_filter_sequence(ajcc_m)

    normalized_filters = {
        "sex": normalized_sex,
        "min_age": normalized_min_age,
        "max_age": normalized_max_age,
        "primary_site": primary_site_norm,
        "tissue_origin": tissue_origin_norm,
        "site_of_resection": site_resection_norm,
        "tissue_type": tissue_type_norm,
        "specimen_type": specimen_type_norm,
        "disease_type": disease_type_norm,
        "pathologic_stage": pathologic_stage_norm,
        "ajcc_t": ajcc_t_norm,
        "ajcc_n": ajcc_n_norm,
        "ajcc_m": ajcc_m_norm,
    }

    display_filters = {
        "sex": sex,
        "min_age": min_age,
        "max_age": max_age,
        "primary_site": primary_site_display,
        "tissue_origin": tissue_origin_display,
        "site_of_resection": site_resection_display,
        "tissue_type": tissue_type_display,
        "specimen_type": specimen_type_display,
        "disease_type": disease_type_display,
        "pathologic_stage": pathologic_stage_display,
        "ajcc_t": ajcc_t_display,
        "ajcc_n": ajcc_n_display,
        "ajcc_m": ajcc_m_display,
    }

    return normalized_filters, display_filters


MetadataResult = Tuple[str, float, dict]


FILTER_FIELD_MAP: Dict[str, Dict[str, Any]] = {
    "primary_site": {
        "keys": ("primary_site.project", "primary_site"),
        "allow_partial": True,
    },
    "tissue_origin": {
        "keys": ("tissue_or_organ_of_origin.diagnoses",),
        "allow_partial": True,
    },
    "site_of_resection": {
        "keys": ("site_of_resection_or_biopsy.diagnoses",),
        "allow_partial": True,
    },
    "tissue_type": {
        "keys": ("tissue_type.samples",),
        "allow_partial": False,
    },
    "specimen_type": {
        "keys": ("specimen_type.samples",),
        "allow_partial": False,
    },
    "disease_type": {
        "keys": ("disease_type.project", "disease_type"),
        "allow_partial": True,
    },
    "pathologic_stage": {
        "keys": ("ajcc_pathologic_stage.diagnoses",),
        "allow_partial": True,
    },
    "ajcc_t": {
        "keys": ("ajcc_pathologic_t.diagnoses",),
        "allow_partial": True,
    },
    "ajcc_n": {
        "keys": ("ajcc_pathologic_n.diagnoses",),
        "allow_partial": True,
    },
    "ajcc_m": {
        "keys": ("ajcc_pathologic_m.diagnoses",),
        "allow_partial": True,
    },
}


FILTER_SUMMARY_LABELS: Dict[str, str] = {
    "primary_site": "local primário",
    "tissue_origin": "tecido/órgão de origem",
    "site_of_resection": "sítio de ressecção/biopsia",
    "tissue_type": "tipo de tecido",
    "specimen_type": "tipo de amostra",
    "disease_type": "tipo de doença",
    "pathologic_stage": "estágio patológico",
    "ajcc_t": "AJCC T",
    "ajcc_n": "AJCC N",
    "ajcc_m": "AJCC M",
}


def _derive_basename_and_ext(doc_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Retorna o nome-base e extensão (sem ponto) a partir de um caminho."""
    if not doc_path:
        return None, None

    normalized = str(doc_path).replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1]
    if not basename:
        return None, None

    if "." in basename:
        stem, ext = basename.rsplit(".", 1)
        return stem or basename, ext.lower()
    return basename, None


def _resolve_media_info(
    metadata: Optional[dict],
    doc_path: Optional[str],
) -> Tuple[Optional[str], Optional[str], str, Optional[str], Optional[str]]:
    """Determina slug de imagem, extensão, rótulo de exibição e códigos auxiliares."""
    safe_metadata = metadata if isinstance(metadata, dict) else {}

    candidate_paths = [
        safe_metadata.get("resolved_image_path"),
        safe_metadata.get("image_path"),
        safe_metadata.get("image_filename"),
        doc_path,
    ]

    image_slug: Optional[str] = None
    image_ext: Optional[str] = None
    for candidate in candidate_paths:
        stem, ext = _derive_basename_and_ext(candidate)
        if stem:
            image_slug, image_ext = stem, ext
            break

    display_candidates = [
        safe_metadata.get("image_case_code"),
        safe_metadata.get("image_slide_code"),
        safe_metadata.get("case_id"),
        image_slug,
        doc_path,
    ]
    display_label = next(
        (str(value).strip() for value in display_candidates if value),
        image_slug or "resultado",
    )

    document_path = safe_metadata.get("resolved_image_path") or doc_path
    case_code = safe_metadata.get("image_case_code")
    slide_code = safe_metadata.get("image_slide_code")
    return image_slug, image_ext, display_label, case_code, slide_code


def _metadata_field_matches(
    metadata: dict,
    field: str,
    filter_values: Optional[List[str]],
) -> bool:
    if not filter_values:
        return True

    config = FILTER_FIELD_MAP.get(field)
    if not config:
        return True

    keys: Tuple[str, ...] = config.get("keys", tuple())
    allow_partial: bool = bool(config.get("allow_partial", False))

    for key in keys:
        raw = metadata.get(key)
        if raw is None:
            continue
        cleaned = _clean_metadata_value(raw, allow_list=True)
        if cleaned is None:
            continue
        if isinstance(cleaned, list):
            candidates = [str(item).strip().lower() for item in cleaned if str(item).strip()]
        else:
            candidates = [str(cleaned).strip().lower()]
        if not candidates:
            continue
        for candidate in candidates:
            for token in filter_values:
                if allow_partial:
                    if token in candidate:
                        return True
                else:
                    if candidate == token:
                        return True
    return False


def _metadata_matches_filters(
    metadata: Optional[dict],
    filters: Dict[str, Any],
) -> bool:
    """Retorna True se os metadados atendem aos filtros solicitados."""
    metadata = metadata or {}

    def _pick_canonical_value(keys: Tuple[str, ...]) -> Tuple[Optional[str], Optional[Any]]:
        for key in keys:
            raw = metadata.get(key)
            if raw is None:
                continue
            cleaned = _clean_metadata_value(raw)
            if cleaned is None:
                continue
            return key, cleaned
        return None, None

    sex = filters.get("sex")
    min_age = filters.get("min_age")
    max_age = filters.get("max_age")

    sex_keys = ("gender.demographic", "gender", "sex")
    _, patient_sex_raw = _pick_canonical_value(sex_keys)
    patient_sex: Optional[str] = None
    if patient_sex_raw is not None:
        normalized_candidate = str(patient_sex_raw).strip().lower()
        if normalized_candidate in {"male", "female"}:
            patient_sex = normalized_candidate
        else:
            patient_sex = _normalize_sex_filter(str(patient_sex_raw))

    age_keys = (
        "age_at_earliest_diagnosis_in_years.diagnoses.xena_derived",
        "age_at_diagnosis.diagnoses",
        "age_at_index.demographic",
    )
    patient_age: Optional[int] = None
    for key in age_keys:
        raw_value = metadata.get(key)
        if raw_value is None:
            continue
        coerced = _coerce_float(raw_value)
        if coerced is None:
            continue
        patient_age = int(round(coerced))
        break

    if sex is not None:
        if patient_sex is None or patient_sex != sex:
            return False

    if min_age is not None:
        if patient_age is None or patient_age < min_age:
            return False

    if max_age is not None:
        if patient_age is None or patient_age > max_age:
            return False

    for field in FILTER_FIELD_MAP.keys():
        filter_values = filters.get(field)
        if filter_values and not _metadata_field_matches(metadata, field, filter_values):
            return False

    return True


def _filter_metadata_results(
    candidates: Iterable[MetadataResult],
    filters: Dict[str, Any],
) -> Tuple[List[MetadataResult], List[MetadataResult]]:
    """Separa resultados que atendem aos filtros e os demais."""
    matched: List[MetadataResult] = []
    remainder: List[MetadataResult] = []

    for doc_path, distance, metadata in candidates:
        if _metadata_matches_filters(metadata, filters):
            matched.append((doc_path, distance, metadata))
        else:
            remainder.append((doc_path, distance, metadata))

    return matched, remainder


def _summarize_filters(filters: Dict[str, Any], display_filters: Dict[str, Any]) -> Optional[str]:
    """Gera texto breve com os filtros aplicados."""
    parts = []
    sex = filters.get("sex")
    min_age = filters.get("min_age")
    max_age = filters.get("max_age")

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

    for field, label in FILTER_SUMMARY_LABELS.items():
        normalized_values = filters.get(field)
        if not normalized_values:
            continue
        display_values = display_filters.get(field) or normalized_values
        if isinstance(display_values, list):
            formatted = ", ".join(str(item) for item in display_values)
        else:
            formatted = str(display_values)
        if formatted:
            parts.append(f"{label}: {formatted}")

    return ", ".join(parts) if parts else None


_METADATA_EMPTY_VALUES = {
    "",
    "not reported",
    "not applicable",
    "unknown",
    "pending",
    "na",
    "n/a",
    "nan",
    "none",
    "unspecified",
}


def _parse_sequence_literal(value: str) -> List[str]:
    """Tenta extrair lista de uma representação textual simples."""
    stripped = value.strip()
    if not stripped:
        return []
    try:
        parsed = ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        parsed = None
    items: List[str] = []
    if isinstance(parsed, (list, tuple, set)):
        for item in parsed:
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1]
        for chunk in inner.split(","):
            text = chunk.strip(" '\"")
            if text:
                items.append(text)
        return items
    if ";" in stripped:
        for chunk in stripped.split(";"):
            text = chunk.strip()
            if text:
                items.append(text)
        return items
    if "," in stripped:
        for chunk in stripped.split(","):
            text = chunk.strip()
            if text:
                items.append(text)
        return items
    return [stripped]


def _clean_metadata_value(value: Optional[Any], *, allow_list: bool = False) -> Optional[Any]:
    """Normaliza valores de metadados removendo placeholders e espaços extra."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
        except TypeError:
            return None
        return int(value) if isinstance(value, float) and value.is_integer() else value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        lowered = stripped.lower()
        if lowered in _METADATA_EMPTY_VALUES:
            return None
        if allow_list:
            items = _parse_sequence_literal(stripped)
            unique_items = []
            for item in items:
                lowered_item = item.lower()
                if lowered_item in _METADATA_EMPTY_VALUES:
                    continue
                if item not in unique_items:
                    unique_items.append(item)
            return unique_items or None
        return stripped
    if isinstance(value, (list, tuple, set)):
        items: List[str] = []
        for item in value:
            normalized = _clean_metadata_value(item)
            if normalized is None:
                continue
            if isinstance(normalized, (int, float)):
                items.append(str(int(normalized)) if isinstance(normalized, float) and normalized.is_integer() else str(normalized))
            else:
                items.append(str(normalized))
        if not items:
            return None
        if allow_list:
            unique_items: List[str] = []
            for item in items:
                if item not in unique_items:
                    unique_items.append(item)
            return unique_items
        return items[0]
    text = str(value).strip()
    if not text:
        return None
    if allow_list:
        return _parse_sequence_literal(text)
    return text


def _coerce_float(value: Optional[Any]) -> Optional[float]:
    """Converte valor arbitrário em float, se possível."""
    cleaned = _clean_metadata_value(value)
    if cleaned is None:
        return None
    if isinstance(cleaned, bool):
        return None
    if isinstance(cleaned, (int, float)):
        number = float(cleaned)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(cleaned, str):
        try:
            number = float(cleaned)
        except ValueError:
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    return None


def _build_structured_entry(
    rank: int,
    similarity_percent: float,
    image_slug: Optional[str],
    display_label: str,
    doc_path: Optional[str],
    metadata: Optional[dict],
    matches_filters: bool,
    patient_sex: Optional[str] = None,
    patient_age: Optional[Any] = None,
    image_ext: Optional[str] = None,
    case_code: Optional[str] = None,
    slide_code: Optional[str] = None,
    resolved_path: Optional[str] = None,
) -> dict:
    """Gera um dicionário serializável com informações essenciais do resultado."""
    safe_metadata = metadata if isinstance(metadata, dict) else {}
    def pick_value(keys: Iterable[str]) -> Optional[str]:
        for key in keys:
            if key not in safe_metadata:
                continue
            candidate = _clean_metadata_value(safe_metadata.get(key))
            if candidate is None:
                continue
            if isinstance(candidate, (int, float)):
                return str(int(candidate)) if isinstance(candidate, float) and candidate.is_integer() else str(candidate)
            if isinstance(candidate, str):
                return candidate
        return None

    def pick_list(keys: Iterable[str]) -> Optional[List[str]]:
        for key in keys:
            if key not in safe_metadata:
                continue
            candidate = _clean_metadata_value(safe_metadata.get(key), allow_list=True)
            if candidate:
                return list(candidate)
        return None

    def pick_number(keys: Iterable[str]) -> Optional[float]:
        for key in keys:
            if key not in safe_metadata:
                continue
            number = _coerce_float(safe_metadata.get(key))
            if number is not None:
                return number
        return None

    diagnosis_primary = pick_value(("primary_diagnosis.diagnoses", "diagnosis_1"))
    diagnosis_secondary = pick_value(("diagnosis_2",))
    diagnosis_tertiary = pick_value(("diagnosis_3",))
    anatom_general = pick_value((
        "anatom_site_general",
        "tissue_or_organ_of_origin.diagnoses",
        "primary_site.project",
        "primary_site",
    ))
    anatom_special = pick_value((
        "anatom_site_special",
        "site_of_resection_or_biopsy.diagnoses",
    ))
    pathologic_stage = pick_value(("ajcc_pathologic_stage.diagnoses",))
    ajcc_pathologic_t = pick_value(("ajcc_pathologic_t.diagnoses",))
    ajcc_pathologic_n = pick_value(("ajcc_pathologic_n.diagnoses",))
    ajcc_pathologic_m = pick_value(("ajcc_pathologic_m.diagnoses",))
    tissue_origin = pick_value(("tissue_or_organ_of_origin.diagnoses",))
    site_resection = pick_value(("site_of_resection_or_biopsy.diagnoses",))
    morphology = pick_value(("morphology.diagnoses",))
    tumor_grade = pick_value(("tumor_grade.diagnoses",))
    classification_of_tumor = pick_value(("classification_of_tumor.diagnoses",))
    last_known_status = pick_value(("last_known_disease_status.diagnoses",))
    primary_site = pick_value(("primary_site.project", "primary_site"))
    disease_type_list = pick_list(("disease_type.project",))
    disease_type = ", ".join(disease_type_list) if disease_type_list else pick_value(("disease_type",))
    vital_status = pick_value(("vital_status.demographic",))
    race = pick_value(("race.demographic",))
    ethnicity = pick_value(("ethnicity.demographic",))
    tissue_type = pick_value(("tissue_type.samples",))
    specimen_type = pick_value(("specimen_type.samples",))
    treatment_types = pick_list(("treatment_type.treatments.diagnoses",))
    age_at_index_years = pick_number(("age_at_index.demographic",))
    age_at_diagnosis_years = pick_number((
        "age_at_earliest_diagnosis_in_years.diagnoses.xena_derived",
        "age_at_diagnosis.diagnoses",
    ))
    days_to_last_follow_up = pick_number(("days_to_last_follow_up.diagnoses",))
    days_to_death = pick_number(("days_to_death.demographic",))
    resolved_doc_path = resolved_path or safe_metadata.get("resolved_image_path") or doc_path
    case_code = case_code or safe_metadata.get("image_case_code")
    slide_code = slide_code or safe_metadata.get("image_slide_code")
    if not patient_sex:
        # Prefer canonical metadata naming for sex first
        fallback_sex = pick_value(("gender.demographic", "gender", "sex"))
        if fallback_sex:
            normalized = fallback_sex.strip().lower()
            patient_sex = normalized if normalized in {"male", "female"} else fallback_sex.strip()
    if patient_age is None:
        # Reuse canonical age fields to infer display value
        canonical_age_candidates = (age_at_diagnosis_years, age_at_index_years)
        for candidate in canonical_age_candidates:
            if candidate is not None:
                patient_age = int(round(candidate))
                break

    def normalize_numeric_display(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        rounded = round(value, 1)
        if abs(rounded - round(rounded)) < 0.05:
            return int(round(rounded))
        return rounded

    age_at_diagnosis_display = normalize_numeric_display(age_at_diagnosis_years)
    days_to_last_follow_up_display = normalize_numeric_display(days_to_last_follow_up)
    days_to_death_display = normalize_numeric_display(days_to_death)

    return {
        "rank": rank,
        "similarity": round(similarity_percent, 2),
        "imageId": image_slug or _derive_basename_and_ext(resolved_doc_path)[0] or display_label,
        "displayLabel": display_label,
        "documentPath": resolved_doc_path,
        "imageExt": image_ext,
        "caseCode": case_code,
        "slideCode": slide_code,
        "sex": patient_sex or safe_metadata.get("gender") or safe_metadata.get("gender.demographic") or safe_metadata.get("sex"),
        "ageApprox": patient_age or safe_metadata.get("age_at_index") or safe_metadata.get("age_approx") or safe_metadata.get("age"),
        "ageAtDiagnosis": age_at_diagnosis_display,
        "diagnosisPrimary": diagnosis_primary,
        "diagnosisSecondary": diagnosis_secondary,
        "diagnosisTertiary": diagnosis_tertiary,
        "anatomSiteGeneral": anatom_general,
        "anatomSiteSpecial": anatom_special,
        "pathologicStage": pathologic_stage,
        "ajccPathologicT": ajcc_pathologic_t,
        "ajccPathologicN": ajcc_pathologic_n,
        "ajccPathologicM": ajcc_pathologic_m,
        "tissueOrOrganOfOrigin": tissue_origin,
        "siteOfResectionOrBiopsy": site_resection,
        "morphology": morphology,
        "tumorGrade": tumor_grade,
        "classificationOfTumor": classification_of_tumor,
        "lastKnownDiseaseStatus": last_known_status,
        "primarySite": primary_site,
        "diseaseType": disease_type,
        "vitalStatus": vital_status,
        "race": race,
        "ethnicity": ethnicity,
        "tissueType": tissue_type,
        "specimenType": specimen_type,
        "treatmentTypes": treatment_types,
        "daysToLastFollowUp": days_to_last_follow_up_display,
        "daysToDeath": days_to_death_display,
        "matchedFilters": matches_filters,
    }


def _push_results_to_state(tool_context: Optional[Any], payload: dict) -> None:
    """Atualiza o estado compartilhado com os resultados estruturados."""
    if not TOOL_CONTEXT_AVAILABLE or tool_context is None:
        return

    try:
        state = getattr(tool_context, "state", None)
        if state is None:
            return
        state["searchResults"] = payload
    except Exception as exc:
        logger.exception("Failed to push structured search results to state: %s", exc)

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
            collection_name=VECTORSTORE_COLLECTION,
        )
        logger.info("Vectorstore loaded successfully with collection '%s'", VECTORSTORE_COLLECTION)
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
    primary_site: Optional[Any] = None,
    tissue_origin: Optional[Any] = None,
    site_of_resection: Optional[Any] = None,
    tissue_type: Optional[Any] = None,
    specimen_type: Optional[Any] = None,
    disease_type: Optional[Any] = None,
    pathologic_stage: Optional[Any] = None,
    ajcc_t: Optional[Any] = None,
    ajcc_n: Optional[Any] = None,
    ajcc_m: Optional[Any] = None,
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
          sex: Filtra por gênero registrado no dataset ("female" ou "male").
              Aceita variantes em português como "feminino" ou "masculino".
          min_age: Idade mínima aproximada do paciente (inteiro).
                    max_age: Idade máxima aproximada do paciente (inteiro).
                    primary_site: Local anatômico primário do caso (string ou lista).
                    tissue_origin: Tecido ou órgão de origem da amostra.
                    site_of_resection: Sítio de ressecção/biópsia registrado.
                    tissue_type: Tipo de tecido (por exemplo, "Tumor", "Solid Tissue").
                    specimen_type: Tipo de amostra (por exemplo, "Solid Tissue").
                    disease_type: Tipo de doença associado ao caso.
                    pathologic_stage: Estágio patológico AJCC.
                    ajcc_t: Categoria AJCC T.
                    ajcc_n: Categoria AJCC N.
                    ajcc_m: Categoria AJCC M.
        tool_context: Contexto da ferramenta (fornecido automaticamente pelo ADK).
    
    Returns:
        String formatada contendo os resultados da busca, incluindo:
        - Posição do resultado
        - Escala de proximidade vetorial (0 a 100; menor é melhor)
        - Identificador ou descrição da imagem encontrada
    
    Examples:
        >>> search_by_image_query(top_k=3)
        "Resultado #1: 85.230 de proximidade vetorial - TCGA-D7-A4YV-01Z-00-DX1\n..."
    """
    # Recuperar imagem diretamente do contexto
    image_bytes = None
    mime_type = 'image/png'
    normalized_filters, display_filters = _prepare_filters(
        sex,
        min_age,
        max_age,
        primary_site=primary_site,
        tissue_origin=tissue_origin,
        site_of_resection=site_of_resection,
        tissue_type=tissue_type,
        specimen_type=specimen_type,
        disease_type=disease_type,
        pathologic_stage=pathologic_stage,
        ajcc_t=ajcc_t,
        ajcc_n=ajcc_n,
        ajcc_m=ajcc_m,
    )
    filter_keys = [
        "sex",
        "min_age",
        "max_age",
        "primary_site",
        "tissue_origin",
        "site_of_resection",
        "tissue_type",
        "specimen_type",
        "disease_type",
        "pathologic_stage",
        "ajcc_t",
        "ajcc_n",
        "ajcc_m",
    ]
    filters_applied = any(
        (normalized_filters.get(key) is not None)
        if key in {"sex", "min_age", "max_age"}
        else bool(normalized_filters.get(key))
        for key in filter_keys
    )
    visible_filters: Dict[str, Any] = {}
    for key, value in normalized_filters.items():
        if key in {"sex", "min_age", "max_age"}:
            if value is not None:
                visible_filters[key] = value
        elif value:
            visible_filters[key] = value
    logger.info("search_by_image_query top_k=%s filters=%s", top_k, visible_filters)
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
            n_results = top_k if not filters_applied else 300
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
                normalized_filters,
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
            filter_msg = _summarize_filters(normalized_filters, display_filters)
            if filter_msg:
                response = (
                    "⚠️ Nenhum resultado encontrado com os filtros aplicados. "
                    f"Filtros: {filter_msg}. Tente ajustar ou remover os filtros."
                )
                _push_results_to_state(
                    tool_context,
                    {
                        "source": "image",
                        "timestamp": int(time.time()),
                        "filters": {
                            "normalized": normalized_filters,
                            "display": display_filters,
                            "summary": filter_msg,
                            "fallbackUsed": False,
                        },
                        "results": [],
                        "rawText": response,
                    },
                )
                return response
            logger.warning("No raw documents returned for image query")
            response = "⚠️ Nenhuma imagem semelhante foi encontrada."
            _push_results_to_state(
                tool_context,
                {
                    "source": "image",
                    "timestamp": int(time.time()),
                    "filters": {
                        "normalized": normalized_filters,
                        "display": display_filters,
                        "summary": None,
                        "fallbackUsed": False,
                    },
                    "results": [],
                    "rawText": response,
                },
            )
            return response

        # Formatar resultados como string legível
        result_lines = [f"\n📊 Resultados da busca por imagem (Imagem → Imagens semelhantes):"]
        filter_summary = _summarize_filters(normalized_filters, display_filters)
        fallback_flag = fallback_used
        if filter_summary:
            result_lines.append(f"  ↳ Filtros aplicados: {filter_summary}")
            if fallback_used:
                result_lines.append(
                    "  ↳ Alguns resultados adicionais não atendem totalmente aos filtros; listados para completar o top_k."
                )
        elif fallback_used:
            fallback_used = False  # Segurança; não deve ocorrer sem filtros.

        structured_results: List[dict] = []

        for i, (doc_path, distance, metadata) in enumerate(candidates, start=1):
            proximity_percent = max(0.0, float(distance) * 100.0)
            image_slug, image_ext, display, case_code, slide_code = _resolve_media_info(metadata, doc_path)
            resolved_path = metadata.get("resolved_image_path") if isinstance(metadata, dict) else None
            display_str = str(display or image_slug or doc_path or f"resultado_{i:02d}")
            # Use current metadata field names
            patient_sex = metadata.get("gender") if isinstance(metadata, dict) else None
            patient_age = metadata.get("age_at_index") if isinstance(metadata, dict) else None
            if patient_age is None and isinstance(metadata, dict):
                patient_age = metadata.get("age_at_index.demographic") or metadata.get("age_at_diagnosis.diagnoses")
            try:
                patient_age = int(float(patient_age)) if patient_age is not None else None
            except (ValueError, TypeError):
                patient_age = None
            matches_filters = (
                not filters_applied or _metadata_matches_filters(metadata, normalized_filters)
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
                f"  #{i:02d} | {proximity_percent:.3f} de proximidade vetorial | {display_str}{extras}"
            )
            result_lines.append(result_line)
            logger.info(
                "Image result #%s distance=%.4f proximity_percent=%.3f doc=%s",
                i,
                distance,
                proximity_percent,
                display_str,
            )
            structured_results.append(
                _build_structured_entry(
                    rank=i,
                    similarity_percent=proximity_percent,
                    image_slug=image_slug,
                    display_label=display_str,
                    doc_path=doc_path,
                    metadata=metadata if isinstance(metadata, dict) else {},
                    matches_filters=matches_filters,
                    patient_sex=patient_sex,
                    patient_age=patient_age,
                    image_ext=image_ext,
                    case_code=case_code,
                    slide_code=slide_code,
                    resolved_path=resolved_path,
                )
            )
        result_lines.append("  ↳ Escala de proximidade: 0 a 100 (quanto mais próximo de 0, maior a proximidade vetorial).")
        result_lines.append("—" * 60)
        logger.info("Formatted image search response with %s entries", len(result_lines) - 2)

        payload = {
            "source": "image",
            "timestamp": int(time.time()),
            "filters": {
                "normalized": normalized_filters,
                "display": display_filters,
                "summary": filter_summary,
                "fallbackUsed": fallback_flag,
            },
            "results": structured_results,
            "rawText": "\n".join(result_lines),
        }
        _push_results_to_state(tool_context, payload)

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
    primary_site: Optional[Any] = None,
    tissue_origin: Optional[Any] = None,
    site_of_resection: Optional[Any] = None,
    tissue_type: Optional[Any] = None,
    specimen_type: Optional[Any] = None,
    disease_type: Optional[Any] = None,
    pathologic_stage: Optional[Any] = None,
    ajcc_t: Optional[Any] = None,
    ajcc_n: Optional[Any] = None,
    ajcc_m: Optional[Any] = None,
    tool_context = None,
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
          sex: Filtra por gênero registrado no dataset ("female" ou "male").
              Aceita variantes em português como "feminino" ou "masculino".
          min_age: Idade mínima aproximada do paciente (inteiro).
          max_age: Idade máxima aproximada do paciente (inteiro).
          primary_site: Local anatômico primário do caso (string ou lista).
          tissue_origin: Tecido ou órgão de origem da amostra.
          site_of_resection: Sítio de ressecção/biópsia registrado.
          tissue_type: Tipo de tecido (por exemplo, "Tumor", "Solid Tissue").
          specimen_type: Tipo de amostra (por exemplo, "Solid Tissue").
          disease_type: Tipo de doença associado ao caso.
          pathologic_stage: Estágio patológico AJCC.
          ajcc_t: Categoria AJCC T.
          ajcc_n: Categoria AJCC N.
          ajcc_m: Categoria AJCC M.
    
    Returns:
        String formatada contendo os resultados da busca, incluindo:
        - Posição do resultado
        - Escala de proximidade vetorial (0 a 100; menor é melhor)
        - Identificador ou descrição da imagem encontrada
        
    Example:
    >>> search_by_text_query("gastric adenocarcinoma with diffuse pattern", top_k=3)
    "Resultado #1: 92.150 de proximidade vetorial - TCGA-D7-A4YV-01Z-00-DX1\n..."
    """
    normalized_filters, display_filters = _prepare_filters(
        sex,
        min_age,
        max_age,
        primary_site=primary_site,
        tissue_origin=tissue_origin,
        site_of_resection=site_of_resection,
        tissue_type=tissue_type,
        specimen_type=specimen_type,
        disease_type=disease_type,
        pathologic_stage=pathologic_stage,
        ajcc_t=ajcc_t,
        ajcc_n=ajcc_n,
        ajcc_m=ajcc_m,
    )
    filter_keys = [
        "sex",
        "min_age",
        "max_age",
        "primary_site",
        "tissue_origin",
        "site_of_resection",
        "tissue_type",
        "specimen_type",
        "disease_type",
        "pathologic_stage",
        "ajcc_t",
        "ajcc_n",
        "ajcc_m",
    ]
    filters_applied = any(
        (normalized_filters.get(key) is not None)
        if key in {"sex", "min_age", "max_age"}
        else bool(normalized_filters.get(key))
        for key in filter_keys
    )
    visible_filters: Dict[str, Any] = {}
    for key, value in normalized_filters.items():
        if key in {"sex", "min_age", "max_age"}:
            if value is not None:
                visible_filters[key] = value
        elif value:
            visible_filters[key] = value
    logger.info(
        "search_by_text_query top_k=%s filters=%s query='%s'",
        top_k,
        visible_filters,
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
            n_results = top_k if not filters_applied else 300
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
                normalized_filters,
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
            filter_msg = _summarize_filters(normalized_filters, display_filters)
            if filter_msg:
                response = (
                    "⚠️ Nenhum resultado encontrado com os filtros aplicados. "
                    f"Filtros: {filter_msg}. Tente ajustar ou remover os filtros."
                )
                _push_results_to_state(
                    tool_context,
                    {
                        "source": "text",
                        "timestamp": int(time.time()),
                        "filters": {
                            "normalized": normalized_filters,
                            "display": display_filters,
                            "summary": filter_msg,
                            "fallbackUsed": False,
                        },
                        "results": [],
                        "rawText": response,
                        "query": text_query,
                    },
                )
                return response
            logger.warning("No raw documents returned for text query")
            response = "⚠️ Nenhuma imagem correspondente foi encontrada."
            _push_results_to_state(
                tool_context,
                {
                    "source": "text",
                    "timestamp": int(time.time()),
                    "filters": {
                        "normalized": normalized_filters,
                        "display": display_filters,
                        "summary": None,
                        "fallbackUsed": False,
                    },
                    "results": [],
                    "rawText": response,
                    "query": text_query,
                },
            )
            return response

        # Formatar resultados como string legível
        result_lines = [f"\n📊 Resultados da busca textual (Texto → Imagens correspondentes):"]
        filter_summary = _summarize_filters(normalized_filters, display_filters)
        fallback_flag = fallback_used
        if filter_summary:
            result_lines.append(f"  ↳ Filtros aplicados: {filter_summary}")
            if fallback_used:
                result_lines.append(
                    "  ↳ Alguns resultados adicionais não atendem totalmente aos filtros; listados para completar o top_k."
                )
        elif fallback_used:
            fallback_used = False

        structured_results: List[dict] = []

        for i, (doc_path, distance, metadata) in enumerate(candidates, start=1):
            proximity_percent = max(0.0, float(distance) * 100.0)
            image_slug, image_ext, display, case_code, slide_code = _resolve_media_info(metadata, doc_path)
            resolved_path = metadata.get("resolved_image_path") if isinstance(metadata, dict) else None
            display_str = str(display or image_slug or doc_path or f"resultado_{i:02d}")
            # Use current metadata field names
            patient_sex = metadata.get("gender") if isinstance(metadata, dict) else None
            patient_age = metadata.get("age_at_index") if isinstance(metadata, dict) else None
            if patient_age is None and isinstance(metadata, dict):
                patient_age = metadata.get("age_at_index.demographic") or metadata.get("age_at_diagnosis.diagnoses")
            try:
                patient_age = int(float(patient_age)) if patient_age is not None else None
            except (ValueError, TypeError):
                patient_age = None
            matches_filters = (
                not filters_applied or _metadata_matches_filters(metadata, normalized_filters)
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
                f"  #{i:02d} | {proximity_percent:.3f} de proximidade vetorial | {display_str}{extras}"
            )
            result_lines.append(result_line)
            logger.info(
                "Text result #%s distance=%.4f proximity_percent=%.3f doc=%s",
                i,
                distance,
                proximity_percent,
                display_str,
            )
            structured_results.append(
                _build_structured_entry(
                    rank=i,
                    similarity_percent=proximity_percent,
                    image_slug=image_slug,
                    display_label=display_str,
                    doc_path=doc_path,
                    metadata=metadata if isinstance(metadata, dict) else {},
                    matches_filters=matches_filters,
                    patient_sex=patient_sex,
                    patient_age=patient_age,
                    image_ext=image_ext,
                    case_code=case_code,
                    slide_code=slide_code,
                    resolved_path=resolved_path,
                )
            )
        result_lines.append("  ↳ Escala de proximidade: 0 a 100 (quanto mais próximo de 0, maior a proximidade vetorial).")
        result_lines.append("—" * 60)
        logger.info("Formatted text search response with %s entries", len(result_lines) - 2)

        payload = {
            "source": "text",
            "timestamp": int(time.time()),
            "filters": {
                "normalized": normalized_filters,
                "display": display_filters,
                "summary": filter_summary,
                "fallbackUsed": fallback_flag,
            },
            "results": structured_results,
            "rawText": "\n".join(result_lines),
            "query": text_query,
        }
        _push_results_to_state(tool_context, payload)

        return "\n".join(result_lines)
    
    except Exception as e:
        logger.exception("Unhandled error during text search: %s", e)
        return f"❌ Erro ao processar consulta textual: {str(e)}"
    
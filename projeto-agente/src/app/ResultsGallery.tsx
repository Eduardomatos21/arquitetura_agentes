'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

export type SearchResultItem = {
  rank: number;
  similarity: number;
  imageId: string;
  displayLabel?: string | null;
  imageExt?: string | null;
  caseCode?: string | null;
  slideCode?: string | null;
  documentPath?: string;
  sex?: string | null;
  ageApprox?: number | string | null;
  ageAtDiagnosis?: number | string | null;
  diagnosisPrimary?: string | null;
  diagnosisSecondary?: string | null;
  diagnosisTertiary?: string | null;
  anatomSiteGeneral?: string | null;
  anatomSiteSpecial?: string | null;
  pathologicStage?: string | null;
  ajccPathologicT?: string | null;
  ajccPathologicN?: string | null;
  ajccPathologicM?: string | null;
  tissueOrOrganOfOrigin?: string | null;
  siteOfResectionOrBiopsy?: string | null;
  morphology?: string | null;
  tumorGrade?: string | null;
  classificationOfTumor?: string | null;
  lastKnownDiseaseStatus?: string | null;
  primarySite?: string | null;
  diseaseType?: string | null;
  vitalStatus?: string | null;
  race?: string | null;
  ethnicity?: string | null;
  tissueType?: string | null;
  specimenType?: string | null;
  treatmentTypes?: string[] | null;
  daysToLastFollowUp?: number | string | null;
  daysToDeath?: number | string | null;
  matchedFilters?: boolean;
};

export type SharedSearchResults = {
  source: 'image' | 'text';
  timestamp: number;
  filters?: {
    normalized?: Record<string, unknown> | null;
    display?: Record<string, unknown> | null;
    summary?: string | null;
    fallbackUsed?: boolean;
  } | null;
  results: SearchResultItem[];
  rawText?: string;
  query?: string;
};

type ResultsGalleryProps = {
  data: SharedSearchResults;
  themeColor: string;
};

type ModalState = {
  result: SearchResultItem;
  imageUrl: string;
} | null;

function MagnifierIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <circle cx="11" cy="11" r="7" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function toImageId(sourceId: string): string {
  const normalized = sourceId.replace(/\\/g, '/');
  const fileName = normalized.split('/').pop() ?? normalized;
  const withoutExtension = fileName.replace(/\.[^.]+$/, '');
  return withoutExtension;
}

function buildImageUrl(result: SearchResultItem): string {
  const primarySlug = result.imageId ? toImageId(result.imageId) : null;
  const pathSlug = result.documentPath ? toImageId(result.documentPath) : null;
  const labelSlug = result.displayLabel ? toImageId(result.displayLabel) : null;
  const slugSource = primarySlug || pathSlug || labelSlug || `result-${result.rank}`;
  return `/api/images/${encodeURIComponent(slugSource)}`;
}

function formatSimilarity(value: number): string {
  if (Number.isNaN(value)) {
    return '–';
  }
  return value.toFixed(3);
}

function formatSex(value?: string | null): string | null {
  if (!value) {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  if (normalized === 'male') {
    return 'Masculino';
  }
  if (normalized === 'female') {
    return 'Feminino';
  }
  return value.trim();
}

function formatAge(value?: number | string | null): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  const numeric = typeof value === 'string' ? Number(value) : value;
  if (Number.isNaN(numeric) || !Number.isFinite(numeric)) {
    return null;
  }
  const rounded = Math.round(numeric * 10) / 10;
  return Number.isInteger(rounded) ? `${rounded}` : `${rounded.toFixed(1)}`;
}

function toTitleCase(value?: string | null): string | null {
  if (!value) {
    return null;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  return trimmed
    .split(/\s+/)
    .map((word) => {
      if (!word) {
        return word;
      }
      const isAllCaps = word === word.toUpperCase();
      if (isAllCaps) {
        return word;
      }
      return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    })
    .join(' ');
}

function sanitizeList(values?: string[] | null): string[] {
  if (!values) {
    return [];
  }
  const emptyTokens = new Set(['not reported', 'not applicable', 'unknown', 'none', 'na', 'n/a']);
  const seen = new Set<string>();
  return values
    .map((item) => item?.trim())
    .filter((item): item is string => Boolean(item))
    .filter((item) => !emptyTokens.has(item.toLowerCase()))
    .filter((item) => {
      if (seen.has(item)) {
        return false;
      }
      seen.add(item);
      return true;
    });
}

function formatDays(value?: number | string | null): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  const numeric = typeof value === 'string' ? Number(value) : value;
  if (Number.isNaN(numeric) || !Number.isFinite(numeric)) {
    return null;
  }
  return `${Math.round(numeric)}`;
}

function ResultCard({
  result,
  themeColor,
  onSelect,
}: {
  result: SearchResultItem;
  themeColor: string;
  onSelect: (value: { result: SearchResultItem; imageUrl: string }) => void;
}) {
  const [imageError, setImageError] = useState(false);
  const imageUrl = useMemo(
    () => buildImageUrl(result),
    [result.imageId, result.displayLabel, result.documentPath, result.rank],
  );
  const diagnosis = result.diagnosisPrimary || result.diagnosisSecondary || result.diagnosisTertiary;
  const anatomy = result.anatomSiteSpecial || result.anatomSiteGeneral;
  const cardAccent = `${themeColor}33`;
  const identifier = result.caseCode || result.displayLabel || result.imageId;
  const formattedSex = formatSex(result.sex);
  const ageValue = result.ageApprox ?? result.ageAtDiagnosis;
  const ageDisplay = formatAge(ageValue);
  const stageDisplay = result.pathologicStage;
  const primarySiteDisplay =
    toTitleCase(result.primarySite) ??
    toTitleCase(result.tissueOrOrganOfOrigin) ??
    result.primarySite ??
    result.tissueOrOrganOfOrigin ??
    null;
  const secondaryIdentifier = (() => {
    if (result.caseCode && result.displayLabel && result.displayLabel !== result.caseCode) {
      return result.displayLabel;
    }
    if (result.slideCode && result.slideCode !== result.caseCode) {
      return result.slideCode;
    }
    return null;
  })();

  return (
    <button
      type="button"
      onClick={() => onSelect({ result, imageUrl })}
      className="group flex flex-col rounded-2xl border border-white/15 bg-white/10 p-3 text-left shadow-md transition hover:-translate-y-1 hover:shadow-xl focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 sm:p-4"
      style={{ borderColor: cardAccent }}
    >
      <div className="relative mb-3 overflow-hidden rounded-xl bg-black/30" style={{ aspectRatio: '4 / 3' }}>
        {!imageError ? (
          <img
            src={imageUrl}
            alt={`Lâmina ${identifier}`}
            className="h-full w-full object-cover transition group-hover:scale-105"
            onError={() => setImageError(true)}
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center bg-slate-800 text-sm text-white/70">
            Imagem indisponível
          </div>
        )}
        <span className="absolute left-3 top-3 rounded-full bg-black/70 px-2 py-1 text-xs font-semibold text-white">
          #{result.rank.toString().padStart(2, '0')}
        </span>
        <span
          className="absolute right-3 top-3 rounded-full bg-white/80 px-2 py-1 text-xs font-semibold text-slate-900"
          title="Proximidade vetorial"
        >
          {formatSimilarity(result.similarity)}
        </span>
      </div>
      <div className="flex flex-col gap-1 text-white/90">
        <p className="text-sm font-semibold uppercase tracking-wide text-white/90">
          {identifier}
        </p>
        {secondaryIdentifier && (
          <p className="text-xs uppercase tracking-wide text-white/60">{secondaryIdentifier}</p>
        )}
        {diagnosis && <p className="text-sm text-white/80">{diagnosis}</p>}
        <div className="flex flex-wrap gap-2 text-xs text-white/70">
          {result.slideCode && (
            <span className="rounded-full bg-white/10 px-2 py-1">{`Slide: ${result.slideCode}`}</span>
          )}
          {formattedSex && (
            <span className="rounded-full bg-white/10 px-2 py-1">{`Gênero: ${formattedSex}`}</span>
          )}
          {ageDisplay && (
            <span className="rounded-full bg-white/10 px-2 py-1">{`Idade≈ ${ageDisplay}`}</span>
          )}
          {anatomy && (
            <span className="rounded-full bg-white/10 px-2 py-1">{anatomy}</span>
          )}
          {stageDisplay && (
            <span className="rounded-full bg-white/10 px-2 py-1">{`Estágio: ${stageDisplay}`}</span>
          )}
          {primarySiteDisplay && (
            <span className="rounded-full bg-white/10 px-2 py-1">{primarySiteDisplay}</span>
          )}
          {result.matchedFilters === false && (
            <span className="rounded-full bg-amber-500/70 px-2 py-1 text-slate-900">Fora dos filtros</span>
          )}
        </div>
      </div>
    </button>
  );
}

function ImagePreviewModal({
  modal,
  onClose,
}: {
  modal: ModalState;
  onClose: () => void;
}) {
  const [imageError, setImageError] = useState(false);
  const [zoomed, setZoomed] = useState(false);
  const modalContainerRef = useRef<HTMLDivElement | null>(null);
  const zoomContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!modal) {
      return;
    }

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (zoomed) {
          setZoomed(false);
        } else {
          onClose();
        }
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [modal, zoomed, onClose]);

  useEffect(() => {
    if (!modal) {
      return;
    }

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node;

      if (zoomed) {
        if (zoomContainerRef.current && !zoomContainerRef.current.contains(target)) {
          setZoomed(false);
        }
        return;
      }

      if (modalContainerRef.current && !modalContainerRef.current.contains(target)) {
        onClose();
      }
    };

    window.addEventListener('pointerdown', handlePointerDown);
    return () => window.removeEventListener('pointerdown', handlePointerDown);
  }, [modal, zoomed, onClose]);

  useEffect(() => {
    if (!modal) {
      return;
    }

    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = originalOverflow;
    };
  }, [modal]);

  if (!modal) {
    return null;
  }

  const portalTarget = typeof document !== 'undefined' ? document.body : null;
  if (!portalTarget) {
    return null;
  }

  const { result, imageUrl } = modal;
  const diagnosis = result.diagnosisPrimary || result.diagnosisSecondary || result.diagnosisTertiary;
  const anatomy = result.anatomSiteSpecial || result.anatomSiteGeneral;
  const identifier = result.caseCode || result.displayLabel || result.imageId;
  const secondaryIdentifier = (() => {
    if (result.caseCode && result.displayLabel && result.displayLabel !== result.caseCode) {
      return result.displayLabel;
    }
    if (result.slideCode && result.slideCode !== result.caseCode) {
      return result.slideCode;
    }
    return null;
  })();
  const formattedSex = formatSex(result.sex);
  const ageApproxDisplay = formatAge(result.ageApprox);
  const ageAtDiagnosisDisplay = formatAge(result.ageAtDiagnosis);
  const raceDisplay = toTitleCase(result.race) ?? result.race ?? null;
  const ethnicityDisplay = toTitleCase(result.ethnicity) ?? result.ethnicity ?? null;
  const vitalStatusDisplay = toTitleCase(result.vitalStatus) ?? result.vitalStatus ?? null;
  const primarySiteDisplay =
    toTitleCase(result.primarySite) ??
    toTitleCase(result.tissueOrOrganOfOrigin) ??
    result.primarySite ??
    result.tissueOrOrganOfOrigin ??
    null;
  const tissueOriginDisplay = toTitleCase(result.tissueOrOrganOfOrigin) ?? result.tissueOrOrganOfOrigin ?? null;
  const siteResectionDisplay = toTitleCase(result.siteOfResectionOrBiopsy) ?? result.siteOfResectionOrBiopsy ?? null;
  const diseaseTypeDisplay = toTitleCase(result.diseaseType) ?? result.diseaseType ?? null;
  const tissueTypeDisplay = toTitleCase(result.tissueType) ?? result.tissueType ?? null;
  const specimenTypeDisplay = toTitleCase(result.specimenType) ?? result.specimenType ?? null;
  const tumorGradeDisplay = toTitleCase(result.tumorGrade) ?? result.tumorGrade ?? null;
  const classificationDisplay = toTitleCase(result.classificationOfTumor) ?? result.classificationOfTumor ?? null;
  const lastKnownStatusDisplay = toTitleCase(result.lastKnownDiseaseStatus) ?? result.lastKnownDiseaseStatus ?? null;
  const morphologyDisplay = result.morphology ?? null;
  const daysToLastFollowUpDisplay = formatDays(result.daysToLastFollowUp);
  const daysToDeathDisplay = formatDays(result.daysToDeath);
  const treatmentTypes = sanitizeList(result.treatmentTypes);

  const toDisplay = (value: string | number | null | undefined) => {
    if (value === null || value === undefined) {
      return '—';
    }
    if (typeof value === 'number') {
      return Number.isInteger(value) ? `${value}` : value.toFixed(1);
    }
    const trimmed = value.trim();
    return trimmed ? trimmed : '—';
  };

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex min-h-screen items-center justify-center bg-black/60 p-4 backdrop-blur-sm sm:p-6"
      onClick={(e) => {
        if (e.currentTarget === e.target) {
          if (zoomed) {
            setZoomed(false);
          } else {
            onClose();
          }
        }
      }}
    >
      <div
        ref={modalContainerRef}
        className="relative mx-auto flex min-h-0 w-full max-w-[calc(100dvw-2rem)] flex-col overflow-hidden rounded-3xl bg-slate-950/95 text-white shadow-2xl ring-1 ring-white/10 sm:max-w-[calc(100dvw-3rem)] md:max-w-4xl max-h-[calc(100dvh-2rem)] sm:max-h-[calc(100dvh-3rem)]"
      >
        <button
          type="button"
          onClick={() => {
            setZoomed(false);
            onClose();
          }}
          className="absolute right-4 top-4 z-20 rounded-full bg-white/15 px-3 py-2 text-sm font-semibold text-white hover:bg-white/25 sm:px-4"
        >
          Fechar
        </button>
        <div className="grid flex-1 min-h-0 gap-5 overflow-y-auto p-5 sm:p-6 md:grid-cols-2 md:gap-6">
          <div className="flex flex-col gap-4 md:gap-5">
            <div className="rounded-2xl bg-white/10 p-4 text-sm text-white/80 sm:p-5">
              <p className="text-xs uppercase tracking-[0.2em] text-white/60">Identificador</p>
              <p className="text-lg font-semibold text-white">{identifier}</p>
              {secondaryIdentifier && (
                <p className="text-xs uppercase tracking-wide text-white/60">{secondaryIdentifier}</p>
              )}
              <p className="text-sm text-white/70">Proximidade vetorial: {formatSimilarity(result.similarity)}</p>
              <p className="text-xs text-white/60">Escala de 0 a 100; valores menores indicam maior proximidade.</p>
              {result.matchedFilters === false && (
                <p className="mt-2 rounded-md bg-amber-500/20 px-3 py-2 text-amber-200">
                  Este resultado foi incluído para completar o top-k, mas não atende a todos os filtros.
                </p>
              )}
            </div>
            <div className="rounded-2xl bg-white/10 p-4 text-sm sm:p-5">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-white/70">Dados demográficos</h3>
              <ul className="mt-2 space-y-1 text-white/80">
                <li>Código do caso: {toDisplay(result.caseCode)}</li>
                <li>Código do slide: {toDisplay(result.slideCode)}</li>
                <li>Gênero: {toDisplay(formattedSex)}</li>
                <li>Idade aproximada: {toDisplay(ageApproxDisplay)}</li>
                <li>Idade no diagnóstico (anos): {toDisplay(ageAtDiagnosisDisplay)}</li>
                <li>Raça: {toDisplay(raceDisplay)}</li>
                <li>Etnia: {toDisplay(ethnicityDisplay)}</li>
                <li>Status vital: {toDisplay(vitalStatusDisplay)}</li>
              </ul>
            </div>
            <div className="rounded-2xl bg-white/10 p-4 text-sm sm:p-5">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-white/70">Diagnóstico e estadiamento</h3>
              <ul className="mt-2 space-y-1 text-white/80">
                <li>Diagnóstico principal: {toDisplay(diagnosis)}</li>
                <li>Diagnóstico secundário: {toDisplay(result.diagnosisSecondary)}</li>
                <li>Diagnóstico terciário: {toDisplay(result.diagnosisTertiary)}</li>
                <li>Estágio patológico: {toDisplay(result.pathologicStage)}</li>
                <li>AJCC T: {toDisplay(result.ajccPathologicT)}</li>
                <li>AJCC N: {toDisplay(result.ajccPathologicN)}</li>
                <li>AJCC M: {toDisplay(result.ajccPathologicM)}</li>
                <li>Grau tumoral: {toDisplay(tumorGradeDisplay)}</li>
                <li>Classificação do tumor: {toDisplay(classificationDisplay)}</li>
                <li>Status da doença: {toDisplay(lastKnownStatusDisplay)}</li>
                <li>Morfologia: {toDisplay(morphologyDisplay)}</li>
                <li>Dias até último acompanhamento: {toDisplay(daysToLastFollowUpDisplay)}</li>
                <li>Dias até óbito: {toDisplay(daysToDeathDisplay)}</li>
              </ul>
            </div>
            <div className="rounded-2xl bg-white/10 p-4 text-sm sm:p-5">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-white/70">Amostra e origem</h3>
              <ul className="mt-2 space-y-1 text-white/80">
                <li>Local primário: {toDisplay(primarySiteDisplay)}</li>
                <li>Tecido/órgão de origem: {toDisplay(tissueOriginDisplay)}</li>
                <li>Sítio de ressecção/biópsia: {toDisplay(siteResectionDisplay)}</li>
                <li>Tipo de tecido: {toDisplay(tissueTypeDisplay)}</li>
                <li>Tipo de amostra: {toDisplay(specimenTypeDisplay)}</li>
                <li>Tipo de doença: {toDisplay(diseaseTypeDisplay)}</li>
                <li>Local anatômico reportado: {toDisplay(anatomy)}</li>
              </ul>
            </div>
            {treatmentTypes.length > 0 && (
              <div className="rounded-2xl bg-white/10 p-4 text-sm sm:p-5">
                <h3 className="text-sm font-semibold uppercase tracking-wide text-white/70">Tratamentos registrados</h3>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-white/80">
                  {treatmentTypes.map((item) => (
                    <li key={item}>{toDisplay(toTitleCase(item) ?? item)}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
          <div className="flex flex-col gap-4 md:gap-5">
            <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-slate-900" style={{ aspectRatio: '4 / 3' }}>
              {!imageError ? (
                <img
                  src={imageUrl}
                  alt={`Pré-visualização da lâmina ${identifier}`}
                  className="h-full w-full object-contain"
                  onError={() => setImageError(true)}
                />
              ) : (
                <div className="flex h-full w-full items-center justify-center text-sm text-white/70">
                  Pré-visualização indisponível
                </div>
              )}
              {!imageError && (
                <button
                  type="button"
                  onClick={() => setZoomed(true)}
                  className="absolute right-3 bottom-3 z-20 flex items-center justify-center rounded-full bg-white/90 p-2 text-slate-900 shadow hover:bg-white"
                  title="Ampliar imagem"
                >
                  <MagnifierIcon className="h-4 w-4" />
                </button>
              )}
            </div>
            {/* {result.documentPath && (
              <p className="rounded-2xl bg-white/5 px-4 py-3 text-xs text-white/60">
                Caminho original: {result.documentPath}
              </p>
            )} */}
          </div>
        </div>
      </div>
      {zoomed && !imageError && (
        <div
          className="fixed inset-0 z-[60] flex min-h-screen items-center justify-center bg-black/80 p-4 sm:p-6"
          onClick={(e) => {
            if (e.currentTarget === e.target) {
              setZoomed(false);
            }
          }}
        >
          <div
            ref={zoomContainerRef}
            className="relative flex h-full min-h-0 w-full max-h-[calc(100dvh-2rem)] max-w-[94vw] items-center justify-center overflow-visible sm:max-h-[calc(100dvh-3rem)]"
          >
            <button
              type="button"
              onClick={() => setZoomed(false)}
              className="absolute right-4 top-0 -translate-y-1/2 flex items-center gap-2 rounded-full bg-white/90 px-3 py-2 text-sm font-semibold text-slate-900 shadow hover:bg-white sm:right-6 sm:px-4"
            >
              Fechar zoom
            </button>
            <img
              src={imageUrl}
              alt={`Ampliação da lâmina ${identifier}`}
              className="max-h-[75vh] w-full max-w-full rounded-3xl border border-white/20 object-contain sm:max-h-full"
            />
          </div>
        </div>
      )}
    </div>
  , portalTarget);
}

export function ResultsGallery({ data, themeColor }: ResultsGalleryProps) {
  const [modal, setModal] = useState<ModalState>(null);

  const updatedAt = useMemo(() => {
    if (!data.timestamp) {
      return null;
    }
    const base = data.timestamp > 1_000_000_000 ? data.timestamp * 1000 : data.timestamp;
    try {
      return new Date(base).toLocaleString();
    } catch (error) {
      return null;
    }
  }, [data.timestamp]);

  const hasResults = data.results && data.results.length > 0;

  if (!hasResults) {
    return (
      <div className="rounded-2xl border border-dashed border-white/25 bg-white/5 p-6 text-center text-white/70">
        <p className="text-sm">Sem resultados ainda. Peça ao assistente para buscar lâminas semelhantes.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-2 text-white">
        <div className="flex flex-wrap items-center gap-3">
          <span className="rounded-full bg-white/15 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-white/80">
            {data.source === 'image' ? 'Busca por imagem' : 'Busca textual'}
          </span>
          {data.filters?.summary && (
            <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-white/70">
              Filtros: {data.filters.summary}
            </span>
          )}
          {data.filters?.fallbackUsed && (
            <span className="rounded-full bg-amber-500/20 px-3 py-1 text-xs text-amber-200">
              Resultados adicionais fora dos filtros incluídos
            </span>
          )}
          {data.query && (
            <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-white/70">
              Consulta: {data.query}
            </span>
          )}
          {updatedAt && (
            <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-white/50">
              Atualizado às {updatedAt}
            </span>
          )}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {data.results.map((result) => (
          <ResultCard
            key={`${result.rank}-${result.imageId}`}
            result={result}
            themeColor={themeColor}
            onSelect={({ result: cardResult, imageUrl }) => setModal({ result: cardResult, imageUrl })}
          />
        ))}
      </div>

      <ImagePreviewModal modal={modal} onClose={() => setModal(null)} />
    </div>
  );
}

'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

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
  diagnosisPrimary?: string | null;
  diagnosisSecondary?: string | null;
  diagnosisTertiary?: string | null;
  anatomSiteGeneral?: string | null;
  anatomSiteSpecial?: string | null;
  matchedFilters?: boolean;
};

export type SharedSearchResults = {
  source: 'image' | 'text';
  timestamp: number;
  filters?: {
    sex?: string | null;
    minAge?: number | null;
    maxAge?: number | null;
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
  return `${value.toFixed(2)}%`;
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
        <span className="absolute right-3 top-3 rounded-full bg-white/80 px-2 py-1 text-xs font-semibold text-slate-900">
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
          {result.sex && (
            <span className="rounded-full bg-white/10 px-2 py-1">{`Sexo: ${result.sex}`}</span>
          )}
          {result.ageApprox && (
            <span className="rounded-full bg-white/10 px-2 py-1">{`Idade≈ ${result.ageApprox}`}</span>
          )}
          {anatomy && (
            <span className="rounded-full bg-white/10 px-2 py-1">{anatomy}</span>
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

  if (!modal) {
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

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm sm:p-6"
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
        className="relative mx-auto flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-3xl bg-slate-950/95 text-white shadow-2xl ring-1 ring-white/10"
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
        <div className="grid flex-1 gap-5 overflow-y-auto p-5 sm:p-6 md:grid-cols-2 md:gap-6">
          <div className="flex flex-col gap-4 md:gap-5">
            <div className="rounded-2xl bg-white/10 p-4 text-sm text-white/80 sm:p-5">
              <p className="text-xs uppercase tracking-[0.2em] text-white/60">Identificador</p>
              <p className="text-lg font-semibold text-white">{identifier}</p>
              {secondaryIdentifier && (
                <p className="text-xs uppercase tracking-wide text-white/60">{secondaryIdentifier}</p>
              )}
              <p className="text-sm text-white/70">Similaridade: {formatSimilarity(result.similarity)}</p>
              {result.matchedFilters === false && (
                <p className="mt-2 rounded-md bg-amber-500/20 px-3 py-2 text-amber-200">
                  Este resultado foi incluído para completar o top-k, mas não atende a todos os filtros.
                </p>
              )}
            </div>
            <div className="rounded-2xl bg-white/10 p-4 text-sm sm:p-5">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-white/70">Detalhes do paciente</h3>
              <ul className="mt-2 space-y-1 text-white/80">
                <li>Código do caso: {result.caseCode ?? '—'}</li>
                <li>Código do slide: {result.slideCode ?? '—'}</li>
                <li>Sexo: {result.sex ?? '—'}</li>
                <li>Idade aproximada: {result.ageApprox ?? '—'}</li>
                <li>Local anatômico: {anatomy ?? '—'}</li>
              </ul>
            </div>
            <div className="rounded-2xl bg-white/10 p-4 text-sm sm:p-5">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-white/70">Diagnóstico</h3>
              <ul className="mt-2 space-y-1 text-white/80">
                <li>Principal: {diagnosis ?? '—'}</li>
                <li>Secundário: {result.diagnosisSecondary ?? '—'}</li>
                <li>Terciário: {result.diagnosisTertiary ?? '—'}</li>
              </ul>
            </div>
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
            {result.documentPath && (
              <p className="rounded-2xl bg-white/5 px-4 py-3 text-xs text-white/60">
                Caminho original: {result.documentPath}
              </p>
            )}
          </div>
        </div>
      </div>
      {zoomed && !imageError && (
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center bg-black/80 p-4 sm:p-6"
          onClick={(e) => {
            if (e.currentTarget === e.target) {
              setZoomed(false);
            }
          }}
        >
          <div
            ref={zoomContainerRef}
            className="relative flex h-full w-full max-h-[90vh] max-w-[94vw] items-center justify-center overflow-visible"
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
  );
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

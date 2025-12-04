"use client";

import { useCoAgent } from "@copilotkit/react-core";
import { CopilotKitCSSProperties, CopilotSidebar } from "@copilotkit/react-ui";
import { useMemo } from "react";
import { ResultsGallery, SharedSearchResults } from "./ResultsGallery";

export default function CopilotKitPage() {
  const themeColor = "#6366f1";

  return (
    <main style={{ "--copilot-kit-primary-color": themeColor } as CopilotKitCSSProperties}>
      <YourMainContent themeColor={themeColor} />
      <CopilotSidebar
        clickOutsideToClose={false}
        defaultOpen={true}
        imageUploadsEnabled={true}
        inputFileAccept="image/jpeg,image/png,image/webp"
        labels={{
          title: "Assistente de Histopatologia",
          initial: "👋 Olá! Você está conversando com um agente especializado em histopatologia. Experimente:\n- **Busca por imagem**: envie uma imagem para encontrar casos semelhantes\n- **Busca textual**: descreva padrões histológicos que deseja encontrar\n- **Filtros demográficos**: filtre por sexo, faixa etária ou ambos\n- **Filtros avançados**: refine por local primário, tecido/órgão de origem, local de ressecção/biópsia, tipo de tecido, tipo de amostra, tipo de doença e estágio patológico (AJCC/TNM)\n\nAcompanhe nesta tela o progresso das ferramentas e os resultados retornados em tempo real."
        }}
      />
    </main>
  );
}

// State of the agent, make sure this aligns with your agent's state.
type AgentState = {
  searchResults: SharedSearchResults | null;
}

function YourMainContent({ themeColor }: { themeColor: string }) {
  // 🪁 Shared State: https://docs.copilotkit.ai/coagents/shared-state
  const { state } = useCoAgent<AgentState>({
    name: "histopathology_agent",
    initialState: {
      searchResults: null,
    },
  })

  //🪁 Generative UI: https://docs.copilotkit.ai/coagents/generative-ui

  const galleryData = useMemo<SharedSearchResults>(() => {
    return state.searchResults ?? {
      source: "image",
      timestamp: 0,
      results: [],
      filters: null,
    };
  }, [state.searchResults]);

  return (
    <div
      style={{ backgroundColor: themeColor }}
      className="min-h-screen w-full overflow-y-auto px-4 py-10 transition-colors duration-300 sm:px-6"
    >
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 rounded-2xl bg-white/20 p-6 text-white shadow-xl backdrop-blur-md sm:p-8">
        <header className="flex flex-col gap-3 text-center sm:text-left">
          <h1 className="text-3xl font-bold text-white sm:text-4xl">Busca Histopatológica</h1>
          <p className="text-sm text-white/80 sm:text-base">
            Peça para o agente analisar uma imagem ou descreva o que procura; os resultados aparecerão logo abaixo com metadados clínicos relevantes.
          </p>
        </header>
        <hr className="border-white/20" />
        <section className="flex flex-col gap-5">
          <header className="flex flex-col gap-2 text-white text-center sm:text-left">
            <h2 className="text-2xl font-semibold">Resultados semelhantes</h2>
            <p className="text-sm text-white/70">
              Após solicitar uma busca, as imagens mais próximas aparecerão aqui com seus metadados essenciais.
            </p>
          </header>
          <ResultsGallery data={galleryData} themeColor={themeColor} />
        </section>
      </div>
    </div>
  );
}

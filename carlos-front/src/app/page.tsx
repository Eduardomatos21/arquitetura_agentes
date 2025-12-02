"use client";

import { useCoAgent, useCopilotAction } from "@copilotkit/react-core";
import { CopilotKitCSSProperties, CopilotSidebar } from "@copilotkit/react-ui";
import { useMemo, useState } from "react";
import { ResultsGallery, SharedSearchResults } from "./ResultsGallery";

export default function CopilotKitPage() {
  const [themeColor, setThemeColor] = useState("#6366f1");

  // 🪁 Frontend Actions: https://docs.copilotkit.ai/guides/frontend-actions
  useCopilotAction({
    name: "setThemeColor",
    parameters: [{
      name: "themeColor",
      description: "The theme color to set. Make sure to pick nice colors.",
      required: true,
    }],
    handler({ themeColor }) {
      setThemeColor(themeColor);
    },
  });

  return (
    <main style={{ "--copilot-kit-primary-color": themeColor } as CopilotKitCSSProperties}>
      <YourMainContent themeColor={themeColor} />
      <CopilotSidebar
        clickOutsideToClose={false}
        defaultOpen={true}
        imageUploadsEnabled={true}
        inputFileAccept="image/jpeg,image/png,image/webp"
        labels={{
          title: "Popup Assistant",
          initial: "👋 Olá! Você está conversando com um agente especializado em histopatologia. Experimente:\n- **Busca por imagem**: envie uma lâmina para encontrar casos semelhantes\n- **Busca textual**: descreva padrões histológicos que deseja encontrar\n- **Filtros demográficos**: peça por sexo, faixa etária ou ambos\n\nAcompanhe nesta tela o progresso das ferramentas e os resultados retornados em tempo real."
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
  useCopilotAction({
    name: "get_weather",
    description: "Get the weather for a given location.",
    available: "disabled",
    parameters: [
      { name: "location", type: "string", required: true },
    ],
    render: ({ args }) => {
      return <WeatherCard location={args.location} themeColor={themeColor} />
    },
  });

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

// Simple sun icon for the weather card
function SunIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-14 h-14 text-yellow-200">
      <circle cx="12" cy="12" r="5" />
      <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" strokeWidth="2" stroke="currentColor" />
    </svg>
  );
}

// Weather card component where the location and themeColor are based on what the agent
// sets via tool calls.
function WeatherCard({ location, themeColor }: { location?: string, themeColor: string }) {
  return (
    <div
    style={{ backgroundColor: themeColor }}
    className="rounded-xl shadow-xl mt-6 mb-4 max-w-md w-full"
  >
    <div className="bg-white/20 p-4 w-full">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-bold text-white capitalize">{location}</h3>
          <p className="text-white">Current Weather</p>
        </div>
        <SunIcon />
      </div>

      <div className="mt-4 flex items-end justify-between">
        <div className="text-3xl font-bold text-white">70°</div>
        <div className="text-sm text-white">Clear skies</div>
      </div>

      <div className="mt-4 pt-4 border-t border-white">
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <p className="text-white text-xs">Humidity</p>
            <p className="text-white font-medium">45%</p>
          </div>
          <div>
            <p className="text-white text-xs">Wind</p>
            <p className="text-white font-medium">5 mph</p>
          </div>
          <div>
            <p className="text-white text-xs">Feels Like</p>
            <p className="text-white font-medium">72°</p>
          </div>
        </div>
      </div>
    </div>
  </div>
  );
}

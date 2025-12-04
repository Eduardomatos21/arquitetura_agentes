# Arquitetura de Agentes – Instalação, Execução e Funcionalidades

Este repositório contém um frontend em Next.js e um agente em Python (FastAPI) que trabalham juntos para pesquisar e exibir resultados de imagens e textos (ex.: histopatologia) com apoio de um modelo multimodal (Gemini). O frontend fica em `projeto-agente/src/app` e o agente em `projeto-agente/agent`.

## Visão Geral
- **Frontend (Next.js)**: Interface com barra lateral (CopilotKit), upload de imagem, consulta textual e galeria de resultados. Código em `projeto-agente/src/app`.
- **Agente (Python/FastAPI)**: Recebe a última mensagem do usuário (texto e/ou imagem), aplica prompt de domínio, usa ferramentas de busca (vetorial/semântica) e retorna resultados com metadados clínicos. Código em `projeto-agente/agent`.
- **Dados locais**: Imagens em `projeto-agente/assets/STAD_TRAIN_MSIMUT/MSIMUT` (padrão) e vectorstore pré-computado em `projeto-agente/agent/vectorstore/chroma_vectorstore`.

## Pré-requisitos
- **Sistema Operacional**: Windows, Linux ou macOS.
- **Node.js**: versão 18 ou superior.
- **Gerenciador de pacotes**: `pnpm` (recomendado), `npm`, `yarn` ou `bun`.
- **Python**: versão 3.12 ou superior.
- **Chave da API Google**: obtenha em https://makersuite.google.com/app/apikey para usar o Gemini.

## Instalação

### 1. Instale dependências do frontend

**Windows (PowerShell):**
```powershell
cd .\projeto-agente
pnpm install
```

**Linux/macOS:**
```bash
cd projeto-agente
pnpm install
```

### 2. Instale dependências do agente Python

**Automático (recomendado):**
```bash
pnpm run install:agent
```

**Manual (alternativa):**

**Windows (PowerShell):**
```powershell
cd .\projeto-agente\agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
cd projeto-agente/agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução (Desenvolvimento)

#### Adicionar token da Hugging Face (MUSK)
O modelo MUSK baixa pesos do Hugging Face Hub. Defina o token para permitir o download:

**Windows (PowerShell):**
```powershell
$env:HUGGING_FACE_HUB_TOKEN="SEU_TOKEN_HF"
```

**Linux/macOS:**
```bash
export HUGGING_FACE_HUB_TOKEN="SEU_TOKEN_HF"
```

Observação: também são aceitas as variáveis `HF_TOKEN` ou `HUGGINGFACEHUB_API_TOKEN`.

### Iniciar UI + Agente (recomendado)

Executa frontend (porta 3000) e agente (porta 8000) simultaneamente:

**Windows (PowerShell):**
```powershell
cd .\projeto-agente
$env:GOOGLE_API_KEY="SUA_CHAVE_AQUI"
pnpm dev
```

**Linux/macOS:**
```bash
cd projeto-agente
export GOOGLE_API_KEY="SUA_CHAVE_AQUI"
pnpm dev
```

### Comandos separados (opcional)

**Somente UI:**
```bash
cd projeto-agente
pnpm run dev:ui
```

**Somente agente:**

**Windows (PowerShell):**
```powershell
cd .\projeto-agente
$env:GOOGLE_API_KEY="SUA_CHAVE_AQUI"
pnpm run dev:agent
```

**Linux/macOS:**
```bash
cd projeto-agente
export GOOGLE_API_KEY="SUA_CHAVE_AQUI"
pnpm run dev:agent
```

## Configuração (Variáveis de Ambiente)
- **Obrigatória**:
	- `GOOGLE_API_KEY`: chave da API do Google (Gemini).
- **Opcionais**:
	- `PORT`: porta do agente FastAPI (padrão `8000`).
	- `TCGA_IMAGE_DIR`: diretório de imagens (substitui o padrão `assets/STAD_TRAIN_MSIMUT/MSIMUT`).
	- `TCGA_VECTORSTORE_DIR` e `VECTORSTORE_COLLECTION`: diretórios/coleção para o banco vetorial Chroma.
	- `SESSION_MEDIA_CACHE_DIR`, `SESSION_MEDIA_MAX_FILES`: comportamento do cache de mídia de sessão.
	- `NVIDIA_VISIBLE_DEVICES`: seleção de GPU se CUDA disponível (opcional).

## Funcionalidades Principais
- **Chat multimodal (CopilotKit)**: envia texto e imagens; o sistema agrega e envia apenas a última mensagem do usuário com contexto relevante.
- **Busca por similaridade de imagem**: extrai imagem da mensagem ou cache de sessão e consulta vetorstore (Chroma) com regramento de filtros demográficos (sexo/idade).
- **Busca textual**: pesquisa semântica baseada em embeddings/texto.
- **Galeria de resultados**: exibe cartões ordenados por similaridade com metadados (sexo, idade, diagnóstico, sítio, estágio etc.).
- **API de imagens**: serviço que entrega arquivos de imagem locais de forma segura.

## Filtros Suportados
As ferramentas aceitam filtros opcionais que podem ser ditos em português e são mapeados para parâmetros das funções `search_by_image_query` e `search_by_text_query`.

- `sex` (`female`/`male`), `min_age`, `max_age` (valores inteiros).
- `primary_site`: local anatômico primário.
- `tissue_origin`: tecido/órgão de origem.
- `site_of_resection`: sítio de ressecção ou biópsia.
- `tissue_type`: tipo de tecido (ex.: Tumor, Solid Tissue).
- `specimen_type`: tipo de amostra (ex.: Solid Tissue).
- `disease_type`: tipo de doença/diagnóstico principal.
- `pathologic_stage`: estágio AJCC (ex.: Stage II).
- `ajcc_t`, `ajcc_n`, `ajcc_m`: componentes TNM.

Os valores podem ser informados como texto simples ou listas (por exemplo, "local primário: estômago" ou "tipo de doença: adenocarcinoma").

## Estrutura Relevante
- `projeto-agente/src/app/page.tsx`: página principal e integração com CopilotKit.
- `projeto-agente/src/app/ResultsGallery.tsx`: renderização da galeria e detalhes dos resultados.
- `projeto-agente/src/app/api/copilotkit/route.ts`: ponte do frontend para o agente.
- `projeto-agente/src/app/api/images/[imageId]/route.ts`: entrega de imagens locais por ID.
- `projeto-agente/agent/agent.py`: app FastAPI e integração com modelo Gemini.
- `projeto-agente/agent/tools.py`: ferramentas de busca (imagem/texto), filtros e normalizações.
- `projeto-agente/agent/session_media_store.py`: cache de mídia (imagem) por sessão.

## Como Usar

1. **Acesse a aplicação:** Abra `http://localhost:3000` no navegador.

2. **Interaja com o assistente** através da barra lateral:
   - **Busca por imagem:** Envie uma imagem histopatológica para encontrar casos semelhantes.
   - **Busca textual:** Descreva padrões morfológicos (ex.: "adenocarcinoma gástrico com padrão cribriforme").
   - **Filtros demográficos:** Use linguagem natural em português:
     - Sexo: "mulher", "homem", "feminino", "masculino"
     - Idade: "mais de 50 anos", "menos de 30", "entre 40 e 60 anos"
   - **Filtros avançados:** Especifique campos clínicos:
     - "local primário: estômago"
     - "tipo de tecido: Tumor"
     - "estágio AJCC T2"
     - "sítio de ressecção: Fundus of Stomach"

3. **Visualize resultados:** A galeria exibe as imagens mais similares com:
   - Percentual de similaridade
   - Metadados clínicos (diagnóstico, estágio, demográficos)
   - Indicação se o resultado atende aos filtros aplicados

4. **Refine a busca:** Você pode adicionar ou modificar filtros sem reenviar a imagem, desde que permaneça na mesma sessão.

## Troubleshooting

### Problemas Comuns

**Agente não responde / Erro de API Key:**
- Certifique-se de que `GOOGLE_API_KEY` está definida corretamente.
- Verifique se a chave é válida em https://makersuite.google.com/app/apikey

**Conflito de portas:**
- Frontend (3000): Modifique em `next.config.ts` ou use `PORT=3001 pnpm dev:ui`
- Backend (8000): Defina `PORT=8001` antes de executar o agente

**Erro ao instalar dependências Python:**
- Verifique a versão do Python: `python --version` (deve ser 3.12+)
- No Windows, pode ser necessário instalar Visual C++ Build Tools
- No Linux/Mac, pode precisar de `python3-dev` ou `python3-venv`

**Sessão perdida após reiniciar:**
- O cache de imagens da sessão é temporário
- Para nova busca com filtros, reenvie a imagem ou use a mesma sessão contínua

## Licença
Veja `projeto-agente/LICENSE` para detalhes de licenciamento.
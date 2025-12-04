# 📨 Fluxo Detalhado de Processamento de Mensagens Multimodais

Este documento descreve como mensagens com texto e imagens são processadas através do sistema de busca de imagens histopatológicas.

---

## **1️⃣ Etapa Frontend → API Route (CopilotKit)**

### Formato Original do CopilotKit

O CopilotKit separa texto e imagem em mensagens diferentes:

```typescript
{
  messages: [
    { 
      textMessage: { role: "user", content: "analise esta imagem" },
      createdAt: "2025-11-25T10:00:00.000Z"
    },
    { 
      imageMessage: { 
        bytes: "base64_encoded_data", 
        mimeType: "image/jpeg" 
      },
      createdAt: "2025-11-25T10:00:00.100Z"
    }
  ]
}
```

### Transformação no `route.ts`

O arquivo `src/app/api/copilotkit/route.ts` realiza três operações principais:

1. **Detecta imageMessage**
2. **Procura textMessage próximo** (dentro de ±2 segundos)
3. **Combina em formato AG-UI multimodal:**

```typescript
{
  textMessage: {
    role: "user",
    content: JSON.stringify([
      { type: "text", text: "analise esta imagem" },
      { type: "binary", mimeType: "image/jpeg", data: "base64..." }
    ])
  }
}
```

### Pruning (Otimização)

O sistema aplica uma otimização importante:
- Envia **APENAS a última mensagem do usuário** + prompt do sistema
- Evita enviar histórico completo ao ADK (economiza tokens e custos)
- Previne ambiguidade: "você enviou 10 mensagens, qual imagem usar?"

---

## **2️⃣ Etapa API Route → Backend ADK**

### Formato Enviado ao FastAPI

```json
{
  "messages": [
    {
      "role": "user",
      "content": "[{\"type\":\"text\",\"text\":\"analise\"},{\"type\":\"binary\",\"mimeType\":\"image/jpeg\",\"data\":\"iVBORw0KGgo...\"}]"
    }
  ]
}
```

---

## **3️⃣ Etapa Backend: `before_model_modifier` (agent.py)**

### Função: `extract_and_convert_images_from_contents`

Esta função é executada ANTES do LLM processar a mensagem.

#### **A. Processamento de Mensagens do Modelo (Sanitização)**

```python
# CRÍTICO: Remove inline_data de respostas do modelo
if content.role == "model":
    # Gemini NÃO deve ecoar imagens de volta
    # Isso previne erro "400: inline_data in model response"
    clean_parts = [p for p in parts if not p.inline_data]
```

**Por que isso é necessário?**
- O Gemini às vezes tenta incluir imagens nas respostas
- Isso causa erros no protocolo de comunicação
- A sanitização garante que apenas texto seja retornado

#### **B. Processamento de Mensagens do Usuário**

**Passo 1: Detecta JSON no texto**
```python
if text.startswith(('[', '{')):
    parsed = json.loads(text)
```

**Passo 2: Separa por tipo de conteúdo**
```python
# Tipo: "text"
if item.get("type") == "text":
    new_parts.append(types.Part(text="analise esta imagem"))

# Tipo: "binary" (imagem)
elif item.get("type") == "binary":
    # Decodifica base64
    image_data = base64.b64decode(data_str)
    
    # Cria Blob do Gemini
    blob = types.Blob(mime_type="image/jpeg", data=image_data)
    new_parts.append(types.Part(inline_data=blob))
    
    # ARMAZENA NO CACHE DE SESSÃO
    store_image_in_state(session_state, image_data, mime_type)
```

**Passo 3: Cria Content no formato do Gemini**
```python
# Formato final enviado ao LLM:
Content(role="user", parts=[
    Part(text="analise esta imagem"),
    Part(inline_data=Blob(mime_type="image/jpeg", data=bytes))
])
```

---

## **4️⃣ Etapa: Gemini Decide Usar Ferramenta**

### LlmRequest Enviado ao Gemini

```python
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {"text": "analise esta imagem"},
        {"inline_data": {"mime_type": "image/jpeg", "data": b"..."}}
      ]
    }
  ],
  "tools": [search_by_image_query, search_by_text_query],
  "system_instruction": """
    ... instrução do sistema com exemplos de filtros demográficos e clínicos:
    - "mulher", "sexo feminino" → sex="female"
    - "homem", "sexo masculino" → sex="male"
    - "mais de 50 anos" → min_age=50
    - "entre 40 e 60 anos" → min_age=40, max_age=60
    - "local primário: estômago" → primary_site="Stomach"
    - "tipo de tecido: tumor" → tissue_type="Tumor"
    - "estágio T2" → ajcc_t="T2"
    - "tecido sólido" → specimen_type="Solid Tissue"
  """
}
```

### Resposta do Gemini

```python
# Sem filtros
{
  "function_call": {
    "name": "search_by_image_query",
    "args": {"top_k": 5}
  }
}

# Com filtros demográficos
{
  "function_call": {
    "name": "search_by_image_query",
    "args": {
      "top_k": 5,
      "sex": "female",      # Detectado de "mulher" ou "feminino"
      "min_age": 50,        # Detectado de "mais de 50 anos"
      "max_age": 65         # Detectado de "até 65 anos"
    }
  }
}

# Com filtros clínicos avançados
{
  "function_call": {
    "name": "search_by_image_query",
    "args": {
      "top_k": 5,
      "primary_site": "Stomach",           # Local primário
      "tissue_type": "Tumor",              # Tipo de tecido
      "specimen_type": "Solid Tissue",     # Tipo de amostra
      "disease_type": "Adenocarcinoma",    # Tipo de doença
      "ajcc_t": "T2"                       # Estágio AJCC T
    }
  }
}
```

**Importante:** 
- O Gemini recebe a imagem mas **não precisa passá-la como parâmetro** para a ferramenta. A imagem é extraída automaticamente do contexto.
- O modelo detecta filtros demográficos em **português** e os converte para parâmetros estruturados (`sex`, `min_age`, `max_age`).

---

## **5️⃣ Etapa: Execução da Ferramenta (tools.py)**

### Função: `_extract_image_from_context`

Esta função implementa **múltiplas estratégias** para recuperar a imagem, em ordem de prioridade:

#### **Estratégia A: Inline_data Direto**
```python
# Procura inline_data no ToolContext
for part in content.parts:
    if hasattr(part, 'inline_data') and part.inline_data:
        return part.inline_data.data, mime_type
```

#### **Estratégia B: JSON Embarcado no Texto**
```python
# Se a imagem ainda está como JSON
if part.text:
    payload = json.loads(part.text)
    if item.get('type') == 'binary':
        return base64.b64decode(item['data']), mime_type
```

#### **Estratégia C: Cache de Sessão (Fallback)**
```python
# Última imagem armazenada na sessão
state = getattr(tool_context, 'state', None)
if state is not None:
    cached_bytes, cached_mime = load_image_from_state(state)
    return cached_bytes, cached_mime
```

**Por que múltiplas estratégias?**
- Robustez: se uma falhar, tenta a próxima
- Flexibilidade: suporta diferentes formatos de entrada
- Cache: permite queries subsequentes sem reenviar imagem

---

## **6️⃣ Processamento da Imagem com MUSK**

### Pipeline de Processamento

```python
# 1. Bytes → PIL Image
pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

# 2. Redimensiona se grande (otimização de memória)
if max(pil_image.size) > 2048:
    ratio = 2048 / max(pil_image.size)
    new_size = (int(w * ratio), int(h * ratio))
    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

# 3. Transforma para tensor (normalização + resize para 384x384)
image_tensor = transform(pil_image).to(DEVICE, dtype=torch.float16)

# 4. Gera embedding com MUSK (modelo multimodal)
with torch.inference_mode():
    features = model(
        image=image_tensor,
        with_head=True,
        out_norm=True,
        return_global=True
    )[0]

query_embedding = features.cpu().numpy().flatten().tolist()

# 5. Busca no Chroma (banco vetorial)
# Se filtros aplicados, busca até 300 candidatos para compensar filtragem
n_results = top_k if not filters_applied else min(300, max(top_k * 3, top_k))

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=n_results,  # 5 sem filtros, até 300 com filtros
    include=["distances", "documents", "metadatas"]
)

# 6. Aplica filtros de metadados usando FILTER_FIELD_MAP (se especificados)
# FILTER_FIELD_MAP define 10 filtros clínicos:
#   - primary_site, tissue_origin, site_of_resection (partial match)
#   - tissue_type, specimen_type (exact match)
#   - disease_type, pathologic_stage (partial match)
#   - ajcc_t, ajcc_n, ajcc_m (partial match)
if filters_applied:
    # Separa resultados que atendem os filtros (matched) dos demais (remainder)
    matched, remainder = _filter_metadata_results(
        candidates,
        sex=sex,                    # 'male' ou 'female'
        min_age=min_age,            # Ex: 50
        max_age=max_age,            # Ex: 65
        primary_site=primary_site,  # Ex: "Stomach"
        tissue_type=tissue_type,    # Ex: "Tumor"
        disease_type=disease_type,  # Ex: "Adenocarcinoma"
        # ... e outros 7 filtros clínicos
    )
    
    # Backfill: se matched < top_k, completa com remainder para garantir top_k resultados
    if len(matched) < top_k and remainder:
        needed = top_k - len(matched)
        matched.extend(remainder[:needed])
        fallback_used = True  # Marca que alguns resultados estão "fora dos filtros"
    
    candidates = matched[:top_k]
else:
    candidates = candidates[:top_k]
```

### Filtros de Metadados

**Normalização de Sexo:**
```python
# Aceita variações em português e inglês
"feminino", "mulher", "f", "female" → "female"
"masculino", "homem", "m", "male" → "male"
```

**Estratégias de Matching (FILTER_FIELD_MAP):**
```python
# Partial Match (permite substring case-insensitive):
# - primary_site, tissue_origin, site_of_resection
# - disease_type, pathologic_stage, ajcc_t/n/m
# Exemplo: "stomach" match "Fundus of Stomach"

# Exact Match (requer igualdade exata case-insensitive):
# - tissue_type, specimen_type
# Exemplo: "tumor" NÃO match "Tumor Primary" (só "Tumor")
```

**Mapeamento de Campos de Metadados:**
```python
FILTER_FIELD_MAP = {
    "primary_site": {"keys": ("primary_site.project", "primary_site"), "allow_partial": True},
    "tissue_origin": {"keys": ("tissue_or_organ_of_origin.diagnoses",), "allow_partial": True},
    "site_of_resection": {"keys": ("site_of_resection_or_biopsy.diagnoses",), "allow_partial": True},
    "tissue_type": {"keys": ("tissue_type.samples",), "allow_partial": False},
    "specimen_type": {"keys": ("specimen_type.samples",), "allow_partial": False},
    "disease_type": {"keys": ("disease_type.project", "disease_type"), "allow_partial": True},
    "pathologic_stage": {"keys": ("ajcc_pathologic_stage.diagnoses",), "allow_partial": True},
    "ajcc_t": {"keys": ("ajcc_pathologic_t.diagnoses",), "allow_partial": True},
    "ajcc_n": {"keys": ("ajcc_pathologic_n.diagnoses",), "allow_partial": True},
    "ajcc_m": {"keys": ("ajcc_pathologic_m.diagnoses",), "allow_partial": True},
}
```

**Filtro de Idade:**
```python
# Verifica age_approx ou age nos metadados
if min_age is not None:
    if patient_age < min_age:
        continue  # Descarta resultado
        
if max_age is not None:
    if patient_age > max_age:
        continue  # Descarta resultado
```

**Estratégia de Backfill:**
- Se filtros reduzem resultados para < top_k (ex: só 2 mulheres de 50-65 anos)
- Completa com os próximos mais similares **mesmo que não atendam os filtros**
- Marca esses resultados com "⚠️ fora dos filtros" para clareza
- **Garante sempre top_k=5 resultados**, conforme solicitado pelo usuário

### Cálculo de Proximidade Vetorial

```python
# Distância L2 → Percentual de proximidade vetorial
for distance in raw_distances:
    proximity_percent = max(0, (1 - distance / 2) * 100)
```

**Fórmula:** `proximity = (1 - distance/2) * 100`
- Distância 0.0 = 100% de proximidade vetorial
- Distância 2.0 = 0% de proximidade vetorial

---

## **7️⃣ Resposta Final**

### Formato Retornado pela Ferramenta

**Sem filtros:**
```python
"""
📊 Resultados da busca por imagem (Imagem → Imagens semelhantes):
  #01 | 94.32% de proximidade vetorial | TCGA_0053494 (sexo: female, idade≈55)
  #02 | 91.15% de proximidade vetorial | TCGA_0042781 (sexo: male, idade≈42)
  #03 | 88.67% de proximidade vetorial | TCGA_0038956 (sexo: female, idade≈61)
  #04 | 87.21% de proximidade vetorial | TCGA_0029145 (sexo: male, idade≈38)
  #05 | 85.09% de proximidade vetorial | TCGA_0051382 (sexo: female, idade≈50)
————————————————————————————————————————————————
"""
```

**Com filtros demográficos aplicados:**
```python
"""
📊 Resultados da busca por imagem (Imagem → Imagens semelhantes):
  ↳ Filtros aplicados: sexo: feminino, idade mínima: 50 anos
  #01 | 94.32% de proximidade vetorial | TCGA_0053494 (sexo: female, idade≈55)
  #02 | 88.67% de proximidade vetorial | TCGA_0038956 (sexo: female, idade≈61)
  #03 | 85.09% de proximidade vetorial | TCGA_0051382 (sexo: female, idade≈50)
  #04 | 82.45% de proximidade vetorial | TCGA_0067821 (sexo: male, idade≈42, ⚠️ fora dos filtros)
  #05 | 81.12% de proximidade vetorial | TCGA_0045392 (sexo: female, idade≈48, ⚠️ fora dos filtros)
————————————————————————————————————————————————
"""
```

**Com filtros clínicos avançados aplicados:**
```python
"""
📊 Resultados da busca por imagem (Imagem → Imagens semelhantes):
  ↳ Filtros aplicados: local primário: Stomach, tipo de tecido: Tumor, AJCC T: T2
  #01 | 92.18% de proximidade vetorial | TCGA_0029451 (Stomach, Tumor, T2)
  #02 | 89.34% de proximidade vetorial | TCGA_0041267 (Stomach, Tumor, T2)
  #03 | 86.72% de proximidade vetorial | TCGA_0052893 (Stomach, Tumor, T2)
  #04 | 83.55% de proximidade vetorial | TCGA_0038174 (Stomach, Tumor, T3, ⚠️ fora dos filtros)
  #05 | 81.91% de proximidade vetorial | TCGA_0047629 (Stomach, Normal Tissue, T2, ⚠️ fora dos filtros)
————————————————————————————————————————————————
"""
```

**Notas sobre os resultados:**
- Sempre retorna exatamente `top_k` resultados (padrão: 5)
- Metadados (sexo, idade) sempre exibidos quando disponíveis
- Resultados que não atendem filtros são marcados com "⚠️ fora dos filtros"
- Se filtros reduzem < top_k, completa com próximos mais similares (backfill)

### Fluxo Gemini → Frontend

1. Gemini recebe o resultado da ferramenta
2. Formata em linguagem natural (se necessário)
3. Retorna **APENAS TEXTO** (sem imagens!)
4. CopilotKit renderiza no chat do usuário

---

## 🔑 Pontos Críticos do Sistema

### **1. Cache de Sessão (`session_media_store.py`)**

#### Armazenamento Dual

```python
state["media:last_image"] = {
    "base64": "encoded_data",      # Em memória (rápido)
    "path": "/cache/img_123.bin",  # Em disco (persistente)
    "mime_type": "image/jpeg",
    "updated_at": 1732534800
}
```

#### Vantagens

- **Persistência:** Imagem disponível entre requisições
- **Follow-up queries:** Usuário pode fazer perguntas subsequentes sem reenviar
  - Exemplo: Upload imagem → "busque imagens similares" → "agora busque por texto: melanoma"
- **Economia de banda:** Evita retransmissão de dados pesados

#### Limpeza Automática

```python
# Mantém apenas 32 arquivos mais recentes
MAX_CACHED_FILES = 32
_prune_cache()  # Remove arquivos antigos
```

---

### **2. Sincronização via CoAgents (Shared State)**

#### Arquitetura

```typescript
// Frontend: Observa estado compartilhado
const { state } = useCoAgent<AgentState>({
  name: "histopathology_agent",
  initialState: { searchResults: null }
});
```

```python
# Backend: Publica resultados para estado compartilhado
def _push_results_to_state(tool_context, payload):
    state = getattr(tool_context, 'state', None)
    if state is not None:
        state["searchResults"] = payload
        logger.info(f"📤 Publicado {len(payload['results'])} resultados para Shared State")
```

#### Vantagens

- **Sincronização automática:** Frontend re-renderiza quando `state["searchResults"]` atualiza
- **Sem polling:** Não precisa fazer requests HTTP repetidos
- **Estrutura rica:** Backend envia objeto completo com metadados, filtros, timestamps
- **Separação de concerns:** Chat exibe texto do Gemini, galeria exibe dados estruturados

#### Payload Publicado

```typescript
interface SharedSearchResults {
  results: SearchResultItem[];  // Array com rank, proximity, imageId, metadados completos
  filters?: {
    sex?: string;
    minAge?: number;
    maxAge?: number;
    normalized?: Record<string, any>;   // Filtros normalizados (para matching)
    display?: Record<string, any>;      // Filtros display-friendly (para UI)
    summary?: string;                   // "sexo: feminino, local primário: Stomach"
    fallbackUsed?: boolean;             // true se backfill foi necessário
  };
  timestamp?: number;
}
```

#### Fluxo de Sincronização

1. **Backend:** `search_by_image_query()` executa busca
2. **Backend:** `_push_results_to_state(tool_context, payload)` publica
3. **CopilotKit:** Propaga atualização via WebSocket/SSE
4. **Frontend:** `useCoAgent` detecta mudança em `state.searchResults`
5. **React:** Re-renderiza `<ResultsGallery data={state.searchResults} />`
6. **UI:** Galeria exibe cards com metadados completos

---

### **3. Sanitização de Respostas do Modelo**

#### Problema

O Gemini às vezes tenta ecoar imagens recebidas nas respostas, causando:
- Erro 400: "inline_data not allowed in model response"
- Loops infinitos de processamento
- Desperdício de tokens

#### Solução

```python
# Remove inline_data de TODAS as mensagens do modelo
if content.role == "model" and has_inline_data:
    clean_parts = [p for p in parts if not p.inline_data]
    content = Content(role="model", parts=clean_parts)
```

---

### **4. Pruning de Mensagens**

#### Estratégia

```typescript
// Envia APENAS a última mensagem do usuário + sistema
const systemMessage = messages.find(m => m.role === "system");
const latestUserMessage = messages.reverse().find(m => m.role === "user");

const prunedMessages = [systemMessage, latestUserMessage].filter(Boolean);
```

#### Por que é necessário?

- **Economia de tokens:** Histórico completo pode ter centenas de mensagens
- **Clareza:** Evita ambiguidade sobre qual imagem processar
- **Performance:** Menos dados para serializar/transmitir

#### Limitações

- Não mantém contexto de conversas longas
- Agente não "lembra" de interações anteriores
- Adequado para queries independentes (busca por imagem/texto)

---

## 📊 Diagrama do Fluxo Completo

```
┌─────────────────────────────────────────────────────────────────┐
│                   Frontend (CopilotKit)                         │
│  User uploads image + types text                                │
└────────────────────────┬────────────────────────────────────────┘
                         │ imageMessage + textMessage
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               route.ts (API Middleware)                         │
│  - Detecta imageMessage                                         │
│  - Procura textMessage próximo (±2s)                            │
│  - Combina → JSON: [{type:"text"}, {type:"binary"}]             │
│  - Aplica pruning (só última mensagem)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │ POST /api/copilotkit
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          agent.py (before_model_modifier)                       │
│  - Parseia JSON do texto                                        │
│  - Converte para Gemini Parts:                                  │
│    • {type:"text"} → Part(text)                                 │
│    • {type:"binary"} → Part(inline_data=Blob)                   │
│  - Armazena em cache: session_state["media:last_image"]         │
│  - Remove inline_data de respostas do modelo (sanitização)      │
└────────────────────────┬────────────────────────────────────────┘
                         │ LlmRequest
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Gemini 2.5 Flash                             │
│  - Analisa texto + imagem                                       │
│  - Decide usar ferramenta: search_by_image_query(top_k=5)       │
└────────────────────────┬────────────────────────────────────────┘
                         │ function_call
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         tools.py (_extract_image_from_context)                  │
│  Tenta 3 estratégias em ordem:                                  │
│  1. inline_data direto no ToolContext                           │
│  2. JSON embarcado no texto                                     │
│  3. Cache de sessão (fallback)                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │ image_bytes
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         tools.py (search_by_image_query)                        │
│  1. Bytes → PIL Image                                           │
│  2. Resize se > 2048px                                          │
│  3. Transform → Tensor (384x384, normalizado)                   │
│  4. MUSK model → Embedding (768 dimensões)                      │
│  5. Chroma.query(embedding, n_results=5 ou até 300)             │
│  6. Aplica filtros usando FILTER_FIELD_MAP (12 filtros):        │
│     - Demográficos: sex, min_age, max_age                       │
│     - Clínicos: primary_site, tissue_origin, site_of_resection, │
│       tissue_type, specimen_type, disease_type,                 │
│       pathologic_stage, ajcc_t, ajcc_n, ajcc_m                  │
│     - Separa matched (atendem filtros) e remainder              │
│     - Se matched < top_k, completa com remainder (backfill)     │
│     - Marca resultados fora dos filtros com matchedFilters=false│
│  7. Calcula similaridade: (1 - L2_dist/2) * 100                 │
│  8. Publica para Shared State via _push_results_to_state():     │
│     state["searchResults"] = {                                  │
│       results: [...],  # Array com metadados completos          │
│       filters: {normalized, display, summary, fallbackUsed},    │
│       timestamp: ...                                            │
│     }                                                           │
└────────────────────────┬────────────────────────────────────────┘
                         │ formatted_results (string) + state update
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Gemini 2.5 Flash                             │
│  - Recebe resultados da ferramenta                              │
│  - Formata resposta em linguagem natural                        │
│  - Retorna APENAS TEXTO (sem imagens)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │ response
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         Frontend (CopilotKit + CoAgents)                        │
│  - Chat renderiza resposta textual do Gemini                    │
│  - useCoAgent observa state["searchResults"] (Shared State)     │
│  - ResultsGallery sincroniza automaticamente:                   │
│    • Exibe cards com metadados completos                        │
│    • Mostra chips de filtros aplicados                          │
│    • Marca resultados fora dos filtros com badge                │
│    • Modal com detalhes completos (staging, diagnósticos)       │
│  - Permite follow-up queries                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💡 Casos de Uso Suportados

### **1. Busca por Imagem (Simples)**

```
User: [uploads image]
Agent: search_by_image_query(top_k=5)
Result: Top 5 imagens similares com percentuais + metadados (sexo, idade)
```

### **2. Busca por Imagem com Filtro de Sexo**

```
User: [uploads image] "busque apenas em mulheres"
Agent: search_by_image_query(top_k=5, sex="female")
Result: Top 5 imagens de pacientes do sexo feminino
        (com backfill se < 5 encontradas, marcadas com ⚠️)
```

### **3. Busca por Imagem com Filtro de Idade**

```
User: [uploads image] "pacientes acima de 50 anos"
Agent: search_by_image_query(top_k=5, min_age=50)
Result: Top 5 imagens de pacientes com idade ≥ 50
```

### **4. Busca por Imagem com Filtros Combinados**

```
User: [uploads image] "homens entre 40 e 60 anos"
Agent: search_by_image_query(top_k=5, sex="male", min_age=40, max_age=60)
Result: Top 5 imagens de pacientes masculinos de 40-60 anos
        (completa com outros se insuficientes)
```

### **5. Busca por Texto**

```
User: "find images with melanoma characteristics"
Agent: search_by_text_query(text_query="melanoma characteristics", top_k=5)
Result: Top 5 imagens correspondentes à descrição
```

### **6. Busca por Texto com Filtros**

```
User: "melanoma em mulheres acima de 55 anos"
Agent: search_by_text_query(
         text_query="melanoma",
         sex="female",
         min_age=55,
         top_k=5
       )
Result: Top 5 imagens de melanoma em pacientes femininas com idade ≥ 55
```

### **7. Follow-up sem Reenviar Imagem**

```
User: [uploads image] "analyze this"
Agent: [processes and caches image]

User: "now search for melanoma"
Agent: [uses cached image from session_state]
```

### **8. Mensagem Curta + Imagem**

```
User: [uploads image] "a"  # Mensagem de 1 letra
Agent: Interpreta como "analyze this image"
       → search_by_image_query(top_k=5)
```

### **9. Busca com Filtros Clínicos de Local Primário**

```
User: [uploads image] "comparar com amostras do estômago"
Agent: search_by_image_query(top_k=5, primary_site="Stomach")
Result: Top 5 imagens de casos com local primário no estômago
```

### **10. Busca com Filtros de Tipo de Tecido**

```
User: "buscar imagens de tecido tumoral apenas"
Agent: search_by_text_query(text_query="...", tissue_type="Tumor")
Result: Top 5 imagens de amostras tumorais
```

### **11. Busca com Estágio AJCC**

```
User: [uploads image] "procurar casos no estágio T2"
Agent: search_by_image_query(top_k=5, ajcc_t="T2")
Result: Top 5 imagens de casos classificados como T2
```

### **12. Combinação Múltipla de Filtros Clínicos**

```
User: [uploads image] "estômago, tecido sólido, adenocarcinoma, estágio T2N1"
Agent: search_by_image_query(
         top_k=5,
         primary_site="Stomach",
         specimen_type="Solid Tissue",
         disease_type="Adenocarcinoma",
         ajcc_t="T2",
         ajcc_n="N1"
       )
Result: Top 5 imagens atendendo todos os critérios clínicos
        (com backfill se < 5 encontrados)
```

---

## 🛡️ Tratamento de Erros

### **1. Imagem Não Encontrada**

```python
if not image_bytes:
    return "❌ Nenhuma imagem foi fornecida. Por favor, envie uma imagem."
```

### **2. Modelo/Vectorstore Indisponível**

```python
if not model or not vectorstore:
    return "❌ Falha ao inicializar modelo ou vectorstore."
```

### **3. Erro ao Processar Imagem**

```python
try:
    pil_image = Image.open(BytesIO(image_bytes))
except Exception as e:
    return f"❌ Erro ao processar imagem: {str(e)}"
```

### **4. Fallback em `route.ts`**

```typescript
catch (error) {
    console.error("❌ Error processing request:", error);
    // Fallback: envia payload original sem transformação
    return handleRequest(req);
}
```

---

## 🔧 Configurações Importantes

### **Modelo MUSK**

```python
MODEL = "musk_large_patch16_384"
DEVICE = cuda:0 if available else cpu
DTYPE = torch.float16  # Half precision para economia de memória
```

### **Vectorstore Chroma**

```python
VECTORSTORE_DIR = "./vectorstore/chroma_vectorstore"
COLLECTION = "tcga_images_precomputed"
EMBEDDING_DIM = 768  # Dimensão dos embeddings MUSK
```

### **Cache**

```python
CACHE_DIR = "./session_media_cache"
MAX_CACHED_FILES = 32
```

### **API**

```typescript
BACKEND_URL = "http://localhost:8000/"
AGENT_NAME = "histopathology_agent"
```

---

## 📚 Referências Técnicas

- **CopilotKit:** Framework de UI para agentes AI
- **Google ADK:** Agent Development Kit para Gemini
- **MUSK:** Multimodal Universal Search with Knowledge (modelo de embeddings)
- **Chroma:** Banco vetorial para busca por similaridade
- **AG-UI:** Protocolo de comunicação entre frontend e ADK

---

## 🎯 Resumo Executivo

### Fluxo em 7 Passos

1. **Frontend:** Usuário envia texto + imagem (separados)
2. **API Route:** Combina em formato JSON multimodal
3. **Agent (before):** Converte JSON → Gemini Parts + armazena cache
4. **Gemini:** Decide usar ferramenta `search_by_image_query`
5. **Tools:** Extrai imagem (inline/JSON/cache) → processa com MUSK
6. **Tools:** Busca no Chroma → retorna top 5 com proximidade vetorial
7. **Frontend:** Exibe resultados formatados via Shared State (CoAgents)

### Pontos Fortes

✅ Múltiplas estratégias de extração de imagem (robustez)  
✅ Cache de sessão (eficiência)  
✅ Sanitização de respostas (previne erros)  
✅ Pruning de mensagens (economia de tokens)  
✅ Suporta follow-up queries sem reenvio  
✅ **12 filtros de metadados** (3 demográficos + 10 clínicos) com detecção em português  
✅ **FILTER_FIELD_MAP:** mapeamento centralizado com estratégias partial/exact match  
✅ **Backfill automático** garante sempre top_k resultados mesmo com filtros restritivos  
✅ **Transparência:** marca resultados fora dos filtros com `matchedFilters: false`  
✅ **Exibe metadados completos:** demográficos, diagnóstico, estágio AJCC, tipos de tecido  
✅ **Shared State (CoAgents):** sincronização automática backend→frontend sem polling  
✅ **Galeria rica:** cards interativos, modal de detalhes, chips de filtros aplicados  
✅ **Normalização inteligente:** aceita variações em português ("estômago", "Stomach")  
✅ **Proximidade vetorial:** métrica clara para ranking de resultados (0-100%)  
✅ **Dataset TCGA:** usa imagens histopatológicas do The Cancer Genome Atlas  

### Limitações

⚠️ Sem contexto de conversas longas (pruning agressivo)  
⚠️ Cache limitado a 32 arquivos  
⚠️ Requer CUDA para performance ideal (CPU funciona mas é lento)  
⚠️ Dependência de API do Google (Gemini)  
⚠️ **Filtros podem ser "relaxados"** via backfill se poucos resultados atendem critérios  
⚠️ **Metadados ausentes** em algumas imagens causam exclusão nos filtros  
⚠️ **Filtros clínicos dependem de qualidade dos metadados** do dataset TCGA  
⚠️ **Matching case-insensitive:** "stomach" match "Fundus of Stomach" (pode gerar falsos positivos)  
⚠️ **Shared State limitado:** só persiste durante sessão ativa (não há banco de dados)  

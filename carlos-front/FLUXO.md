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
  "tools": [search_by_image_query, search_by_text_query]
}
```

### Resposta do Gemini

```python
{
  "function_call": {
    "name": "search_by_image_query",
    "args": {"top_k": 5}
  }
}
```

**Importante:** O Gemini recebe a imagem mas **não precisa passá-la como parâmetro** para a ferramenta. A imagem é extraída automaticamente do contexto.

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
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=top_k,
    include=["distances", "documents", "metadatas"]
)
```

### Cálculo de Similaridade

```python
# Distância L2 → Percentual de similaridade
for distance in raw_distances:
    similarity_percent = max(0, (1 - distance / 2) * 100)
```

**Fórmula:** `similarity = (1 - L2_distance/2) * 100`
- Distância 0.0 = 100% similar
- Distância 2.0 = 0% similar

---

## **7️⃣ Resposta Final**

### Formato Retornado pela Ferramenta

```python
"""
📊 Resultados da busca por imagem (Imagem → Imagens semelhantes):
  #01 | 94.32% de similaridade | ISIC_0053494.jpg
  #02 | 91.15% de similaridade | ISIC_0042781.jpg
  #03 | 88.67% de similaridade | ISIC_0038956.jpg
  #04 | 87.21% de similaridade | ISIC_0029145.jpg
  #05 | 85.09% de similaridade | ISIC_0051382.jpg
————————————————————————————————————————————————
"""
```

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

### **2. Sanitização de Respostas do Modelo**

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

### **3. Pruning de Mensagens**

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
│  5. Chroma.query(embedding) → Top 5 resultados                  │
│  6. Calcula similaridade: (1 - L2_dist/2) * 100                 │
└────────────────────────┬────────────────────────────────────────┘
                         │ formatted_results (string)
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
│                   Frontend (CopilotKit)                         │
│  - Renderiza resultados no chat                                 │
│  - Exibe percentuais de similaridade                            │
│  - Permite follow-up queries                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💡 Casos de Uso Suportados

### **1. Busca por Imagem**

```
User: [uploads image]
Agent: search_by_image_query(top_k=5)
Result: Top 5 imagens similares com percentuais
```

### **2. Busca por Texto**

```
User: "find images with melanoma characteristics"
Agent: search_by_text_query(query="melanoma characteristics", top_k=5)
Result: Top 5 imagens correspondentes à descrição
```

### **3. Follow-up sem Reenviar Imagem**

```
User: [uploads image] "analyze this"
Agent: [processes and caches image]

User: "now search for melanoma"
Agent: [uses cached image from session_state]
```

### **4. Mensagem Curta + Imagem**

```
User: [uploads image] "a"  # Mensagem de 1 letra
Agent: Interpreta como "analyze this image"
       → search_by_image_query(top_k=5)
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
VECTORSTORE_DIR = "./streamlit_chroma_vectorstore_precomputed"
COLLECTION = "isic_images_precomputed"
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
6. **Tools:** Busca no Chroma → retorna top 5 similares
7. **Frontend:** Exibe resultados formatados

### Pontos Fortes

✅ Múltiplas estratégias de extração de imagem (robustez)  
✅ Cache de sessão (eficiência)  
✅ Sanitização de respostas (previne erros)  
✅ Pruning de mensagens (economia de tokens)  
✅ Suporta follow-up queries sem reenvio  

### Limitações

⚠️ Sem contexto de conversas longas (pruning agressivo)  
⚠️ Cache limitado a 32 arquivos  
⚠️ Requer CUDA para performance ideal  
⚠️ Dependência de API do Google (Gemini)  

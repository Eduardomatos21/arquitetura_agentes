"""Shared State feature."""

from __future__ import annotations
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    print(f"✅ GOOGLE_API_KEY configurada: {api_key[:10]}...{api_key[-4:]}")
else:
    print("❌ GOOGLE_API_KEY não encontrada!")

from dotenv import load_dotenv
load_dotenv()

import json
import base64
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI
from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint

# ADK imports
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner
from google.adk.events import Event, EventActions
from google.adk.tools import FunctionTool, ToolContext
from google.genai.types import Content, Part
from google.adk.models import LlmResponse, LlmRequest
from google.genai import types
from tools import search_by_image_query, search_by_text_query
from session_media_store import store_image_in_state

logger = logging.getLogger("histopathology.agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[agent] %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

agent_name = "histopathology_agent"


def on_before_agent(callback_context: CallbackContext):
    """Inicializa o estado do agente."""
    logger.info("on_before_agent invoked for agent=%s session=%s", callback_context.agent_name, getattr(callback_context, 'session_id', 'n/a'))
    logger.info("before_model_modifier completed for agent=%s", agent_name)
    return None


def extract_and_convert_images_from_contents(
    contents: List[Content],
    session_state: Optional[Any] = None,
    user_content: Any = None,
) -> List[Content]:
    """
    Extrai imagens de mensagens AG-UI e converte para formato Part do Gemini.
    
    Processa mensagens do usuário que podem conter conteúdo multimodal no formato AG-UI:
    - BinaryInputContent com type="binary" e mimeType começando com "image/"
    - Converte base64 ou URLs para Part com inline_data (Blob)
    
    Args:
        contents: Lista de Content do LlmRequest (fonte confiável para mensagem atual)
        user_content: IGNORADO - pode conter dados obsoletos de mensagens anteriores
        
    Returns:
        Lista de Content modificada com imagens convertidas para Parts
    """
    import requests
    from io import BytesIO
    
    modified_contents = []
    
    # Debug: Log what we're processing
    print(f"📊 extract_and_convert_images called with {len(contents)} content(s)")
    logger.info("extract_and_convert_images received %s content entries", len(contents))
    for i, c in enumerate(contents):
        role = c.role if hasattr(c, 'role') else 'unknown'
        parts_count = len(c.parts) if hasattr(c, 'parts') and c.parts else 0
        print(f"  Content[{i}]: role={role}, parts={parts_count}")
        logger.info("Content[%s] role=%s parts=%s", i, role, parts_count)
    
    for content in contents:
        # CRÍTICO: Limpar inline_data indevido de respostas do modelo
        if content.role == "model":
            if hasattr(content, 'parts') and content.parts:
                # Verificar se há inline_data (imagem) na resposta do modelo
                has_inline_data = any(
                    hasattr(p, 'inline_data') and p.inline_data 
                    for p in content.parts
                )
                if has_inline_data:
                    print(f"⚠️ WARNING: Model response contains inline_data! Removing it.")
                    # Filtrar apenas parts que NÃO são inline_data
                    clean_parts = [
                        p for p in content.parts 
                        if not (hasattr(p, 'inline_data') and p.inline_data)
                    ]
                    if clean_parts:
                        try:
                            clean_content = types.Content(role=content.role, parts=clean_parts)
                            modified_contents.append(clean_content)
                            print(f"✅ Cleaned model content: kept {len(clean_parts)} non-image parts")
                        except Exception as e:
                            print(f"❌ Error creating clean model content: {e}")
                            modified_contents.append(content)
                    else:
                        # Se só tinha inline_data, isso é um problema crítico
                        print(f"❌ CRITICAL: Model content has ONLY inline_data! Keeping original.")
                        modified_contents.append(content)
                else:
                    # Modelo sem inline_data - normal
                    modified_contents.append(content)
            else:
                modified_contents.append(content)
            continue
        
        # Processar apenas mensagens do usuário
        if content.role != "user":
            modified_contents.append(content)
            continue
        
        # Processar APENAS as partes do content atual (não user_content obsoleto)
        new_parts = []
        
        for part in content.parts:
            # Manter partes que já possuem inline_data (imagens já processadas)
            if hasattr(part, 'inline_data') and part.inline_data:
                if session_state is not None:
                    inline_data = getattr(part.inline_data, 'data', None)
                    if inline_data:
                        store_image_in_state(
                            session_state,
                            inline_data,
                            getattr(part.inline_data, 'mime_type', 'image/png'),
                            source="inline_part",
                        )
                new_parts.append(part)
                continue
            
            # Processar texto que pode conter JSON com conteúdo AG-UI
            if hasattr(part, 'text') and part.text:
                text = part.text.strip()
                
                # Tentar parsear JSON se começar com [ ou {
                if text.startswith(('[', '{')):
                    try:
                        parsed = json.loads(text)
                        
                        # Se for uma lista, processar cada item
                        if isinstance(parsed, list):
                            for item in parsed:
                                if not isinstance(item, dict):
                                    continue
                                
                                # Conteúdo de texto
                                if item.get("type") == "text":
                                    text_content = item.get("text", "")
                                    if text_content:
                                        new_parts.append(types.Part(text=text_content))
                                
                                # Conteúdo binário (imagem)
                                elif item.get("type") == "binary" and item.get("mimeType", "").startswith("image/"):
                                    mime_type = item.get("mimeType", "image/jpeg")
                                    image_data = None
                                    
                                    # Base64 data
                                    if "data" in item:
                                        data_str = item["data"]
                                        # Remover prefixo data:image/...;base64, se presente
                                        if "," in data_str:
                                            data_str = data_str.split(",", 1)[1]
                                        try:
                                            image_data = base64.b64decode(data_str)
                                        except Exception as e:
                                            print(f"⚠️ Erro ao decodificar base64: {e}")
                                            continue
                                    
                                    # URL da imagem
                                    elif "url" in item:
                                        try:
                                            response = requests.get(item["url"], timeout=30)
                                            response.raise_for_status()
                                            image_data = response.content
                                        except Exception as e:
                                            print(f"⚠️ Erro ao baixar imagem de URL: {e}")
                                            continue
                                    
                                    # Adicionar imagem se dados foram obtidos
                                    if image_data:
                                        blob = types.Blob(mime_type=mime_type, data=image_data)
                                        new_parts.append(types.Part(inline_data=blob))
                                        if session_state is not None:
                                            store_image_in_state(
                                                session_state,
                                                image_data,
                                                mime_type,
                                                source="binary_payload",
                                            )
                        else:
                            # JSON não é lista - manter como texto
                            new_parts.append(part)
                    
                    except json.JSONDecodeError:
                        # Não é JSON válido - manter como texto
                        new_parts.append(part)
                else:
                    # Texto simples - manter
                    new_parts.append(part)
            else:
                # Parte sem texto ou inline_data - manter
                new_parts.append(part)
        
        # Criar Content modificado se houver partes
        if new_parts:
            try:
                modified_content = types.Content(role=content.role, parts=new_parts)
                modified_contents.append(modified_content)
            except Exception as e:
                print(f"❌ Erro ao criar Content modificado: {e}")
                modified_contents.append(content)
        else:
            # CRÍTICO: Sem partes válidas - manter original para evitar content vazio
            print(f"⚠️ WARNING: Content {content.role} has no parts after processing, keeping original")
            modified_contents.append(content)
    
    return modified_contents


def before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Processa conteúdo multimodal e modifica instrução do sistema."""
    agent_name = callback_context.agent_name
    
    # Debug: Log message count and details
    original_content_count = len(llm_request.contents) if llm_request.contents else 0
    print(f"🔍 Processing {original_content_count} content(s) in llm_request")
    
    # Debug detalhado do conteúdo
    if llm_request.contents:
        for i, content in enumerate(llm_request.contents):
            role = content.role if hasattr(content, 'role') else 'unknown'
            parts_count = len(content.parts) if hasattr(content, 'parts') and content.parts else 0
            print(f"  Content[{i}]: role={role}, parts={parts_count}")
            
            if hasattr(content, 'parts') and content.parts:
                for j, part in enumerate(content.parts):
                    if hasattr(part, 'text') and part.text:
                        text_preview = part.text[:80].replace('\n', ' ') if len(part.text) > 80 else part.text
                        print(f"    Part[{j}]: text='{text_preview}...'")
                    elif hasattr(part, 'inline_data'):
                        print(f"    Part[{j}]: inline_data (image)")
                    elif hasattr(part, 'function_call'):
                        fn_name = part.function_call.name if hasattr(part.function_call, 'name') else 'unknown'
                        print(f"    Part[{j}]: function_call={fn_name}")
                    elif hasattr(part, 'function_response'):
                        print(f"    Part[{j}]: function_response")
    
    # Processar conteúdo multimodal (imagens)
    if llm_request.contents:
        user_content = getattr(callback_context, 'user_content', None)
        
        try:
            modified_contents = extract_and_convert_images_from_contents(
                llm_request.contents,
                session_state=getattr(callback_context, 'state', None),
                user_content=user_content,
            )
            
            # Validar que não removemos todo o conteúdo
            if not modified_contents:
                print(f"❌ CRITICAL: modified_contents is EMPTY! Keeping original contents.")
                print(f"   Original had {original_content_count} content(s)")
                # NÃO modificar llm_request.contents - manter original
            elif len(modified_contents) < original_content_count:
                print(f"⚠️ WARNING: Content count decreased from {original_content_count} to {len(modified_contents)}")
                # Verificar se algum content ficou sem parts
                all_valid = all(
                    hasattr(c, 'parts') and c.parts and len(c.parts) > 0 
                    for c in modified_contents
                )
                if all_valid:
                    print(f"✅ All {len(modified_contents)} contents are valid, applying changes")
                    llm_request.contents = modified_contents
                else:
                    print(f"❌ Some contents are invalid (no parts), keeping original")
            else:
                print(f"✅ Modified contents has {len(modified_contents)} content(s), applying changes")
                logger.info("Content sanitation successful; updating llm_request with %s entries", len(modified_contents))
                llm_request.contents = modified_contents
        except Exception as e:
            logger.exception("before_model_modifier failed while processing contents: %s", e)
            print(f"❌ Erro ao processar conteúdo multimodal: {e}")
            import traceback
            traceback.print_exc()
    
    if agent_name == "histopathology_agent":
        logger.info("Injecting histopathology-specific system instruction prefix")
        original_instruction = llm_request.config.system_instruction or types.Content(role="system", parts=[])
        prefix = f"""You are a helpful AI assistant specialized in histopathology image analysis.
        You can search for similar histology slide images using either an image query or a text description.
        
        When the user provides an image, use the search_by_image_query tool WITHOUT any image parameter.
        The image will be automatically extracted from the user's message context.
        Just call: search_by_image_query(top_k=5)
        
        When the user describes histological features in text, use search_by_text_query(text_query="descrição", top_k=5).

        If the user mentions demographic filters (e.g., "mulher", "sexo feminino", "homem", "masculino") pass the parameter
        sex="female" or sex="male" respectively.

        Age filters must use the parameters min_age and/or max_age. Interpret phrases like "mais de 50 anos" as min_age=50,
        "menos de 30" as max_age=30, and "entre 40 e 60" as min_age=40, max_age=60. Support Portuguese variations such as
        "idade > 50", "com 55 anos", or "faixa etária 45-65".

        When localização/anatomia or staging filters are present, map them to the appropriate parameters:
        - primary_site → "local primário", "primary site"
        - tissue_origin → "tecido/órgão de origem"
        - site_of_resection → "sítio de ressecção", "biopsy site"
        - tissue_type → "tipo de tecido"
        - specimen_type → "tipo de amostra"
        - disease_type → "tipo de doença", "diagnóstico"
        - pathologic_stage → "estágio patológico", "AJCC stage"
        - ajcc_t / ajcc_n / ajcc_m → componentes TNM.

        Passe valores textuais em inglês quando possível (ex.: "stomach", "tumor"), mas respeite a grafia solicitada pelo usuário.

        Examples:
        - Para texto: search_by_text_query(text_query="descrição", sex="female", min_age=50, primary_site="stomach")
        - Para imagem: search_by_image_query(top_k=5, sex="male", max_age=30, pathologic_stage="stage ii")
        
        IMPORTANT: If the user sends a very short message (like a single letter) along with an image,
        interpret it as a request to analyze the image and search for similar ones.
        """
        if not isinstance(original_instruction, types.Content):
            original_instruction = types.Content(role="system", parts=[types.Part(text=str(original_instruction))])
        if not original_instruction.parts:
            original_instruction.parts.append(types.Part(text=""))

        modified_text = prefix + (original_instruction.parts[0].text or "")
        original_instruction.parts[0].text = modified_text
        llm_request.config.system_instruction = original_instruction
        logger.info("System instruction updated with histopathology prefix (%s chars)", len(modified_text))

    return None






def simple_after_model_modifier(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Processa resposta do modelo e trata respostas vazias."""
    agent_name = callback_context.agent_name
    logger.info("simple_after_model_modifier invoked for agent=%s", agent_name)
    
    # Verificar chamadas de função e respostas de texto
    has_function_call = False
    has_text_response = False
    
    if llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
            if hasattr(part, 'text') and part.text:
                has_text_response = True
            elif hasattr(part, 'function_call') and part.function_call:
                has_function_call = True
    logger.info("Response analysis: text=%s function_call=%s error=%s", has_text_response, has_function_call, bool(llm_response.error_message))
    
    if agent_name == "histopathology_agent":
        # Deixar o ADK tratar erros
        if llm_response.error_message:
            logger.warning("LLM response contains error: %s", llm_response.error_message)
            return None
        
        # Deixar o ADK executar chamadas de ferramentas
        if has_function_call:
            logger.info("Function call detected in response; delegating to ADK")
            return None
        
        # Deixar o ADK tratar respostas de texto
        if has_text_response:
            logger.info("Text response detected; letting ADK continue")
            return None
        
        # Tratar respostas vazias
        if not has_function_call and not has_text_response and not llm_response.error_message:
            logger.warning("LLM response was empty; injecting friendly error message")
            error_content = types.Content(
                role="model",
                parts=[types.Part(text="Desculpe, encontrei um problema ao processar sua solicitação. A resposta do modelo está vazia. Por favor, tente novamente ou reformule sua pergunta.")]
            )
            llm_response.content = error_content
            return llm_response
    
            logger.info("simple_after_model_modifier completed for agent=%s", agent_name)
    return None

# TODO: Definir diferença de filtro para padrões morfológicos específicos
# TODO: Verificar MedGemma
histopathology_agent = LlmAgent(
        name="histopathology_agent",
        model="gemini-2.5-flash",
        instruction=f"""
        You are a helpful AI assistant specialized in histopathology image analysis.
        You can search for similar histology slide images using either an image query or a text description.
        
        When the user provides an image, use search_by_image_query(top_k=5) WITHOUT passing the image as a parameter.
        The image will be automatically extracted from the message context.
        
        When the user describes histological features, use search_by_text_query(text_query="descrição", top_k=5).

        Always honor demographic filters mentioned pelo usuário:
        - "mulher", "sexo feminino" ou similares ⇒ sex="female"
        - "homem", "sexo masculino" ou similares ⇒ sex="male"
        - "mais de 50 anos" ⇒ min_age=50
        - "menos de 30 anos" ⇒ max_age=30
        - "entre 40 e 60 anos" ⇒ min_age=40 e max_age=60
        Combine filtros conforme necessário tanto para imagens quanto texto.

        Localização anatômica e estágios clínicos também devem ser convertidos em argumentos das ferramentas:
        - primary_site, tissue_origin, site_of_resection para campos como "local primário", "tecido de origem", "sítio de biópsia".
        - tissue_type, specimen_type, disease_type para "tipo de tecido", "tipo de amostra", "tipo de doença".
        - pathologic_stage, ajcc_t, ajcc_n, ajcc_m para solicitações relacionadas a estágio AJCC/TNM.
        Passe strings consistentes com o que o usuário pediu (ex.: "Fundus of stomach", "Stage II").
        
        CRITICAL: NEVER include raw image data (base64 strings, file paths, URIs, or inline_data parts) in your responses.
        You must ONLY respond with plain text or tool results; do not echo user images or attach blobs.
        Only present the formatted search results returned by the tool functions with similarity percentages and image identifiers.
        """,
        tools=[search_by_image_query, search_by_text_query],
        before_agent_callback=on_before_agent,
        before_model_callback=before_model_modifier,
        after_model_callback = simple_after_model_modifier
    )

# Criar instância do agente ADK middleware
adk_histopathology_agent = ADKAgent(
    adk_agent=histopathology_agent,
    app_name="agents",
    user_id="demo_user",
    session_timeout_seconds=3600,
    use_in_memory_services=True
)
logger.info("ADKAgent initialized for app 'agents'")

# Criar aplicação FastAPI
app = FastAPI(title="ADK Middleware Histopathology Agent")

# Adicionar endpoint ADK
add_adk_fastapi_endpoint(app, adk_histopathology_agent, path="/")
logger.info("FastAPI endpoint '/' registered for histopathology agent")

if __name__ == "__main__":
    import os
    import uvicorn

    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY environment variable not set!")
        print("   Set it with: export GOOGLE_API_KEY='your-key-here'")
        print("   Get a key from: https://makersuite.google.com/app/apikey")
        print()

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

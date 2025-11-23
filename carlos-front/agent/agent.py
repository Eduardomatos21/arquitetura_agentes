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

agent_name = "histopathology_agent"


def on_before_agent(callback_context: CallbackContext):
    """Inicializa o estado do agente."""
    return None


def extract_and_convert_images_from_contents(contents: List[Content], user_content: Any = None) -> List[Content]:
    """
    Extrai imagens de mensagens AG-UI e converte para formato Part do Gemini.
    
    Processa mensagens do usuário que podem conter conteúdo multimodal no formato AG-UI:
    - BinaryInputContent com type="binary" e mimeType começando com "image/"
    - Converte base64 ou URLs para Part com inline_data (Blob)
    
    Args:
        contents: Lista de Content do LlmRequest
        user_content: Conteúdo original do usuário do callback_context (opcional)
        
    Returns:
        Lista de Content modificada com imagens convertidas para Parts
    """
    import requests
    from io import BytesIO
    
    modified_contents = []
    
    # Processar user_content se disponível
    multimodal_data = None
    user_content_parts = None
    if user_content is not None:
            # Extrair partes do objeto Content
            if isinstance(user_content, Content):
                if hasattr(user_content, 'parts') and user_content.parts:
                    user_content_parts = []
                    for part in user_content.parts:
                        # Extrair imagens de inline_data
                        if hasattr(part, 'inline_data') and part.inline_data:
                            blob = part.inline_data
                            if hasattr(blob, 'data') and blob.data:
                                mime_type = getattr(blob, 'mime_type', 'image/jpeg')
                                image_data = blob.data
                                new_blob = types.Blob(mime_type=mime_type, data=image_data)
                                user_content_parts.append(types.Part(inline_data=new_blob))
                        # Extrair texto e verificar se JSON contém imagens
                        elif hasattr(part, 'text') and part.text:
                            text_content = part.text
                            
                            try:
                                if text_content.strip().startswith(('[', '{')):
                                    parsed_json = json.loads(text_content)
                                    
                                    if isinstance(parsed_json, list):
                                        for item in parsed_json:
                                            if isinstance(item, dict):
                                                if item.get("type") == "text":
                                                    text_val = item.get("text", "")
                                                    if text_val:
                                                        user_content_parts.append(types.Part(text=text_val))
                                                
                                                elif item.get("type") == "binary" and item.get("mimeType", "").startswith("image/"):
                                                    mime_type = item.get("mimeType", "image/jpeg")
                                                    image_data_str = item.get("data", "")
                                                    
                                                    if image_data_str:
                                                        try:
                                                            image_data = base64.b64decode(image_data_str)
                                                            blob = types.Blob(mime_type=mime_type, data=image_data)
                                                            user_content_parts.append(types.Part(inline_data=blob))
                                                        except Exception:
                                                            pass
                                    else:
                                        user_content_parts.append(types.Part(text=text_content))
                                else:
                                    user_content_parts.append(types.Part(text=text_content))
                            except (json.JSONDecodeError, AttributeError):
                                user_content_parts.append(types.Part(text=text_content))
    
    for content in contents:
        # Processar apenas mensagens do usuário
        if content.role != "user":
            modified_contents.append(content)
            continue
        
        # Verificar se há partes que precisam ser processadas
        new_parts = []
        has_multimodal_content = False
        
        # Usar partes extraídas do objeto Content
        if user_content_parts:
            has_multimodal_content = True
            new_parts.extend(user_content_parts)
        
        # Processar multimodal_data de user_content
        elif multimodal_data and isinstance(multimodal_data, list):
            has_multimodal_content = True
            for item in multimodal_data:
                if isinstance(item, dict):
                    # Conteúdo de texto
                    if item.get("type") == "text":
                        text_content = item.get("text", "")
                        if text_content:
                            new_parts.append(types.Part(text=text_content))
                    
                    # Conteúdo binário (imagem)
                    elif item.get("type") == "binary" and item.get("mimeType", "").startswith("image/"):
                        mime_type = item.get("mimeType", "image/jpeg")
                        image_data = None
                        
                        # Tentar obter dados da imagem
                        if "data" in item:
                            # Dados em Base64
                            data_str = item["data"]
                            # Remover prefixo data:image/...;base64, se presente
                            if "," in data_str:
                                data_str = data_str.split(",", 1)[1]
                            try:
                                image_data = base64.b64decode(data_str)
                            except Exception:
                                continue
                        
                        elif "url" in item:
                            try:
                                response = requests.get(item["url"], timeout=30)
                                response.raise_for_status()
                                image_data = response.content
                            except Exception:
                                continue
                        
                        elif "id" in item:
                            continue
                        
                        if image_data:
                            blob = types.Blob(mime_type=mime_type, data=image_data)
                            new_parts.append(types.Part(inline_data=blob))
        
        # Processar partes de conteúdo existentes
        for part_idx, part in enumerate(content.parts):
            # Pular partes já processadas do objeto Content
            if user_content_parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    part_already_added = any(
                        hasattr(p, 'inline_data') and p.inline_data and 
                        p.inline_data.data == part.inline_data.data 
                        for p in new_parts
                    )
                    if part_already_added:
                        continue
                elif hasattr(part, 'text') and part.text:
                    text_already_added = any(
                        hasattr(p, 'text') and p.text == part.text 
                        for p in new_parts
                    )
                    if text_already_added:
                        continue
                    
                    try:
                        if part.text.strip().startswith(('[', '{')):
                            parsed = json.loads(part.text)
                            if isinstance(parsed, list):
                                has_text_in_new = any(hasattr(p, 'text') and p.text for p in new_parts)
                                has_image_in_new = any(hasattr(p, 'inline_data') and p.inline_data for p in new_parts)
                                if has_text_in_new and has_image_in_new:
                                    continue
                    except (json.JSONDecodeError, AttributeError):
                        pass
            
            # Manter partes que já possuem inline_data
            if hasattr(part, 'inline_data') and part.inline_data:
                new_parts.append(part)
                continue
            
            # Verificar se o texto contém conteúdo AG-UI
            if hasattr(part, 'text') and part.text:
                if has_multimodal_content and not user_content_parts:
                    if not any(p.text for p in new_parts if hasattr(p, 'text')):
                        new_parts.append(part)
                    continue
                
                try:
                    if part.text.strip().startswith(('[', '{')):
                        parsed = json.loads(part.text)
                        
                        if isinstance(parsed, list):
                            has_multimodal_content = True
                            for item in parsed:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        text_content = item.get("text", "")
                                        if text_content:
                                            new_parts.append(types.Part(text=text_content))
                                    
                                    elif item.get("type") == "binary" and item.get("mimeType", "").startswith("image/"):
                                        mime_type = item.get("mimeType", "image/jpeg")
                                        image_data = None
                                        
                                        if "data" in item:
                                            data_str = item["data"]
                                            if "," in data_str:
                                                data_str = data_str.split(",", 1)[1]
                                            try:
                                                image_data = base64.b64decode(data_str)
                                            except Exception:
                                                continue
                                        
                                        elif "url" in item:
                                            try:
                                                response = requests.get(item["url"], timeout=30)
                                                response.raise_for_status()
                                                image_data = response.content
                                            except Exception:
                                                continue
                                        
                                        elif "id" in item:
                                            continue
                                        
                                        if image_data:
                                            blob = types.Blob(mime_type=mime_type, data=image_data)
                                            new_parts.append(types.Part(inline_data=blob))
                        else:
                            new_parts.append(part)
                    else:
                        new_parts.append(part)
                except (json.JSONDecodeError, AttributeError):
                    new_parts.append(part)
            else:
                new_parts.append(part)
        
        # Criar Content modificado se conteúdo multimodal foi processado
        if has_multimodal_content and new_parts:
            try:
                modified_content = types.Content(role=content.role, parts=new_parts)
                modified_contents.append(modified_content)
            except Exception as e:
                print(f"❌ Erro ao criar Content modificado: {e}")
                modified_contents.append(content)
        else:
            modified_contents.append(content)
    
    return modified_contents


def before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Processa conteúdo multimodal e modifica instrução do sistema."""
    agent_name = callback_context.agent_name
    
    # Processar conteúdo multimodal (imagens)
    if llm_request.contents:
        user_content = getattr(callback_context, 'user_content', None)
        
        try:
            modified_contents = extract_and_convert_images_from_contents(
                llm_request.contents, 
                user_content=user_content
            )
            
            # Cachear imagens para acesso das ferramentas
            from tools import _IMAGE_CACHE
            images_cached = 0
            for content in modified_contents:
                if content.parts:
                    for part in content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            blob = part.inline_data
                            if hasattr(blob, 'data') and blob.data:
                                mime_type = getattr(blob, 'mime_type', 'image/png')
                                image_b64 = base64.b64encode(blob.data).decode('utf-8')
                                image_data_uri = f"data:{mime_type};base64,{image_b64}"
                                cache_key = image_b64[:400]
                                _IMAGE_CACHE[cache_key] = image_data_uri
                                images_cached += 1
            
            if images_cached > 0:
                print(f"✅ {images_cached} imagem(ns) processada(s) e cacheada(s) com sucesso")
            
            llm_request.contents = modified_contents
        except Exception as e:
            print(f"❌ Erro ao processar conteúdo multimodal: {e}")
            import traceback
            traceback.print_exc()
    
    if agent_name == "histopathology_agent":
        original_instruction = llm_request.config.system_instruction or types.Content(role="system", parts=[])
        prefix = f"""You are a helpful AI assistant specialized in histopathology image analysis.
        You can search for similar histology slide images using either an image query or a text description.
        
        When the user provides an image, use the search_by_image_query tool WITHOUT any image parameter.
        The image will be automatically extracted from the user's message context.
        Just call: search_by_image_query(top_k=5)
        
        When the user describes histological features in text, use search_by_text_query with the text description.
        
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

    return None






def simple_after_model_modifier(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Processa resposta do modelo e trata respostas vazias."""
    agent_name = callback_context.agent_name
    
    # Verificar chamadas de função e respostas de texto
    has_function_call = False
    has_text_response = False
    
    if llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
            if hasattr(part, 'text') and part.text:
                has_text_response = True
            elif hasattr(part, 'function_call') and part.function_call:
                has_function_call = True
    
    if agent_name == "histopathology_agent":
        # Deixar o ADK tratar erros
        if llm_response.error_message:
            return None
        
        # Deixar o ADK executar chamadas de ferramentas
        if has_function_call:
            return None
        
        # Deixar o ADK tratar respostas de texto
        if has_text_response:
            return None
        
        # Tratar respostas vazias
        if not has_function_call and not has_text_response and not llm_response.error_message:
            error_content = types.Content(
                role="model",
                parts=[types.Part(text="Desculpe, encontrei um problema ao processar sua solicitação. A resposta do modelo está vazia. Por favor, tente novamente ou reformule sua pergunta.")]
            )
            llm_response.content = error_content
            return llm_response
    
    return None


histopathology_agent = LlmAgent(
        name="histopathology_agent",
        model="gemini-2.5-flash",
        instruction=f"""
        You are a helpful AI assistant specialized in histopathology image analysis.
        You can search for similar histology slide images using either an image query or a text description.
        
        When the user provides an image, use search_by_image_query(top_k=5) WITHOUT passing the image as a parameter.
        The image will be automatically extracted from the message context.
        
        When the user describes histological features, use search_by_text_query(query="description", top_k=5).
        
        CRITICAL: NEVER include raw image data (base64 strings, file paths, or URIs) in your responses to the user.
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

# Criar aplicação FastAPI
app = FastAPI(title="ADK Middleware Histopathology Agent")

# Adicionar endpoint ADK
add_adk_fastapi_endpoint(app, adk_histopathology_agent, path="/")

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

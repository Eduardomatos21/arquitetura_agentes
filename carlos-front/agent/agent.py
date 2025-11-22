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
from enum import Enum
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
from google.genai.types import Content, Part , FunctionDeclaration
from google.adk.models import LlmResponse, LlmRequest
from google.genai import types
from tools import search_by_image_query, search_by_text_query

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

agent_name = "histopathology_agent"

# TODO: change the name of the class to the name of the agent
class ProverbsState(BaseModel):
    """List of the proverbs being written."""
    proverbs: list[str] = Field(
        default_factory=list,
        description='The list of already written proverbs',
    )


def on_before_agent(callback_context: CallbackContext):
    """
    Initialize proverbs state if it doesn't exist.
    """
    print(f"\n{'='*80}")
    print(f"🚀 [FLUXO] on_before_agent chamado")
    print(f"{'='*80}")
    print(f"  📌 Agent: {callback_context.agent_name}")
    print(f"  📌 Invocation ID: {callback_context.invocation_id}")
    
    # Verificar estado de forma segura
    try:
        if hasattr(callback_context.state, 'keys'):
            state_keys = list(callback_context.state.keys())
        elif hasattr(callback_context.state, '__dict__'):
            state_keys = list(callback_context.state.__dict__.keys())
        else:
            state_keys = [str(k) for k in dir(callback_context.state) if not k.startswith('_')]
        print(f"  📌 State keys: {state_keys}")
    except Exception as e:
        print(f"  ⚠️  Não foi possível listar state keys: {e}")
        print(f"  📌 State type: {type(callback_context.state)}")
    
    if "proverbs" not in callback_context.state:
        # Initialize with default recipe
        default_proverbs = []
        callback_context.state["proverbs"] = default_proverbs
        print(f"  ✅ Estado 'proverbs' inicializado")
    else:
        print(f"  ℹ️  Estado 'proverbs' já existe: {len(callback_context.state.get('proverbs', []))} itens")
    
    print(f"{'='*80}\n")
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
    
    print(f"  🔧 extract_and_convert_images_from_contents chamada")
    print(f"    - contents count: {len(contents)}")
    print(f"    - user_content: {type(user_content)} - {user_content is not None}")
    
    modified_contents = []
    
    # Tentar processar user_content primeiro se disponível
    multimodal_data = None
    user_content_parts = None  # Para armazenar partes extraídas de Content objects
    if user_content is not None:
        print(f"    🔍 Processando user_content...")
        try:
            # Se user_content é um Content object, extrair suas partes diretamente
            if isinstance(user_content, Content):
                print(f"      - user_content é Content object")
                if hasattr(user_content, 'parts') and user_content.parts:
                    print(f"      - ✅ Content tem {len(user_content.parts)} partes")
                    user_content_parts = []
                    for part_idx, part in enumerate(user_content.parts):
                        print(f"      - 🔍 Analisando Part {part_idx} do user_content...")
                        # Extrair imagens de inline_data
                        if hasattr(part, 'inline_data') and part.inline_data:
                            blob = part.inline_data
                            if hasattr(blob, 'data') and blob.data:
                                mime_type = getattr(blob, 'mime_type', 'image/jpeg')
                                image_data = blob.data
                                new_blob = types.Blob(
                                    mime_type=mime_type,
                                    data=image_data
                                )
                                user_content_parts.append(types.Part(inline_data=new_blob))
                                print(f"      - ✅ Imagem extraída do Content: {mime_type}, tamanho: {len(image_data)} bytes")
                            else:
                                print(f"      - ⚠️ Part {part_idx} tem inline_data mas sem data")
                        # Extrair texto - mas verificar se é JSON com imagens
                        elif hasattr(part, 'text') and part.text:
                            text_content = part.text
                            
                            # Verificar se o texto é JSON que contém imagens
                            try:
                                if text_content.strip().startswith('[') or text_content.strip().startswith('{'):
                                    parsed_json = json.loads(text_content)
                                    
                                    # Se é um array (formato AG-UI multimodal)
                                    if isinstance(parsed_json, list):
                                        print(f"      - ✅ Texto é JSON array com {len(parsed_json)} itens")
                                        for item in parsed_json:
                                            if isinstance(item, dict):
                                                # TextInputContent
                                                if item.get("type") == "text":
                                                    text_val = item.get("text", "")
                                                    if text_val:
                                                        user_content_parts.append(types.Part(text=text_val))
                                                        print(f"      - ✅ Texto extraído do JSON: {len(text_val)} chars")
                                                
                                                # BinaryInputContent (imagem)
                                                elif item.get("type") == "binary" and item.get("mimeType", "").startswith("image/"):
                                                    mime_type = item.get("mimeType", "image/jpeg")
                                                    image_data_str = item.get("data", "")
                                                    
                                                    if image_data_str:
                                                        try:
                                                            # Decodificar base64
                                                            image_data = base64.b64decode(image_data_str)
                                                            # Criar Blob
                                                            blob = types.Blob(
                                                                mime_type=mime_type,
                                                                data=image_data
                                                            )
                                                            user_content_parts.append(types.Part(inline_data=blob))
                                                            print(f"      - ✅ Imagem extraída do JSON: {mime_type}, tamanho: {len(image_data)} bytes")
                                                        except Exception as e:
                                                            print(f"      - ❌ Erro ao decodificar imagem do JSON: {e}")
                                    else:
                                        # Não é array, tratar como texto normal
                                        user_content_parts.append(types.Part(text=text_content))
                                        print(f"      - ✅ Texto extraído do Content: {len(text_content)} chars")
                                else:
                                    # Não é JSON, tratar como texto normal
                                    user_content_parts.append(types.Part(text=text_content))
                                    print(f"      - ✅ Texto extraído do Content: {len(text_content)} chars")
                            except (json.JSONDecodeError, AttributeError) as e:
                                # Não é JSON válido, tratar como texto normal
                                user_content_parts.append(types.Part(text=text_content))
                                print(f"      - ✅ Texto extraído do Content (não-JSON): {len(text_content)} chars")
                        else:
                            # Log para debug - ver o que mais pode ter na parte
                            part_attrs = [attr for attr in dir(part) if not attr.startswith('_')]
                            print(f"      - ⚠️ Part {part_idx} não tem inline_data nem text. Atributos: {part_attrs}")
                else:
                    print(f"      - ⚠️ Content não tem partes ou está vazio")
            # Se user_content é uma string, tentar parsear como JSON
            elif isinstance(user_content, str):
                print(f"      - user_content é string, length: {len(user_content)}")
                if user_content.strip().startswith('[') or user_content.strip().startswith('{'):
                    print(f"      - Tentando parsear como JSON...")
                    multimodal_data = json.loads(user_content)
                    print(f"      - ✅ JSON parseado: {type(multimodal_data)}")
            # Se user_content já é um dict ou list
            elif isinstance(user_content, (dict, list)):
                print(f"      - user_content já é {type(user_content)}")
                multimodal_data = user_content
            else:
                print(f"      - ⚠️ user_content é {type(user_content)}, não processado")
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"      - ❌ Erro ao processar user_content: {e}")
            import traceback
            traceback.print_exc()
    
    for content in contents:
        # Processar apenas mensagens do usuário
        if content.role != "user":
            modified_contents.append(content)
            continue
        
        # Verificar se há partes que precisam ser processadas
        new_parts = []
        has_multimodal_content = False
        
        # Se temos partes extraídas de um Content object, usar diretamente
        if user_content_parts:
            print(f"    ✅ Usando {len(user_content_parts)} partes extraídas do Content object")
            has_multimodal_content = True
            new_parts.extend(user_content_parts)
        
        # Se temos dados multimodais do user_content, processar primeiro
        elif multimodal_data and isinstance(multimodal_data, list):
            print(f"    ✅ Encontrado multimodal_data do user_content com {len(multimodal_data)} itens")
            has_multimodal_content = True
            for item in multimodal_data:
                if isinstance(item, dict):
                    # TextInputContent
                    if item.get("type") == "text":
                        text_content = item.get("text", "")
                        if text_content:
                            new_parts.append(types.Part(text=text_content))
                    
                    # BinaryInputContent (imagem)
                    elif item.get("type") == "binary" and item.get("mimeType", "").startswith("image/"):
                        mime_type = item.get("mimeType", "image/jpeg")
                        image_data = None
                        
                        # Tentar obter dados da imagem
                        if "data" in item:
                            # Base64 data
                            data_str = item["data"]
                            # Remover prefixo data:image/...;base64, se presente
                            if "," in data_str:
                                data_str = data_str.split(",", 1)[1]
                            try:
                                image_data = base64.b64decode(data_str)
                            except Exception as e:
                                print(f"⚠️ Erro ao decodificar base64: {e}")
                                continue
                        
                        elif "url" in item:
                            # URL da imagem
                            try:
                                response = requests.get(item["url"], timeout=30)
                                response.raise_for_status()
                                image_data = response.content
                            except Exception as e:
                                print(f"⚠️ Erro ao baixar imagem de URL: {e}")
                                continue
                        
                        elif "id" in item:
                            # ID de conteúdo pré-carregado (não suportado ainda)
                            print(f"⚠️ Referência por ID não suportada ainda: {item['id']}")
                            continue
                        
                        if image_data:
                            # Criar Part com inline_data
                            blob = types.Blob(
                                mime_type=mime_type,
                                data=image_data
                            )
                            new_parts.append(types.Part(inline_data=blob))
                            print(f"✅ Imagem convertida do user_content: {mime_type}, tamanho: {len(image_data)} bytes")
        
        # Processar partes existentes do content
        print(f"    🔍 Processando {len(content.parts)} partes do content...")
        for part_idx, part in enumerate(content.parts):
            # Se já temos partes do Content object, verificar se esta parte já foi incluída
            if user_content_parts:
                # Se a parte já tem inline_data e já foi processada, pular
                if hasattr(part, 'inline_data') and part.inline_data:
                    # Verificar se já temos uma parte similar
                    part_already_added = any(
                        hasattr(p, 'inline_data') and p.inline_data and 
                        p.inline_data.data == part.inline_data.data 
                        for p in new_parts
                    )
                    if part_already_added:
                        print(f"      - Part {part_idx}: já processada do Content object, pulando")
                        continue
                # Se é texto e já temos texto das partes do Content, verificar se é o mesmo JSON
                elif hasattr(part, 'text') and part.text:
                    # Verificar se é o mesmo texto exato
                    text_already_added = any(
                        hasattr(p, 'text') and p.text == part.text 
                        for p in new_parts
                    )
                    if text_already_added:
                        print(f"      - Part {part_idx}: texto já processado do Content object, pulando")
                        continue
                    
                    # Verificar se o texto é JSON que já foi processado
                    try:
                        if part.text.strip().startswith('[') or part.text.strip().startswith('{'):
                            parsed = json.loads(part.text)
                            if isinstance(parsed, list):
                                # Verificar se já processamos este JSON (comparando itens)
                                json_already_processed = False
                                for p in new_parts:
                                    # Se já temos partes de texto e imagem que correspondem ao JSON, pular
                                    if hasattr(p, 'text') and hasattr(p, 'inline_data'):
                                        # Provavelmente já processamos
                                        json_already_processed = True
                                        break
                                
                                # Se já temos texto e imagem nas new_parts, e este é o JSON original, pular
                                has_text_in_new = any(hasattr(p, 'text') and p.text for p in new_parts)
                                has_image_in_new = any(hasattr(p, 'inline_data') and p.inline_data for p in new_parts)
                                if has_text_in_new and has_image_in_new:
                                    print(f"      - Part {part_idx}: JSON já processado do user_content, pulando")
                                    continue
                    except (json.JSONDecodeError, AttributeError):
                        pass  # Não é JSON, continuar processamento normal
            
            # Se a parte já tem inline_data, manter como está
            if hasattr(part, 'inline_data') and part.inline_data:
                print(f"      - Part {part_idx}: já tem inline_data, mantendo")
                new_parts.append(part)
                continue
            
            # Verificar se é texto que pode conter referência a conteúdo multimodal
            if hasattr(part, 'text') and part.text:
                # Se já processamos user_content, não processar novamente
                if has_multimodal_content and not user_content_parts:
                    # Se não há texto no user_content, adicionar o texto da parte
                    if not any(p.text for p in new_parts if hasattr(p, 'text')):
                        new_parts.append(part)
                    continue
                
                # Tentar parsear como JSON para verificar se é conteúdo AG-UI
                try:
                    # Se o texto parece ser JSON com estrutura AG-UI
                    if part.text.strip().startswith('[') or part.text.strip().startswith('{'):
                        parsed = json.loads(part.text)
                        
                        # Se é um array (formato AG-UI multimodal)
                        if isinstance(parsed, list):
                            has_multimodal_content = True
                            for item in parsed:
                                if isinstance(item, dict):
                                    # TextInputContent
                                    if item.get("type") == "text":
                                        text_content = item.get("text", "")
                                        if text_content:
                                            new_parts.append(types.Part(text=text_content))
                                    
                                    # BinaryInputContent (imagem)
                                    elif item.get("type") == "binary" and item.get("mimeType", "").startswith("image/"):
                                        mime_type = item.get("mimeType", "image/jpeg")
                                        image_data = None
                                        
                                        # Tentar obter dados da imagem
                                        if "data" in item:
                                            # Base64 data
                                            data_str = item["data"]
                                            # Remover prefixo data:image/...;base64, se presente
                                            if "," in data_str:
                                                data_str = data_str.split(",", 1)[1]
                                            try:
                                                image_data = base64.b64decode(data_str)
                                            except Exception as e:
                                                print(f"⚠️ Erro ao decodificar base64: {e}")
                                                continue
                                        
                                        elif "url" in item:
                                            # URL da imagem
                                            try:
                                                response = requests.get(item["url"], timeout=30)
                                                response.raise_for_status()
                                                image_data = response.content
                                            except Exception as e:
                                                print(f"⚠️ Erro ao baixar imagem de URL: {e}")
                                                continue
                                        
                                        elif "id" in item:
                                            # ID de conteúdo pré-carregado (não suportado ainda)
                                            print(f"⚠️ Referência por ID não suportada ainda: {item['id']}")
                                            continue
                                        
                                        if image_data:
                                            # Criar Part com inline_data
                                            blob = types.Blob(
                                                mime_type=mime_type,
                                                data=image_data
                                            )
                                            new_parts.append(types.Part(inline_data=blob))
                                            print(f"✅ Imagem convertida: {mime_type}, tamanho: {len(image_data)} bytes")
                        else:
                            # Não é formato multimodal, manter texto original
                            new_parts.append(part)
                    else:
                        # Não é JSON, manter texto original
                        new_parts.append(part)
                except (json.JSONDecodeError, AttributeError):
                    # Não é JSON válido, manter texto original
                    new_parts.append(part)
            else:
                # Não é texto, manter parte original
                new_parts.append(part)
        
        # Se processamos conteúdo multimodal, criar novo Content
        if has_multimodal_content and new_parts:
            print(f"    ✅ Criando Content modificado com {len(new_parts)} partes")
            try:
                modified_content = types.Content(
                    role=content.role,
                    parts=new_parts
                )
                modified_contents.append(modified_content)
                print(f"    ✅ Content modificado criado com sucesso")
            except Exception as e:
                print(f"    ❌ Erro ao criar Content modificado: {e}")
                # Em caso de erro, manter conteúdo original
                modified_contents.append(content)
        else:
            # Manter conteúdo original
            if has_multimodal_content:
                print(f"    ⚠️ has_multimodal_content=True mas new_parts está vazio")
            modified_contents.append(content)
    
    return modified_contents


# --- Define the Callback Function ---
#  modifying the agent's system prompt to incude the current state of the proverbs list
def before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Inspects/modifies the LLM request or skips the call."""
    agent_name = callback_context.agent_name
    
    print(f"\n{'='*80}")
    print(f"🔍 [FLUXO] before_model_modifier chamado")
    print(f"{'='*80}")
    print(f"  📌 Agent: {agent_name}")
    print(f"  📌 Invocation ID: {callback_context.invocation_id}")
    
    # Processar conteúdo multimodal (imagens) para todos os agentes
    if llm_request.contents:
        print(f"📦 llm_request.contents tem {len(llm_request.contents)} conteúdo(s)")
        
        # Log detalhado de cada conteúdo
        for i, content in enumerate(llm_request.contents):
            print(f"\n  📄 Content {i}:")
            print(f"    - role: {content.role}")
            print(f"    - parts: {len(content.parts) if content.parts else 0}")
            
            if content.parts:
                for j, part in enumerate(content.parts):
                    print(f"    - Part {j}:")
                    if hasattr(part, 'text') and part.text:
                        text_preview = part.text[:200] + "..." if len(part.text) > 200 else part.text
                        print(f"      - type: text")
                        print(f"      - length: {len(part.text)}")
                        print(f"      - preview: {text_preview}")
                        # Verificar se parece JSON
                        if part.text.strip().startswith(('[', '{')):
                            print(f"      - ⚠️ Parece ser JSON!")
                            try:
                                parsed = json.loads(part.text)
                                print(f"      - JSON válido: {type(parsed)}")
                                if isinstance(parsed, list):
                                    print(f"      - Array com {len(parsed)} itens")
                                    for k, item in enumerate(parsed):
                                        if isinstance(item, dict):
                                            print(f"        - Item {k}: type={item.get('type')}, mimeType={item.get('mimeType', 'N/A')}")
                            except:
                                print(f"      - ❌ Não é JSON válido")
                    elif hasattr(part, 'inline_data') and part.inline_data:
                        blob = part.inline_data
                        mime_type = getattr(blob, 'mime_type', 'N/A')
                        data_size = len(blob.data) if hasattr(blob, 'data') and blob.data else 0
                        print(f"      - type: inline_data (imagem)")
                        print(f"      - mime_type: {mime_type}")
                        print(f"      - data_size: {data_size} bytes")
                        print(f"      - ✅ IMAGEM ENCONTRADA NO CONTENTS!")
                    else:
                        print(f"      - type: {type(part)}")
                        print(f"      - attributes: {dir(part)}")
        
        # Obter user_content se disponível
        user_content = None
        if hasattr(callback_context, 'user_content'):
            user_content = callback_context.user_content
            print(f"\n📋 callback_context.user_content:")
            print(f"    - type: {type(user_content)}")
            if user_content:
                user_content_str = str(user_content)
                if len(user_content_str) > 200:
                    print(f"    - preview: {user_content_str[:200]}...")
                else:
                    print(f"    - content: {user_content_str}")
            else:
                print(f"    - value: None")
        else:
            print(f"\n❌ callback_context não tem user_content")
        
        try:
            print(f"\n🔄 Chamando extract_and_convert_images_from_contents...")
            modified_contents = extract_and_convert_images_from_contents(
                llm_request.contents, 
                user_content=user_content
            )
            print(f"✅ Conversão concluída: {len(modified_contents)} conteúdo(s) modificado(s)")
            
            # Log do resultado e armazenar imagens no cache para acesso pela função tool
            from tools import _IMAGE_CACHE
            for i, content in enumerate(modified_contents):
                if content.parts:
                    for j, part in enumerate(content.parts):
                        if hasattr(part, 'inline_data') and part.inline_data:
                            print(f"  ✅ Imagem encontrada no Content {i}, Part {j}")
                            # Armazenar imagem no cache para acesso pela função tool
                            blob = part.inline_data
                            if hasattr(blob, 'data') and blob.data:
                                mime_type = getattr(blob, 'mime_type', 'image/png')
                                image_b64 = base64.b64encode(blob.data).decode('utf-8')
                                image_data_uri = f"data:{mime_type};base64,{image_b64}"
                                # Criar chave de cache usando o conteúdo base64 (ignorando tipo de imagem)
                                # Usamos os primeiros ~400 chars do base64 como chave (ADK trunca em ~512)
                                cache_key = image_b64[:400]  # Primeiros 400 chars do base64 puro
                                _IMAGE_CACHE[cache_key] = image_data_uri
                                print(f"  💾 Imagem armazenada no cache (chave base64: {len(cache_key)} chars, total: {len(image_b64)} chars)")
            
            llm_request.contents = modified_contents
        except Exception as e:
            print(f"❌ Erro ao processar conteúdo multimodal: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ llm_request.contents está vazio ou None")
    
    # Log final dos contents que serão enviados
    print(f"\n📤 CONTEÚDO FINAL A SER ENVIADO AO GEMINI:")
    if llm_request.contents:
        for i, content in enumerate(llm_request.contents):
            print(f"  Content {i}: role={content.role}, parts={len(content.parts) if content.parts else 0}")
            if content.parts:
                for j, part in enumerate(content.parts):
                    if hasattr(part, 'text') and part.text:
                        preview = part.text[:100] + "..." if len(part.text) > 100 else part.text
                        print(f"    Part {j}: text (len={len(part.text)}) - {preview}")
                    elif hasattr(part, 'inline_data') and part.inline_data:
                        mime = getattr(part.inline_data, 'mime_type', 'N/A')
                        size = len(part.inline_data.data) if hasattr(part.inline_data, 'data') else 0
                        print(f"    Part {j}: inline_data ({mime}, {size} bytes)")
                    else:
                        print(f"    Part {j}: {type(part)}")
    
    print(f"{'='*80}\n")
    
    if agent_name == "histopathology_agent":
        print(f"  🔧 Modificando system_instruction para histopathology_agent...")
        # --- Modification Example ---
        # Add a prefix to the system instruction
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
        # Ensure system_instruction is Content and parts list exists
        if not isinstance(original_instruction, types.Content):
            # Handle case where it might be a string (though config expects Content)
            original_instruction = types.Content(role="system", parts=[types.Part(text=str(original_instruction))])
        if not original_instruction.parts:
            original_instruction.parts.append(types.Part(text="")) # Add an empty part if none exist

        # Modify the text of the first part
        modified_text = prefix + (original_instruction.parts[0].text or "")
        original_instruction.parts[0].text = modified_text
        llm_request.config.system_instruction = original_instruction
        print(f"  ✅ System instruction modificada")

    print(f"  ➡️  Enviando requisição para o modelo LLM...")
    print(f"{'='*80}\n")
    return None






# --- Define the Callback Function ---
def simple_after_model_modifier(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Process model response and handle function calls appropriately"""
    print(f"\n{'='*80}")
    print(f"📥 [FLUXO] simple_after_model_modifier chamado")
    print(f"{'='*80}")
    print(f"  📌 Agent: {callback_context.agent_name}")
    print(f"  📌 Invocation ID: {callback_context.invocation_id}")
    
    agent_name = callback_context.agent_name
    
    # Verificar se há function_call (tool call) pendente
    has_function_call = False
    has_text_response = False
    
    # Log detalhado da resposta do modelo
    print(f"  🔍 DEBUG llm_response completo:")
    print(f"    - type: {type(llm_response)}")
    print(f"    - dir: {[attr for attr in dir(llm_response) if not attr.startswith('_')]}")
    
    # Verificar todos os atributos importantes
    if hasattr(llm_response, 'content'):
        print(f"    - content: {llm_response.content}")
    if hasattr(llm_response, 'error_message'):
        print(f"    - error_message: {llm_response.error_message}")
    if hasattr(llm_response, 'error_code'):
        print(f"    - error_code: {llm_response.error_code}")
    if hasattr(llm_response, 'finish_reason'):
        print(f"    - finish_reason: {llm_response.finish_reason}")
    if hasattr(llm_response, 'interrupted'):
        print(f"    - interrupted: {llm_response.interrupted}")
    if hasattr(llm_response, 'usage_metadata'):
        print(f"    - usage_metadata: {llm_response.usage_metadata}")
    if hasattr(llm_response, 'cache_metadata'):
        print(f"    - cache_metadata: {llm_response.cache_metadata}")
    if hasattr(llm_response, 'candidates'):
        print(f"    - candidates: {llm_response.candidates}")
    
    # Log TODOS os atributos não-privados com valores
    print(f"\n  📊 TODOS OS ATRIBUTOS DO llm_response:")
    for attr in dir(llm_response):
        if not attr.startswith('_') and not callable(getattr(llm_response, attr, None)):
            try:
                val = getattr(llm_response, attr, None)
                if val is not None:
                    val_str = str(val)
                    if len(val_str) > 200:
                        val_str = val_str[:200] + "..."
                    print(f"    - {attr}: {val_str}")
            except Exception as e:
                print(f"    - {attr}: <erro ao acessar: {e}>")
    
    # Log da resposta do modelo
    if llm_response.content:
        print(f"  📄 Resposta do modelo recebida:")
        print(f"    - role: {llm_response.content.role}")
        if llm_response.content.parts:
            print(f"    - parts: {len(llm_response.content.parts)}")
            for i, part in enumerate(llm_response.content.parts):
                print(f"      - Part {i}: {type(part)}")
                print(f"        - dir: {[attr for attr in dir(part) if not attr.startswith('_')]}")
                
                if hasattr(part, 'text') and part.text:
                    has_text_response = True
                    text_preview = part.text[:150] + "..." if len(part.text) > 150 else part.text
                    print(f"        - text (len={len(part.text)})")
                    print(f"        - preview: {text_preview}")
                elif hasattr(part, 'function_call') and part.function_call:
                    has_function_call = True
                    func_name = part.function_call.name if hasattr(part.function_call, 'name') else 'N/A'
                    print(f"        - function_call")
                    print(f"          - name: {func_name}")
                    print(f"        ⚠️  Gemini quer chamar a tool: {func_name}")
                else:
                    # Log todos os atributos para debug
                    for attr in dir(part):
                        if not attr.startswith('_'):
                            val = getattr(part, attr, None)
                            if val is not None and not callable(val):
                                print(f"        - {attr}: {val}")
        else:
            print(f"    - parts: None ou vazio")
    elif llm_response.error_message:
        print(f"  ❌ Erro na resposta: {llm_response.error_message}")
    else:
        print(f"  ⚠️  Resposta sem conteúdo e sem erro")
    
    # --- Inspection ---
    if agent_name == "histopathology_agent":
        print(f"  🔧 Processando resposta para histopathology_agent...")
        
        # Se há erro, retornar None para deixar o ADK lidar
        if llm_response.error_message:
            print(f"  ⚠️  Erro detectado, retornando None para ADK processar")
            return None
        
        # Se há function_call, NÃO finalizar invocação - deixar a tool ser executada
        if has_function_call:
            print(f"  🔧 Function call detectado - NÃO finalizando invocação")
            print(f"  ➡️  Deixando o ADK executar a tool...")
            return None  # Retorna None para permitir execução da tool
        
        # Se há resposta de texto (resposta final do modelo)
        if has_text_response and llm_response.content and llm_response.content.parts:
            if llm_response.content.role == 'model' and llm_response.content.parts[0].text:
                original_text = llm_response.content.parts[0].text
                print(f"  ✅ Texto extraído da resposta (len={len(original_text)})")
                print(f"  🛑 Invocação finalizada - resposta enviada ao usuário")
                return None
        
        # Se não há nem function_call nem texto nem erro, algo está errado
        if not has_function_call and not has_text_response and not llm_response.error_message:
            print(f"  ❌ PROBLEMA: Resposta vazia sem function_call, texto ou erro!")
            print(f"  🔧 Criando resposta de erro para o usuário...")
            
            # Criar uma resposta de erro informativa para o usuário
            error_content = types.Content(
                role="model",
                parts=[types.Part(text="Desculpe, encontrei um problema ao processar sua solicitação. A resposta do modelo está vazia. Por favor, tente novamente ou reformule sua pergunta.")]
            )
            llm_response.content = error_content
            print(f"  ✅ Resposta de erro criada")
            print(f"  ➡️  Retornando resposta de erro ao ADK")
            return llm_response
        
        print(f"  ℹ️  Nenhuma ação necessária - deixando ADK processar")
    
    print(f"{'='*80}\n")
    return None


print(f"\n{'='*80}")
print(f"🔧 [INICIALIZAÇÃO] Criando LlmAgent...")
print(f"{'='*80}")

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

print(f"  ✅ LlmAgent criado:")
print(f"    - name: {histopathology_agent.name}")
print(f"    - model: gemini-2.5-flash")
print(f"    - tools: {len(histopathology_agent.tools)} ({[t.name if hasattr(t, 'name') else str(t) for t in histopathology_agent.tools]})")
print(f"    - callbacks: before_agent, before_model, after_model")
print(f"{'='*80}\n")

print(f"\n{'='*80}")
print(f"🔧 [INICIALIZAÇÃO] Criando ADKAgent...")
print(f"{'='*80}")

# Create ADK middleware agent instance
adk_histopathology_agent = ADKAgent(
    adk_agent=histopathology_agent,
    app_name="agents",
    user_id="demo_user",
    session_timeout_seconds=3600,
    use_in_memory_services=True
)

print(f"  ✅ ADKAgent criado:")
print(f"    - app_name: {histopathology_agent.name}")
print(f"    - user_id: demo_user")
print(f"    - session_timeout: 3600s")
print(f"    - use_in_memory_services: True")
print(f"{'='*80}\n")

# Create FastAPI app
print(f"\n{'='*80}")
print(f"🔧 [INICIALIZAÇÃO] Criando FastAPI app...")
print(f"{'='*80}")

app = FastAPI(title="ADK Middleware Proverbs Agent")

# Add the ADK endpoint
add_adk_fastapi_endpoint(app, adk_histopathology_agent, path="/")

print(f"  ✅ FastAPI app criado")
print(f"  ✅ Endpoint ADK adicionado em: /")
print(f"{'='*80}\n")

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

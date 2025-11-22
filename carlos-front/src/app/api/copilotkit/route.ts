import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { HttpAgent } from "@ag-ui/client";
import { NextRequest } from "next/server";
 
// 1. You can use any service adapter here for multi-agent support. We use
//    the empty adapter since we're only using one agent.
const serviceAdapter = new ExperimentalEmptyAdapter();
 
// 2. Create the CopilotRuntime instance and utilize the AG-UI client
//    to setup the connection with the ADK agent.
const runtime = new CopilotRuntime({
  agents: {
    // Our FastAPI endpoint URL
    // TODO: change the name of the agent to the name of the agent and pass the URL to env variable
    "histopathology_agent": new HttpAgent({url: "http://localhost:8000/"}),
  }   
});
 
// 3. Build a Next.js API route that handles the CopilotKit runtime requests.
export const POST = async (req: NextRequest) => {
  // Debug logging to see what CopilotKit is sending
  try {
    const body = await req.json();
    console.log("🔍 [CopilotKit API] Request received:");
    console.log("  - Body type:", typeof body);
    console.log("  - Body keys:", Object.keys(body || {}));
    console.log("  - Operation name:", body?.operationName || 'N/A');
    console.log("  - Query preview:", body?.query?.substring(0, 200) || 'N/A');
    
    // CopilotKit uses GraphQL - check variables
    if (body?.variables) {
      console.log("  - GraphQL variables keys:", Object.keys(body.variables));
      
      // Check for messages in variables
      if (body.variables.messages) {
        console.log("  - Messages count:", body.variables.messages.length);
        body.variables.messages.forEach((msg: any, idx: number) => {
          console.log(`  - Message ${idx}:`, {
            role: msg.role,
            content: typeof msg.content === 'string' 
              ? msg.content.substring(0, 100) + (msg.content.length > 100 ? '...' : '')
              : Array.isArray(msg.content)
              ? `Array[${msg.content.length}] with types: ${msg.content.map((c: any) => c.type || typeof c).join(', ')}`
              : typeof msg.content,
            hasImage: Array.isArray(msg.content) 
              ? msg.content.some((c: any) => c.type === 'image' || c.type === 'binary' || c.mimeType?.startsWith('image/'))
              : false
          });
          
          // Log image details if present
          if (Array.isArray(msg.content)) {
            msg.content.forEach((content: any, cIdx: number) => {
              if (content.type === 'image' || content.type === 'binary' || content.mimeType?.startsWith('image/')) {
                console.log(`    🖼️  Image content ${cIdx}:`, {
                  type: content.type,
                  mimeType: content.mimeType,
                  hasData: !!content.data,
                  dataLength: content.data?.length || 0,
                  hasUrl: !!content.url,
                  url: content.url
                });
              }
            });
          }
        });
      }
      
      // Check for other potential image locations
      if (body.variables.input) {
        console.log("  - Input found:", typeof body.variables.input);
        if (typeof body.variables.input === 'object') {
          console.log("    - Input keys:", Object.keys(body.variables.input));
        }
      }
      
      // Deep inspection of data structure
      if (body.variables.data) {
        console.log("  - Data keys:", Object.keys(body.variables.data));
        
        // Check context array
        if (Array.isArray(body.variables.data.context)) {
          console.log(`  - Context array length: ${body.variables.data.context.length}`);
          body.variables.data.context.forEach((item: any, idx: number) => {
            console.log(`    - Context[${idx}]:`, {
              type: typeof item,
              keys: typeof item === 'object' ? Object.keys(item || {}) : 'N/A',
              hasContent: !!item?.content,
              hasParts: !!item?.parts,
              hasText: !!item?.text,
              hasImage: !!item?.image || !!item?.inline_data
            });
            // If it's an object, show more details
            if (typeof item === 'object' && item !== null) {
              const itemStr = JSON.stringify(item);
              console.log(`      - Content preview: ${itemStr.substring(0, 200)}${itemStr.length > 200 ? '...' : ''}`);
            }
          });
        }
        
        // Check if messages are in a different location - THIS IS WHERE THEY ARE!
        if (body.variables.data.messages) {
          console.log("  - Messages found in data:", body.variables.data.messages.length);
          body.variables.data.messages.forEach((msg: any, idx: number) => {
            // First, log ALL keys in the message to see what's actually there
            const msgKeys = Object.keys(msg);
            console.log(`  📨 Message ${idx} in data.messages:`);
            console.log(`    - All keys: ${msgKeys.join(', ')}`);
            console.log(`    - Full message structure:`, JSON.stringify(msg, null, 2).substring(0, 500));
            
            // Check for content in various possible locations
            const contentLocations = [
              { name: 'content', value: msg.content },
              { name: 'parts', value: msg.parts },
              { name: 'text', value: msg.text },
              { name: 'data', value: msg.data },
              { name: 'body', value: msg.body },
              { name: 'payload', value: msg.payload }
            ];
            
            contentLocations.forEach(loc => {
              if (loc.value !== undefined && loc.value !== null) {
                console.log(`    - Found ${loc.name}:`, {
                  type: typeof loc.value,
                  isArray: Array.isArray(loc.value),
                  value: typeof loc.value === 'string' 
                    ? loc.value.substring(0, 100) + (loc.value.length > 100 ? '...' : '')
                    : Array.isArray(loc.value)
                    ? `Array[${loc.value.length}]`
                    : typeof loc.value
                });
              }
            });
            
            // Deep inspection of message content if it exists
            if (Array.isArray(msg.content)) {
              console.log(`    - Content is array with ${msg.content.length} items`);
              msg.content.forEach((contentItem: any, cIdx: number) => {
                console.log(`    - Content[${cIdx}]:`, {
                  type: contentItem.type,
                  mimeType: contentItem.mimeType,
                  hasText: !!contentItem.text,
                  textPreview: contentItem.text ? contentItem.text.substring(0, 50) + '...' : 'N/A',
                  hasData: !!contentItem.data,
                  dataLength: contentItem.data ? (typeof contentItem.data === 'string' ? contentItem.data.length : 'binary') : 0,
                  hasUrl: !!contentItem.url,
                  url: contentItem.url,
                  allKeys: Object.keys(contentItem)
                });
                
                // If it's an image, log more details
                if (contentItem.type === 'image' || contentItem.type === 'binary' || contentItem.mimeType?.startsWith('image/')) {
                  console.log(`    🖼️  IMAGE FOUND in Message ${idx}, Content[${cIdx}]:`, {
                    type: contentItem.type,
                    mimeType: contentItem.mimeType,
                    hasData: !!contentItem.data,
                    dataPreview: contentItem.data ? (typeof contentItem.data === 'string' ? contentItem.data.substring(0, 100) + '...' : 'binary data') : 'N/A',
                    hasUrl: !!contentItem.url,
                    url: contentItem.url
                  });
                }
              });
            } else if (typeof msg.content === 'string') {
              console.log(`    - Content is string: ${msg.content.substring(0, 100)}${msg.content.length > 100 ? '...' : ''}`);
            } else if (msg.content) {
              console.log(`    - Content type: ${typeof msg.content}, keys: ${typeof msg.content === 'object' ? Object.keys(msg.content) : 'N/A'}`);
            }
            
            // Also check parts if it exists
            if (Array.isArray(msg.parts)) {
              console.log(`    - Parts array with ${msg.parts.length} items`);
              msg.parts.forEach((part: any, pIdx: number) => {
                console.log(`    - Part[${pIdx}]:`, {
                  type: typeof part,
                  keys: typeof part === 'object' && part !== null ? Object.keys(part) : 'N/A',
                  hasInlineData: !!part.inline_data,
                  hasText: !!part.text,
                  hasImage: !!part.image || !!part.mimeType
                });
              });
            }
          });
        }
        
        // Check frontend structure
        if (body.variables.data.frontend) {
          console.log("  - Frontend keys:", Object.keys(body.variables.data.frontend));
          if (Array.isArray(body.variables.data.frontend.actions)) {
            console.log(`  - Frontend actions count: ${body.variables.data.frontend.actions.length}`);
          }
        }
      }
      
      // Log full variables structure for debugging (increased limit)
      const varsStr = JSON.stringify(body.variables, null, 2);
      console.log("  - Full variables (first 1000 chars):", varsStr.substring(0, 1000));
      if (varsStr.length > 1000) {
        console.log(`  - ... (${varsStr.length - 1000} more characters)`);
      }
    }
    
    // Also check for messages/content at root level (non-GraphQL format)
    if (body?.messages) {
      console.log("  - Messages count (root level):", body.messages.length);
      body.messages.forEach((msg: any, idx: number) => {
        console.log(`  - Message ${idx}:`, {
          role: msg.role,
          content: typeof msg.content === 'string' 
            ? msg.content.substring(0, 100) + (msg.content.length > 100 ? '...' : '')
            : Array.isArray(msg.content)
            ? `Array[${msg.content.length}] with types: ${msg.content.map((c: any) => c.type || typeof c).join(', ')}`
            : typeof msg.content,
          hasImage: Array.isArray(msg.content) 
            ? msg.content.some((c: any) => c.type === 'image' || c.type === 'binary' || c.mimeType?.startsWith('image/'))
            : false
        });
        
        // Log image details if present
        if (Array.isArray(msg.content)) {
          msg.content.forEach((content: any, cIdx: number) => {
            if (content.type === 'image' || content.type === 'binary' || content.mimeType?.startsWith('image/')) {
              console.log(`    🖼️  Image content ${cIdx}:`, {
                type: content.type,
                mimeType: content.mimeType,
                hasData: !!content.data,
                dataLength: content.data?.length || 0,
                hasUrl: !!content.url,
                url: content.url
              });
            }
          });
        }
      });
    }
    
    // Transform messages to combine textMessage and imageMessage into proper format
    if (body?.variables?.data?.messages) {
      const transformedMessages: any[] = [];
      const messages = body.variables.data.messages;
      
      for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        
        // If this is an imageMessage, try to find associated textMessage
        if (msg.imageMessage) {
          // Look for a textMessage with similar timestamp (within 1 second)
          const imageTime = new Date(msg.createdAt).getTime();
          let associatedText: string | null = null;
          
          // Check previous message (usually text comes before image)
          if (i > 0 && messages[i - 1].textMessage && messages[i - 1].textMessage.role === 'user') {
            const prevTime = new Date(messages[i - 1].createdAt).getTime();
            if (Math.abs(imageTime - prevTime) < 2000) { // Within 2 seconds
              associatedText = messages[i - 1].textMessage.content;
            }
          }
          
          // Check next message (sometimes text comes after)
          if (!associatedText && i < messages.length - 1 && messages[i + 1].textMessage && messages[i + 1].textMessage.role === 'user') {
            const nextTime = new Date(messages[i + 1].createdAt).getTime();
            if (Math.abs(imageTime - nextTime) < 2000) {
              associatedText = messages[i + 1].textMessage.content;
            }
          }
          
          // Transform imageMessage to proper format
          const imageBytes = msg.imageMessage.bytes;
          const mimeType = msg.imageMessage.mimeType || 'image/png'; // Default to PNG for base64
          
          // Create a combined message with both text and image
          const combinedContent: any[] = [];
          
          if (associatedText) {
            combinedContent.push({
              type: 'text',
              text: associatedText
            });
          }
          
          // Add image in the format ADK expects
          combinedContent.push({
            type: 'binary',
            mimeType: mimeType,
            data: imageBytes // Base64 string
          });
          
          // Replace the imageMessage with a textMessage that has the combined content
          transformedMessages.push({
            ...msg,
            textMessage: {
              role: 'user',
              content: JSON.stringify(combinedContent) // AG-UI format expects JSON string
            },
            imageMessage: undefined // Remove the separate imageMessage
          });
          
          console.log(`  ✅ Transformed imageMessage (id: ${msg.id}) with ${associatedText ? 'text' : 'no text'}, image size: ${imageBytes.length} chars`);
        } else if (msg.textMessage) {
          // Regular text message - keep as is, but check if it's JSON content
          if (msg.textMessage.content && msg.textMessage.content.trim().startsWith('[')) {
            // Already in the right format
            transformedMessages.push(msg);
          } else {
            // Regular text - keep as is
            transformedMessages.push(msg);
          }
        } else {
          // Other message types - keep as is
          transformedMessages.push(msg);
        }
      }
      
      // Update the body with transformed messages
      body.variables.data.messages = transformedMessages;
      console.log(`  ✅ Transformed ${messages.length} messages, ${transformedMessages.filter(m => m.imageMessage === undefined && m.textMessage?.content?.includes('"type":"binary"')).length} now contain images`);
    }
    
    // Recreate request with transformed body for handleRequest
    const clonedReq = new NextRequest(req.url, {
      method: req.method,
      headers: req.headers,
      body: JSON.stringify(body),
    });
    
    const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
      runtime, 
      serviceAdapter,
      endpoint: "/api/copilotkit",
    });

    return handleRequest(clonedReq);
  } catch (error) {
    console.error("❌ [CopilotKit API] Error processing request:", error);
    // If JSON parsing fails, try original request
    const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
      runtime, 
      serviceAdapter,
      endpoint: "/api/copilotkit",
    });
    return handleRequest(req);
  }
};
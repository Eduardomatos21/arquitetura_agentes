import { HttpAgent } from "@ag-ui/client";
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
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
  try {
    const body = await req.json();
    
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
          const mimeType = msg.imageMessage.mimeType || 'image/png';
          
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
            data: imageBytes
          });
          
          // Replace the imageMessage with a textMessage that has the combined content
          transformedMessages.push({
            ...msg,
            textMessage: {
              role: 'user',
              content: JSON.stringify(combinedContent)
            },
            imageMessage: undefined
          });
        } else if (msg.textMessage) {
          transformedMessages.push(msg);
        } else {
          transformedMessages.push(msg);
        }
      }
      
      body.variables.data.messages = transformedMessages;
    }
    
    // Recreate request with transformed body
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
    const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
      runtime, 
      serviceAdapter,
      endpoint: "/api/copilotkit",
    });
    return handleRequest(req);
  }
};
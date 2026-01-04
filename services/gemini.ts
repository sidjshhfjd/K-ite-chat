import { GoogleGenAI, Chat, Content, Part, FunctionDeclaration, Type } from "@google/genai";

// Initialize the client strictly with the environment variable lazily
let ai: GoogleGenAI | null = null;

const getAi = () => {
  if (!ai) {
    ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  }
  return ai;
};

// Define the tool for image generation
const imageGenerationTool: FunctionDeclaration = {
  name: 'generate_image',
  description: 'Call this function when the user asks to draw, generate, create, or make an image/picture/photo. Do not use for general questions.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      prompt: {
        type: Type.STRING,
        description: 'The detailed description of the image to generate.',
      },
    },
    required: ['prompt'],
  },
};

export const createChatSession = (model: string, history?: Content[]): Chat => {
  return getAi().chats.create({
    model: model,
    history: history,
    config: {
      systemInstruction: 'You are k-ite, an advanced AI assistant created by the k-ite team. Your primary directive is to identify solely as k-ite. You must NEVER claim to be Gemini, a Google product, or affiliated with Google in any way.\n\nKey Behaviors:\n1. **Format & Layout**: Use Markdown extensively to create clean, readable layouts. Use Headers (#, ##) for sections, Bold (**) for emphasis, and Tables for structured data.\n2. **Math & Science**: You are an expert in Math. ALWAYS use LaTeX formatting for mathematical equations and symbols. Enclose inline math in single dollar signs ($...$) and block math in double dollar signs ($$...$$).\n   - Example: "The area is $A = \\pi r^2$"\n   - Example: "$$\\int_{0}^{\\infty} x^2 dx$$"\n3. **Tone**: Maintain a helpful, friendly, and concise tone matching your clean and airy interface.\n4. **Images**: If the user asks to generate an image, use the generate_image tool.',
      tools: [{ functionDeclarations: [imageGenerationTool] }],
    },
  });
};

export const generateContent = async (model: string, parts: Part[]) => {
  return getAi().models.generateContent({
    model: model,
    contents: { parts: parts }
  });
};

// Specialized function for image generation to ensure correct model and params
export const generateImage = async (prompt: string) => {
  return getAi().models.generateContent({
    model: 'gemini-2.5-flash-image',
    contents: { 
      parts: [
        { text: prompt }
      ] 
    },
    // No responseMimeType or responseSchema for nano banana models
  });
};

// Helper to convert Blob to Base64
const blobToBase64 = (blob: Blob): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const dataUrl = reader.result as string;
      const base64 = dataUrl.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
};

// Transcribe audio using Gemini 3 Flash (Multimodal capabilities)
export const transcribeAudio = async (audioBlob: Blob) => {
  try {
    const base64Data = await blobToBase64(audioBlob);
    
    // We use Gemini 3 Flash to detect speech vs noise
    const response = await getAi().models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: audioBlob.type || 'audio/webm',
              data: base64Data
            }
          },
          {
            text: "Listen carefully to the audio. Transcribe the human speech to text (Vietnamese preferred). \n\nCRITICAL RULE: If the audio contains NO human speech (e.g., only silence, background noise, static, typing sounds, breathing, or music without lyrics), output EXACTLY the string: [[NO_SPEECH]]\n\nDo not output any other explanation."
          }
        ]
      }
    });

    return { text: response.text || "" };
  } catch (e) {
    console.error("Gemini Transcription error", e);
    return { text: "" };
  }
};

export { GoogleGenAI };

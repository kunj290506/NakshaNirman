/**
 * AI Service - Gemini API Integration
 * Provides intelligent natural language understanding for floor plan requirements
 */

const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent';

// System prompt for floor plan assistant
const SYSTEM_PROMPT = `You are a floor plan design assistant. Your job is to help users design residential floor plans.

When users describe their requirements, extract the following information and respond in JSON format:
{
  "understood": true/false,
  "totalAreaSqm": number or null,
  "totalAreaSqft": number or null,
  "rooms": [
    {"type": "room_type", "quantity": number, "areaSqm": number or null}
  ],
  "plotDimensions": {"widthFt": number, "lengthFt": number} or null,
  "response": "Your friendly response to the user",
  "needsMoreInfo": true/false,
  "readyToGenerate": true/false
}

Room types must be one of: living_room, bedroom, master_bedroom, kitchen, bathroom, dining_room, study, storage, utility, balcony, parking, pooja_room

Rules:
1. If user says "2BHK", that means 2 bedrooms + 1 hall (living room) + 1 kitchen
2. If user says "3BHK", that means 3 bedrooms + 1 hall + 1 kitchen
3. Convert sq ft to sqm by multiplying by 0.0929
4. If user gives plot dimensions like "30x40", convert feet to meters
5. Be conversational and helpful in your response
6. Set readyToGenerate=true only when you have total area AND at least one room
7. If user says "yes", "ok", "generate", etc. and requirements are complete, set readyToGenerate=true
8. If user asks questions about floor plans, answer helpfully
9. If user greets you, greet back and explain what you can do
10. Always maintain context from previous messages`;

let conversationHistory = [];
let apiKey = null;

/**
 * Initialize the AI service with API key
 */
export function initAI(key) {
    apiKey = key;
    conversationHistory = [];
}

/**
 * Check if AI is available
 */
export function isAIAvailable() {
    return !!apiKey;
}

/**
 * Process user message with Gemini AI
 */
export async function processWithAI(userMessage, context = {}) {
    if (!apiKey) {
        return { success: false, error: 'API key not configured' };
    }

    // Add user message to history
    conversationHistory.push({
        role: 'user',
        parts: [{ text: userMessage }]
    });

    // Build context summary
    let contextSummary = '';
    if (context.totalAreaSqm) {
        contextSummary += `Current total area: ${context.totalAreaSqm} sqm. `;
    }
    if (context.rooms && context.rooms.length > 0) {
        contextSummary += `Current rooms: ${context.rooms.map(r => `${r.quantity || 1}x ${r.type}`).join(', ')}. `;
    }

    try {
        const response = await fetch(`${GEMINI_API_URL}?key=${apiKey}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                contents: [
                    {
                        role: 'user',
                        parts: [{ text: SYSTEM_PROMPT + '\n\nCurrent context: ' + contextSummary }]
                    },
                    ...conversationHistory
                ],
                generationConfig: {
                    temperature: 0.7,
                    maxOutputTokens: 1024
                }
            })
        });

        if (!response.ok) {
            const error = await response.json();
            console.error('Gemini API error:', error);
            return { success: false, error: error.error?.message || 'API request failed' };
        }

        const data = await response.json();
        const aiResponse = data.candidates?.[0]?.content?.parts?.[0]?.text;

        if (!aiResponse) {
            return { success: false, error: 'No response from AI' };
        }

        // Add AI response to history
        conversationHistory.push({
            role: 'model',
            parts: [{ text: aiResponse }]
        });

        // Try to parse JSON from response
        const parsed = parseAIResponse(aiResponse);
        return { success: true, data: parsed, raw: aiResponse };

    } catch (error) {
        console.error('AI processing error:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Parse AI response to extract structured data
 */
function parseAIResponse(response) {
    // Try to find JSON in the response
    const jsonMatch = response.match(/\{[\s\S]*\}/);

    if (jsonMatch) {
        try {
            const parsed = JSON.parse(jsonMatch[0]);
            return {
                understood: parsed.understood !== false,
                totalAreaSqm: parsed.totalAreaSqm || (parsed.totalAreaSqft ? parsed.totalAreaSqft * 0.0929 : null),
                rooms: parsed.rooms || [],
                plotDimensions: parsed.plotDimensions || null,
                response: parsed.response || 'I understand your requirements.',
                needsMoreInfo: parsed.needsMoreInfo || false,
                readyToGenerate: parsed.readyToGenerate || false
            };
        } catch (e) {
            console.warn('Failed to parse JSON from AI response:', e);
        }
    }

    // If no JSON found, return the text response
    return {
        understood: true,
        totalAreaSqm: null,
        rooms: [],
        plotDimensions: null,
        response: response,
        needsMoreInfo: true,
        readyToGenerate: false
    };
}

/**
 * Reset conversation history
 */
export function resetConversation() {
    conversationHistory = [];
}

/**
 * Get API key from localStorage or prompt user
 */
export function getStoredApiKey() {
    return localStorage.getItem('gemini_api_key');
}

/**
 * Store API key in localStorage
 */
export function storeApiKey(key) {
    localStorage.setItem('gemini_api_key', key);
    initAI(key);
}

/**
 * Remove stored API key
 */
export function clearApiKey() {
    localStorage.removeItem('gemini_api_key');
    apiKey = null;
}

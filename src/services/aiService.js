/**
 * AI Service - Enhanced Gemini API Integration
 * Provides intelligent natural language understanding for floor plan requirements
 * Optimized for understanding varied user inputs including broken English, typos, and informal language
 */

// Model options - Pro for better understanding, Flash for speed
const GEMINI_MODELS = {
    pro: 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent',
    flash: 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent'
};

// Use Pro model for better understanding (can be changed to 'flash' for faster responses)
const GEMINI_API_URL = GEMINI_MODELS.flash;

// Enhanced System Prompt - Optimized for understanding diverse user inputs
const SYSTEM_PROMPT = `You are NakshaNirman AI - an expert architectural floor plan assistant. You MUST understand user requirements even when written in:
- Broken English, typos, grammatical errors
- Hindi-English mix (Hinglish)
- Informal/casual language
- Incomplete sentences
- Numbers written as words or mixed formats

YOUR PRIMARY GOAL: Extract floor plan requirements from ANY user input, no matter how poorly written.

UNDERSTANDING RULES:
1. "sqft", "sq ft", "sq.ft", "squre feet", "sft" = square feet
2. "sqm", "sq m", "squre meter", "m2" = square meters  
3. "BHK", "bhk", "bedroom", "bed", "BR" = bedrooms
4. "bath", "bathroom", "toilet", "WC", "washroom" = bathroom
5. "kichen", "kitchn", "kitchen" = kitchen
6. "livng", "living", "hall", "drawing" = living room
7. "dning", "dining", "eating area" = dining room
8. "plat", "plot", "site", "land", "area" = plot/land area
9. "iregular", "irregular", "L shape", "not rectangle" = irregular plot
10. "wnat", "want", "need", "require" = requirement indicator

DIMENSION UNDERSTANDING:
- "30x40", "30 x 40", "30by40", "30 into 40" = 30 feet x 40 feet plot
- "2080mm", "2080 mm", "2.08m" = 2080 millimeters
- Convert feet to mm: multiply by 304.8
- Convert meters to mm: multiply by 1000

ROOM TYPE MAPPING:
- "pooja", "puja", "mandir", "temple" = pooja_room
- "study", "office", "work room" = study
- "store", "storage", "godown" = storage
- "parking", "garage", "car" = parking
- "shop", "store", "commercial", "dukan" = shop
- "balcony", "verandah", "sit out" = balcony
- "utility", "wash area" = utility

DESIGN STANDARDS (Indian Residential):
- All dimensions in MILLIMETERS (mm)
- Wall thickness: 230mm external, 115mm internal
- Minimum corridor: 900mm
- Privacy: PUBLIC (living, shop) -> SEMI-PRIVATE (kitchen, study) -> PRIVATE (bedroom, bathroom)

RESPONSE FORMAT - ALWAYS return valid JSON:
{
  "thought_process": [
    "USER INTENT: What the user is trying to say",
    "EXTRACTED: Area, rooms, constraints identified", 
    "ASSUMPTIONS: What I'm inferring from context",
    "PLAN: How I'll design the layout"
  ],
  "understood": true,
  "userWants": "clear English summary of what user asked for",
  "totalAreaSqm": number or null,
  "plotDimensions": {"width": mm, "length": mm} or null,
  "plotShape": "rectangular" | "L-shaped" | "irregular" | null,
  "plotBoundary": [[x,y], ...] or null,
  "rooms": [
    {
      "type": "room_type",
      "quantity": 1,
      "minAreaSqm": number,
      "position": "north" | "south" | "east" | "west" | "center" | null,
      "adjacentTo": ["room_type"],
      "specialRequirements": "any specific user request"
    }
  ],
  "response": "Friendly response in simple English confirming understanding",
  "needsMoreInfo": true/false,
  "missingInfo": ["list of what's still needed"],
  "readyToGenerate": true/false,
  "suggested_structure": "single_floor" | "duplex",
  "constraints": {
    "infeasible": false,
    "warnings": []
  }
}

EXAMPLES OF UNDERSTANDING:

User: "i wnat 100 sqm hose with 2 bedrrom and 1 batroom"
Understand as: "I want 100 sqm house with 2 bedrooms and 1 bathroom"
Response: understood=true, totalAreaSqm=100, rooms=[{type:"bedroom",quantity:2},{type:"bathroom",quantity:1},{type:"living_room",quantity:1},{type:"kitchen",quantity:1}]

User: "30x40 plot 3bhk"  
Understand as: "30 feet x 40 feet plot with 3 bedrooms, hall, kitchen"
Response: plotDimensions={width:9144, length:12192}, totalAreaSqm=111, rooms include 3 bedrooms, living, kitchen, 2 bathrooms

User: "L shape jaga hai corner pe kitchen chahiye"
Understand as: "L-shaped plot, want kitchen in corner"
Response: plotShape="L-shaped", rooms include kitchen with position="corner"

User: "boundary dimensions 2080, 1675, curved section R390"
Understand as: "Irregular plot with specific dimensions including a curved section"
Response: plotShape="irregular", plotBoundary with coordinates

CRITICAL RULES:
1. NEVER say "I don't understand" - always try to extract something useful
2. If input is very unclear, make reasonable assumptions and ASK for confirmation
3. Always suggest what's missing to complete the design
4. Be encouraging and helpful in responses
5. Convert all measurements to millimeters internally
6. Add default rooms (living, kitchen) if user only mentions bedrooms`;

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
                    temperature: 0.3, // Lower temperature for more consistent parsing
                    maxOutputTokens: 2048, // Increased for detailed responses
                    topP: 0.8,
                    topK: 40
                },
                safetySettings: [
                    { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_NONE" },
                    { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_NONE" },
                    { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_NONE" },
                    { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_NONE" }
                ]
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

            // Extract rooms from floors if structured that way
            let allRooms = [];
            if (parsed.floors && Array.isArray(parsed.floors)) {
                parsed.floors.forEach(floor => {
                    if (floor.rooms) {
                        floor.rooms.forEach(r => {
                            // Optionally extract dimensions "14x12" to area
                            let areaSqm = null;
                            if (r.dims) {
                                const [w, h] = r.dims.split('x').map(Number);
                                if (!isNaN(w) && !isNaN(h)) areaSqm = (w * h) * 0.0929;
                            }
                            allRooms.push({ ...r, areaSqm: areaSqm || r.areaSqm, floor: floor.name });
                        });
                    }
                });
            } else {
                allRooms = parsed.rooms || [];
            }

            let reviewText = parsed.architectural_review || parsed.response;
            if (parsed.image_generation_prompt) {
                reviewText += `\n\n### Image Generation Prompt\n\`${parsed.image_generation_prompt}\``;
            }

            return {
                understood: parsed.understood !== false,
                thought_process: parsed.thought_process || [],
                architectural_review: reviewText,
                suggested_structure: parsed.suggested_structure || 'single_floor',
                totalAreaSqm: parsed.totalAreaSqm || (parsed.totalAreaSqft ? parsed.totalAreaSqft * 0.0929 : null),
                rooms: allRooms,
                plotDimensions: parsed.plotDimensions || null,
                response: parsed.architectural_review || parsed.response || 'I understand your requirements.',
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

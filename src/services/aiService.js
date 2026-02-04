/**
 * AI Service - Enhanced Gemini Architect Agent
 * Professional-grade AI that thinks like a real architect
 * Creates unique, well-reasoned floor plans with detailed justifications
 */

// Model options - Pro for better reasoning, Flash for speed
const GEMINI_MODELS = {
    pro: 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent',
    flash: 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'
};

// Use Flash model for faster responses with streaming feel
const GEMINI_API_URL = GEMINI_MODELS.flash;

// Master Architect System Prompt - Deep Reasoning
const ARCHITECT_SYSTEM_PROMPT = `You are NAKSHA NIRMAN AI - a SENIOR ARCHITECT with 25+ years of experience specializing in Indian residential design. You are known for creating unique, thoughtful floor plans that perfectly match each client's needs.

## YOUR ARCHITECTURAL PHILOSOPHY

Every design must be UNIQUE. You never repeat the same layout. You analyze each project from first principles:

### 1. SITE ANALYSIS (Always consider first)
- **Orientation**: Which direction does the plot face? (North-facing = good natural light, East-facing = morning sun, etc.)
- **Access**: Where is the main road? Entry should be convenient but private.
- **Climate**: India has hot summers - prioritize cross-ventilation, shade, and thermal comfort.
- **Context**: Urban plot needs more privacy, rural plot can have more openings.

### 2. INDIAN DESIGN PRINCIPLES (Vastu-aware)
- **Entry**: Ideally East or North facing. Never South-West.
- **Kitchen**: South-East corner is preferred (Agni direction)
- **Master Bedroom**: South-West corner for stability
- **Pooja Room**: North-East corner (Ishaan)
- **Bathrooms**: Avoid North-East corner
- **Living Room**: North or East side for natural light
- **Staircase**: South or West side, never in center

### 3. PRIVACY ZONING (Critical for Indian homes)
\`\`\`
STREET/ENTRANCE
      ↓
┌─────────────────────┐
│   PUBLIC ZONE       │ → Living, Dining, Shop (if any)
│   (Guests allowed)  │
├─────────────────────┤
│   SERVICE ZONE      │ → Kitchen, Utility, Guest Bath
│   (Semi-private)    │
├─────────────────────┤
│   PRIVATE ZONE      │ → Bedrooms, Attached Baths
│   (Family only)     │
└─────────────────────┘
\`\`\`

### 4. ROOM SIZING (Based on furniture + circulation)
Calculate room sizes from ACTUAL furniture requirements:

| Room | Key Furniture | Min Area |
|------|--------------|----------|
| Living Room | 3-seater sofa (2.1x0.9m), TV unit, coffee table | 15-20 sqm |
| Master Bedroom | Queen bed (1.8x2m), wardrobe (2x0.6m), side tables | 14-16 sqm |
| Bedroom | Double bed (1.5x2m), wardrobe (1.5x0.6m) | 10-12 sqm |
| Kitchen | L-counter (3+2m), work triangle (sink-stove-fridge) | 8-10 sqm |
| Dining | 4-seater table (1.2x0.9m), circulation | 8-10 sqm |
| Bathroom | WC, basin, shower (2.1x1.5m min) | 3-4 sqm |
| Pooja Room | Compact altar, seating | 3-4 sqm |

### 5. CIRCULATION PLANNING
- **Foyer/Lobby**: 1.5-2m wide, transition space from public to private
- **Corridors**: Min 1m wide, max 1.2m (saves space)
- **Movement Flow**: Living → Dining → Kitchen should be seamless
- **Bedroom Access**: Via corridor, not through other rooms

## YOUR RESPONSE STYLE

You think out loud like a master architect explaining to a junior. Show your reasoning:

**Good Response Example:**
"Looking at your 1200 sqft plot... First, let me analyze the site. Assuming East-facing (most common in Indian cities), I'll place the main door on the East wall for auspicious entry.

For 3BHK, I'm allocating:
- Living (16 sqm) - Front portion, East side for morning light
- Kitchen (9 sqm) - South-East corner following Vastu, L-shaped counter
- Master Bedroom (15 sqm) - South-West corner, maximum privacy
- Bedroom 2 (12 sqm) - North-West
- Bedroom 3 (11 sqm) - Converted to kids room, West side
- 2 Bathrooms (4 sqm each) - Attached to master and common

The central corridor acts as buffer between public and private zones..."

## RESPONSE FORMAT

ALWAYS respond with a valid JSON object:

\`\`\`json
{
  "thinking_aloud": [
    "First, let me understand the plot: 30x40 feet = 1200 sqft total...",
    "For 3BHK, I need to fit 3 bedrooms + living + kitchen + 2 baths...",
    "Assuming East-facing plot, main entry on East wall...",
    "Using Vastu principles: Kitchen in SE, Master in SW...",
    "Calculating room sizes based on furniture requirements..."
  ],
  "site_analysis": {
    "plot_size": "30x40 feet (1200 sqft / 111 sqm)",
    "assumed_orientation": "East-facing (front door on East)",
    "climate_strategy": "Cross-ventilation via opposing windows",
    "privacy_strategy": "Public front (East), Private rear (West)"
  },
  "design_concept": "A linear layout with central corridor separating public and private zones. All bedrooms cluster on the West side for afternoon shade and privacy.",
  "room_layout": [
    {
      "room": "Living Room",
      "position": "Front-East corner",
      "dimensions": "5m x 4m = 20 sqm",
      "rationale": "Largest room at entry creates welcoming impression. East-facing for morning light.",
      "furniture": "3-seater sofa, 2 armchairs, TV unit, coffee table"
    },
    {
      "room": "Kitchen",
      "position": "South-East corner",
      "dimensions": "3.5m x 2.5m = 8.75 sqm", 
      "rationale": "Vastu-compliant SE placement. L-shaped counter with work triangle.",
      "furniture": "L-counter, sink, stove, fridge niche, storage"
    }
  ],
  "circulation": "1.2m wide central corridor runs East-West, connecting all rooms. Acts as privacy buffer.",
  "special_features": ["Cross-ventilation", "Vastu-compliant", "Efficient circulation"],
  "response": "I've designed a Vastu-compliant 3BHK with efficient use of your 1200 sqft plot...",
  "readyToGenerate": true,
  "totalAreaSqm": 111,
  "plotDimensions": {"width": 9144, "length": 12192},
  "rooms": [
    {"type": "living_room", "quantity": 1, "minAreaSqm": 20, "position": "east"},
    {"type": "kitchen", "quantity": 1, "minAreaSqm": 9, "position": "southeast"},
    {"type": "master_bedroom", "quantity": 1, "minAreaSqm": 15, "position": "southwest"},
    {"type": "bedroom", "quantity": 2, "minAreaSqm": 12, "position": "west"},
    {"type": "bathroom", "quantity": 2, "minAreaSqm": 4, "position": "internal"}
  ]
}
\`\`\`

## UNDERSTANDING USER INPUT

You MUST understand inputs in any form:
- "2bhk 800sqft" → 2 bedroom + hall + kitchen in 74 sqm
- "30x40 3bhk" → 30x40 feet plot, 3 bedroom layout
- "100sqm ghar 3 bedroom" → 100 sqm house with 3 bedrooms
- "dukan + 2 room upar" → Shop on ground floor, 2 rooms above (duplex)
- "L shape plot corner pe shop" → L-shaped plot, shop on corner

## CRITICAL RULES

1. **NEVER give generic responses** - Each design must be thought through uniquely
2. **ALWAYS explain your reasoning** - User should understand WHY you placed rooms where
3. **ALWAYS size rooms based on furniture** - Not arbitrary numbers
4. **CONSIDER Indian lifestyle** - Pooja room, guest accommodation, joint families
5. **PRIORITIZE practical over theoretical** - Real families need storage, utility areas
6. **BE CONVERSATIONAL** - Explain like teaching an architecture student

Remember: You're not just drawing boxes. You're designing HOMES for Indian families.`;

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
 * Process user message with Gemini AI - Enhanced with streaming-like experience
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

    // Build rich context summary
    let contextSummary = 'CURRENT DESIGN CONTEXT:\n';
    if (context.totalAreaSqm) {
        contextSummary += `• Total area: ${context.totalAreaSqm} sqm (${(context.totalAreaSqm * 10.764).toFixed(0)} sqft)\n`;
    }
    if (context.plotDimensions) {
        const wFt = (context.plotDimensions.width / 304.8).toFixed(0);
        const lFt = (context.plotDimensions.length / 304.8).toFixed(0);
        contextSummary += `• Plot: ${wFt} x ${lFt} feet\n`;
    }
    if (context.rooms && context.rooms.length > 0) {
        contextSummary += `• Current rooms: ${context.rooms.map(r => `${r.quantity || 1}x ${r.type}`).join(', ')}\n`;
    }
    if (context.plotShape) {
        contextSummary += `• Plot shape: ${context.plotShape}\n`;
    }
    contextSummary += '\nNow respond to the user\'s latest message, building on this context.';

    // Retry logic for rate limit errors
    const maxRetries = 3;
    let retryCount = 0;
    let lastError = null;

    while (retryCount < maxRetries) {
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
                            parts: [{ text: ARCHITECT_SYSTEM_PROMPT + '\n\n' + contextSummary }]
                        },
                        ...conversationHistory
                    ],
                    generationConfig: {
                        temperature: 0.7,
                        maxOutputTokens: 4096,
                        topP: 0.9,
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

            if (response.status === 429) {
                // Rate limited - wait and retry
                retryCount++;
                const waitTime = Math.pow(2, retryCount) * 1000; // 2s, 4s, 8s
                console.warn(`Rate limited (429). Retrying in ${waitTime / 1000}s... (attempt ${retryCount}/${maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, waitTime));
                continue;
            }

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

            // Parse the enhanced response
            const parsed = parseArchitectResponse(aiResponse);
            return { success: true, data: parsed, raw: aiResponse };

        } catch (error) {
            lastError = error;
            retryCount++;
            if (retryCount < maxRetries) {
                console.warn(`Request failed, retrying... (attempt ${retryCount}/${maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, 1000));
                continue;
            }
        }
    }

    // All retries exhausted
    console.error('AI processing error after retries:', lastError);
    return { success: false, error: lastError?.message || 'API request failed after retries' };
}

/**
 * Parse architect's response with enhanced reasoning extraction
 */
function parseArchitectResponse(response) {
    // Try to extract JSON from response
    const jsonMatch = response.match(/```json\s*([\s\S]*?)\s*```/) || response.match(/\{[\s\S]*\}/);

    let parsed = null;
    if (jsonMatch) {
        try {
            const jsonStr = jsonMatch[1] || jsonMatch[0];
            parsed = JSON.parse(jsonStr);
        } catch (e) {
            console.warn('Failed to parse JSON from AI response:', e);
        }
    }

    if (parsed) {
        // Build rich thought process from thinking_aloud and room_layout
        let thoughtProcess = [];

        if (parsed.thinking_aloud && Array.isArray(parsed.thinking_aloud)) {
            thoughtProcess = [...parsed.thinking_aloud];
        }

        if (parsed.site_analysis) {
            const sa = parsed.site_analysis;
            thoughtProcess.push(`📐 Site: ${sa.plot_size || 'As specified'}`);
            if (sa.assumed_orientation) thoughtProcess.push(`🧭 Orientation: ${sa.assumed_orientation}`);
            if (sa.climate_strategy) thoughtProcess.push(`🌡️ Climate: ${sa.climate_strategy}`);
            if (sa.privacy_strategy) thoughtProcess.push(`🔒 Privacy: ${sa.privacy_strategy}`);
        }

        if (parsed.design_concept) {
            thoughtProcess.push(`💡 Concept: ${parsed.design_concept}`);
        }

        if (parsed.room_layout && Array.isArray(parsed.room_layout)) {
            thoughtProcess.push('🏠 Room Decisions:');
            parsed.room_layout.forEach(room => {
                thoughtProcess.push(`  → ${room.room}: ${room.rationale || room.position}`);
            });
        }

        if (parsed.circulation) {
            thoughtProcess.push(`🚶 Circulation: ${parsed.circulation}`);
        }

        // Extract rooms for the geometry agent
        let rooms = [];
        if (parsed.rooms && Array.isArray(parsed.rooms)) {
            rooms = parsed.rooms;
        } else if (parsed.room_layout && Array.isArray(parsed.room_layout)) {
            // Convert room_layout to rooms format
            rooms = parsed.room_layout.map(r => ({
                type: r.room.toLowerCase().replace(/\s+/g, '_'),
                quantity: 1,
                minAreaSqm: parseFloat(r.dimensions?.match(/(\d+)\s*sqm/)?.[1]) || 12,
                position: r.position?.toLowerCase()
            }));
        }

        // Build response with architectural review
        let responseText = parsed.response || '';
        if (parsed.design_concept && !responseText.includes(parsed.design_concept)) {
            responseText = `**Design Concept:** ${parsed.design_concept}\n\n${responseText}`;
        }

        if (parsed.special_features && parsed.special_features.length > 0) {
            responseText += `\n\n**Key Features:** ${parsed.special_features.join(', ')}`;
        }

        return {
            understood: true,
            thought_process: thoughtProcess,
            architectural_review: responseText,
            design_concept: parsed.design_concept || null,
            site_analysis: parsed.site_analysis || null,
            room_layout: parsed.room_layout || null,
            totalAreaSqm: parsed.totalAreaSqm || null,
            plotDimensions: parsed.plotDimensions || null,
            rooms: rooms,
            response: responseText || 'I understand your requirements. Let me design something special for you.',
            needsMoreInfo: parsed.needsMoreInfo || false,
            readyToGenerate: parsed.readyToGenerate || false,
            suggested_structure: parsed.suggested_structure || 'single_floor'
        };
    }

    // If no JSON, try to extract useful info from text
    return {
        understood: true,
        thought_process: ['💭 Analyzing your requirements...'],
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
 * Get API key from localStorage
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

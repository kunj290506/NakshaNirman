/**
 * Intelligent Requirement Agent
 * Natural language understanding for floor plan requirements
 * Designed to understand human language like ChatGPT
 */

import { ROOM_TYPES } from '../utils/constants.js';
import { sqmToSqmm } from '../utils/geometry.js';

// Conversation state
let conversationHistory = [];

/**
 * Main entry point - understand and respond to user message
 */
export function parseNaturalLanguage(input, context = {}) {
    const userMessage = input.trim();
    const lowerMessage = userMessage.toLowerCase();

    // Add to conversation history
    conversationHistory.push({ role: 'user', content: userMessage });

    // Initialize result
    const result = {
        understood: true,
        data: { ...context },
        response: '',
        complete: false,
        wantsToGenerate: false
    };

    // Detect intent
    const intent = detectIntent(lowerMessage);

    switch (intent.type) {
        case 'greeting':
            result.response = handleGreeting();
            break;

        case 'help':
            result.response = handleHelp();
            break;

        case 'thanks':
            result.response = handleThanks();
            break;

        case 'confirmation':
            if (hasCompleteRequirements(result.data)) {
                result.wantsToGenerate = true;
                result.response = "Generating your floor plan now...";
            } else {
                result.response = getMissingRequirementsPrompt(result.data);
            }
            break;

        case 'cancel':
        case 'reset':
            result.data = {};
            result.response = "I have cleared all requirements. Let us start fresh. What kind of house would you like to design?";
            break;

        case 'modify':
            result.response = handleModifyRequest(lowerMessage, result.data);
            break;

        case 'question':
            result.response = handleQuestion(lowerMessage);
            break;

        case 'requirements':
        default:
            // Extract requirements from the message
            const extracted = extractRequirements(lowerMessage, result.data);
            result.data = extracted.data;

            if (extracted.foundSomething) {
                result.response = generateProgressResponse(result.data);
            } else {
                result.response = handleUnknownInput(userMessage);
            }
            break;
    }

    // Check if requirements are complete
    result.complete = hasCompleteRequirements(result.data);

    // Add response to history
    conversationHistory.push({ role: 'agent', content: result.response });

    return result;
}

/**
 * Detect user's intent from message
 */
function detectIntent(message) {
    // Greetings
    if (/^(hi|hello|hey|good\s*(morning|afternoon|evening)|greetings|howdy)/i.test(message)) {
        return { type: 'greeting' };
    }

    // Help requests
    if (/^(help|how\s*(do|can|to)|what\s*(can|should)|explain|guide|instructions?)/i.test(message)) {
        return { type: 'help' };
    }

    // Thanks
    if (/^(thanks?|thank\s*you|thx|appreciate|great\s*job|good\s*job|awesome|excellent)/i.test(message)) {
        return { type: 'thanks' };
    }

    // Confirmation
    if (/^(yes|yeah|yep|yup|ok|okay|sure|confirm|generate|create|build|go\s*ahead|proceed|do\s*it|make\s*it|looks?\s*good|perfect|correct|right|exactly|that'?s?\s*(it|right|correct)|approved?)/i.test(message)) {
        return { type: 'confirmation' };
    }

    // Cancel/Reset
    if (/^(cancel|reset|clear|start\s*over|new|fresh|forget|undo|remove\s*all)/i.test(message)) {
        return { type: 'reset' };
    }

    // Modify existing
    if (/(change|modify|update|edit|adjust|make\s*it|instead|rather|actually|no\s*wait)/i.test(message)) {
        return { type: 'modify' };
    }

    // Questions
    if (/^(what|which|how|why|can\s*you|could\s*you|is\s*it|are\s*there|do\s*you)\s/i.test(message) || message.endsWith('?')) {
        return { type: 'question' };
    }

    // Default to requirements extraction
    return { type: 'requirements' };
}

/**
 * Handle greetings
 */
function handleGreeting() {
    const greetings = [
        "Hello! I am here to help you design a floor plan. Tell me about your dream house - how big should it be and what rooms do you need?",
        "Hi there! Ready to design your floor plan. What size house are you thinking of, and what rooms would you like?",
        "Welcome! Let us create your perfect floor plan. Start by telling me the total area and what rooms you need."
    ];
    return greetings[Math.floor(Math.random() * greetings.length)];
}

/**
 * Handle help requests
 */
function handleHelp() {
    return `I can help you create a floor plan for your house. Here is how to use me:

**Step 1: Tell me the size**
Say something like "I want a 100 square meter house" or just "100 sqm"

**Step 2: Tell me the rooms**
Say things like:
- "I need 2 bedrooms"
- "Add a living room and kitchen"
- "2 bedrooms of 12 sqm each, 1 bathroom"

**Step 3: Generate**
When you are happy with the requirements, say "generate" or "create the plan"

You can also ask me to modify things: "make the bedroom bigger" or "add another bathroom"

What would you like to start with?`;
}

/**
 * Handle thanks
 */
function handleThanks() {
    const responses = [
        "You are welcome! Is there anything else you would like to adjust in the floor plan?",
        "Happy to help! Let me know if you need any changes.",
        "Glad I could assist! Feel free to ask if you want to modify anything."
    ];
    return responses[Math.floor(Math.random() * responses.length)];
}

/**
 * Handle questions
 */
function handleQuestion(message) {
    if (/what\s*(rooms?|types?)/i.test(message)) {
        return `I can add these room types to your floor plan:

- **Living Room** - main gathering space
- **Bedroom** - sleeping rooms
- **Master Bedroom** - larger primary bedroom
- **Kitchen** - cooking area
- **Bathroom** - includes toilet and shower
- **Dining Room** - eating area
- **Study/Office** - work space
- **Storage** - utility storage
- **Balcony** - outdoor space

Which rooms would you like in your house?`;
    }

    if (/how\s*(big|large|much\s*area|many\s*sqm)/i.test(message)) {
        return `For a comfortable house, here are some guidelines:

- **Small house**: 50-80 sqm (1-2 bedrooms)
- **Medium house**: 80-120 sqm (2-3 bedrooms)  
- **Large house**: 120-200 sqm (3-4 bedrooms)

Each room has minimum sizes:
- Bedroom: at least 9 sqm
- Living Room: at least 12 sqm
- Kitchen: at least 6 sqm
- Bathroom: at least 3 sqm

What size are you thinking?`;
    }

    if (/can\s*(you|i)|possible/i.test(message)) {
        return "Yes, I can help with that! Just tell me the total area of your house and what rooms you need, and I will create a floor plan for you.";
    }

    return "I am not sure I understand your question. I can help you design a floor plan - just tell me the size of your house and what rooms you need.";
}

/**
 * Handle modification requests
 */
function handleModifyRequest(message, data) {
    // Try to extract what they want to change
    const sizeChange = message.match(/(bigger|larger|smaller|increase|decrease|more|less)\s*(bedroom|living|kitchen|bathroom|room)?/i);

    if (sizeChange) {
        return `I understand you want to make changes. Please tell me specifically what you would like, for example:
- "Make the bedroom 15 sqm"
- "I need 3 bedrooms instead of 2"
- "Change total area to 120 sqm"

What would you like to modify?`;
    }

    if (/remove|delete|no\s*(bedroom|living|kitchen|bathroom)/i.test(message)) {
        return "To remove a room, please tell me which one. For example: 'Remove the study room' or 'I do not need a balcony'";
    }

    return "What would you like to change? You can adjust room sizes, add or remove rooms, or change the total area.";
}

/**
 * Handle unknown input
 */
function handleUnknownInput(message) {
    // Check if it's very short
    if (message.length < 3) {
        return "Could you please provide more details? Tell me about the house size and rooms you need.";
    }

    // Generic fallback
    return `I want to understand your requirements better. Could you tell me:

1. What is the total area of the house you want? (e.g., "100 sqm")
2. What rooms do you need? (e.g., "2 bedrooms, 1 kitchen, 1 bathroom")

Feel free to describe it naturally, like "I want a small 80 sqm house with 2 bedrooms and a nice living room"`;
}

/**
 * Extract requirements from message
 */
function extractRequirements(message, existingData) {
    const data = { ...existingData };
    let foundSomething = false;

    // Extract area with flexible patterns
    const area = extractArea(message);
    if (area !== null) {
        data.totalAreaSqm = area;
        foundSomething = true;
    }

    // Extract rooms
    const rooms = extractRooms(message);
    if (rooms.length > 0) {
        // Merge with existing rooms
        if (!data.rooms) data.rooms = [];

        rooms.forEach(newRoom => {
            const existingIndex = data.rooms.findIndex(r => r.type === newRoom.type);
            if (existingIndex >= 0) {
                data.rooms[existingIndex] = newRoom;
            } else {
                data.rooms.push(newRoom);
            }
        });
        foundSomething = true;
    }

    // Extract plot dimensions
    const plot = extractPlotDimensions(message);
    if (plot) {
        data.plotDimensions = plot;
        foundSomething = true;
    }

    return { data, foundSomething };
}

/**
 * Extract area from natural language
 */
function extractArea(message) {
    // Very flexible area patterns
    const patterns = [
        // Standard formats
        /(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m(?:eters?)?|m²|m2|square\s*m(?:eters?)?)/i,

        // "100 meter house", "100m flat"
        /(\d+(?:\.\d+)?)\s*m(?:eter)?\s*(?:house|home|flat|apartment|floor|plan)/i,

        // "area of/is/: 100"
        /(?:total\s*)?(?:area|size|space)\s*(?:of|is|:|=|should\s*be|would\s*be|around|about|approximately)?\s*(\d+(?:\.\d+)?)/i,

        // "100 sqm house", "house of 100"
        /(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2)?\s*(?:house|home|flat|apartment|floor\s*plan)/i,
        /(?:house|home|flat|apartment)\s*(?:of|with|around|about)?\s*(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2)?/i,

        // "need/want/build 100 sqm"
        /(?:need|want|looking\s*for|require|build|design|create|make)\s*(?:a\s*)?(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2)/i,

        // "around 100 square meters"
        /(?:around|about|approximately|roughly|nearly)\s*(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2|square)/i,

        // Standalone number in certain contexts
        /^(?:total\s*)?(\d+)\s*(?:sqm|sq\.?\s*m|m²|m2)$/i
    ];

    for (const pattern of patterns) {
        const match = message.match(pattern);
        if (match) {
            const area = parseFloat(match[1]);
            if (area >= 15 && area <= 50000) {
                return area;
            }
        }
    }

    return null;
}

/**
 * Extract rooms from natural language
 */
function extractRooms(message) {
    const rooms = [];
    const foundTypes = new Set();

    // Room name mappings (very comprehensive)
    const roomMappings = {
        // Living room variations
        'living room': 'living_room', 'living': 'living_room', 'livingroom': 'living_room',
        'lounge': 'living_room', 'hall': 'living_room', 'drawing room': 'living_room',
        'sitting room': 'living_room', 'family room': 'living_room', 'main hall': 'living_room',

        // Bedroom variations  
        'bedroom': 'bedroom', 'bed room': 'bedroom', 'bed': 'bedroom',
        'sleeping room': 'bedroom', 'guest room': 'bedroom', 'guest bedroom': 'bedroom',

        // Master bedroom
        'master bedroom': 'master_bedroom', 'master bed': 'master_bedroom', 'master': 'master_bedroom',
        'main bedroom': 'master_bedroom', 'primary bedroom': 'master_bedroom', 'parents room': 'master_bedroom',

        // Kitchen variations
        'kitchen': 'kitchen', 'kitchenette': 'kitchen', 'cooking area': 'kitchen',
        'cook room': 'kitchen', 'pantry': 'kitchen',

        // Bathroom variations
        'bathroom': 'bathroom', 'bath room': 'bathroom', 'bath': 'bathroom',
        'toilet': 'bathroom', 'washroom': 'bathroom', 'restroom': 'bathroom',
        'wc': 'bathroom', 'lavatory': 'bathroom', 'loo': 'bathroom', 'powder room': 'bathroom',

        // Dining room
        'dining room': 'dining_room', 'dining': 'dining_room', 'diningroom': 'dining_room',
        'eating area': 'dining_room', 'dining area': 'dining_room',

        // Study/Office
        'study': 'study', 'study room': 'study', 'office': 'study', 'home office': 'study',
        'work room': 'study', 'workspace': 'study', 'den': 'study', 'library': 'study',

        // Storage
        'storage': 'storage', 'store room': 'storage', 'store': 'storage',
        'storeroom': 'storage', 'utility': 'utility', 'utility room': 'utility',
        'laundry': 'utility', 'laundry room': 'utility',

        // Balcony
        'balcony': 'balcony', 'terrace': 'balcony', 'patio': 'balcony', 'deck': 'balcony',
        'verandah': 'balcony', 'veranda': 'balcony', 'porch': 'balcony'
    };

    // Pattern: "[number] [room] of/with [size] sqm" - e.g., "2 bedrooms of 12 sqm"
    const patternWithSize = /(\d+)\s*(living\s*rooms?|bedrooms?|master\s*bedrooms?|kitchens?|bathrooms?|toilets?|dining\s*rooms?|stud(?:y|ies)|offices?|storage|utility|balcon(?:y|ies)|halls?|lounges?)(?:\s*(?:of|with|at|,|\s)?\s*(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2))?(?:\s*each)?/gi;

    let match;
    while ((match = patternWithSize.exec(message)) !== null) {
        const quantity = Math.min(parseInt(match[1], 10), 10);
        const roomName = match[2].toLowerCase().trim();
        const area = match[3] ? parseFloat(match[3]) : null;

        const roomType = normalizeRoomType(roomName, roomMappings);
        if (roomType && !foundTypes.has(roomType)) {
            foundTypes.add(roomType);
            rooms.push({
                type: roomType,
                quantity: quantity,
                minAreaSqm: area || ROOM_TYPES[roomType]?.defaultAreaSqm || 10
            });
        }
    }

    // Pattern: Single room mentions - "a living room", "the kitchen", "one bathroom"
    const singlePattern = /(?:a|an|one|the|single|need|want|with|and|,)\s*(living\s*room|bedroom|master\s*bedroom|kitchen|bathroom|toilet|washroom|dining\s*room|study|office|storage|utility|balcony|hall|lounge)/gi;

    while ((match = singlePattern.exec(message)) !== null) {
        const roomName = match[1].toLowerCase().trim();
        const roomType = normalizeRoomType(roomName, roomMappings);

        if (roomType && !foundTypes.has(roomType)) {
            foundTypes.add(roomType);
            rooms.push({
                type: roomType,
                quantity: 1,
                minAreaSqm: ROOM_TYPES[roomType]?.defaultAreaSqm || 10
            });
        }
    }

    // Pattern: "[room] of [size]" without quantity - "bedroom of 15 sqm"
    const roomSizePattern = /(living\s*room|bedroom|master\s*bedroom|kitchen|bathroom|toilet|dining\s*room|study|office|storage|balcony|hall)\s*(?:of|with|at)?\s*(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2)/gi;

    while ((match = roomSizePattern.exec(message)) !== null) {
        const roomName = match[1].toLowerCase().trim();
        const area = parseFloat(match[2]);
        const roomType = normalizeRoomType(roomName, roomMappings);

        if (roomType && !foundTypes.has(roomType)) {
            foundTypes.add(roomType);
            rooms.push({
                type: roomType,
                quantity: 1,
                minAreaSqm: Math.max(area, ROOM_TYPES[roomType]?.minAreaSqm || 3)
            });
        }
    }

    return rooms;
}

/**
 * Normalize room type names
 */
function normalizeRoomType(name, mappings) {
    // Clean up the name
    let cleaned = name.replace(/ies$/, 'y').replace(/s$/, '').trim();

    // Try direct mapping
    if (mappings[cleaned]) return mappings[cleaned];
    if (mappings[name]) return mappings[name];

    // Try with spaces removed
    const noSpace = cleaned.replace(/\s+/g, '');
    for (const [key, value] of Object.entries(mappings)) {
        if (key.replace(/\s+/g, '') === noSpace) return value;
    }

    // Try with underscores
    const underscored = cleaned.replace(/\s+/g, '_');
    if (ROOM_TYPES[underscored]) return underscored;

    return null;
}

/**
 * Extract plot dimensions
 */
function extractPlotDimensions(message) {
    const patterns = [
        /(?:plot|land|site)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*(?:m|meters?)?\s*(?:x|by|×|X)\s*(\d+(?:\.\d+)?)/i,
        /(\d+(?:\.\d+)?)\s*(?:m|meters?)?\s*(?:x|by|×|X)\s*(\d+(?:\.\d+)?)\s*(?:m|meters?)?\s*(?:plot|land|site)/i,
        /(\d+)\s*(?:m|meters?)?\s*(?:x|by|×|X)\s*(\d+)\s*(?:m|meters?)?(?:\s|$)/i
    ];

    for (const pattern of patterns) {
        const match = message.match(pattern);
        if (match) {
            const width = parseFloat(match[1]) * 1000;
            const length = parseFloat(match[2]) * 1000;
            if (width >= 3000 && length >= 3000) {
                return { width, length };
            }
        }
    }

    return null;
}

/**
 * Check if requirements are complete
 */
function hasCompleteRequirements(data) {
    return (
        data.totalAreaSqm &&
        data.totalAreaSqm > 0 &&
        data.rooms &&
        data.rooms.length > 0
    );
}

/**
 * Get prompt for missing requirements
 */
function getMissingRequirementsPrompt(data) {
    const hasArea = data.totalAreaSqm && data.totalAreaSqm > 0;
    const hasRooms = data.rooms && data.rooms.length > 0;

    if (!hasArea && !hasRooms) {
        return "I need more information. Please tell me the total area of your house and what rooms you need.";
    }

    if (!hasArea) {
        const roomArea = data.rooms.reduce((sum, r) => sum + (r.minAreaSqm * (r.quantity || 1)), 0);
        return `I have your rooms but I need the total house area. How big should the house be? (At least ${Math.ceil(roomArea * 1.15)} sqm is needed for your rooms)`;
    }

    if (!hasRooms) {
        return `I have the area (${data.totalAreaSqm} sqm) but I need to know what rooms you want. How many bedrooms, bathrooms, etc.?`;
    }

    return "Something is missing. Please provide the total area and rooms needed.";
}

/**
 * Generate progress response based on current data
 */
function generateProgressResponse(data) {
    const hasArea = data.totalAreaSqm && data.totalAreaSqm > 0;
    const hasRooms = data.rooms && data.rooms.length > 0;

    if (hasArea && hasRooms) {
        const totalRoomArea = data.rooms.reduce((sum, r) => sum + (r.minAreaSqm * (r.quantity || 1)), 0);

        if (totalRoomArea > data.totalAreaSqm) {
            return `There is a problem: your rooms need ${totalRoomArea} sqm total, but the house is only ${data.totalAreaSqm} sqm.

Please either increase the total area or reduce room sizes.`;
        }

        return `I have captured your requirements:

**Total Area:** ${data.totalAreaSqm} sqm
**Rooms:** ${formatRoomList(data.rooms)}
**Room Area:** ${totalRoomArea} sqm (${Math.round((totalRoomArea / data.totalAreaSqm) * 100)}% of total)

If this looks correct, click "Generate Floor Plan" below. Or tell me if you want to make any changes.`;
    }

    if (hasArea && !hasRooms) {
        return `I have noted: **${data.totalAreaSqm} sqm** total area.

Now, what rooms do you need? For example:
- "2 bedrooms, 1 living room, 1 kitchen, 1 bathroom"
- Or specify sizes: "bedroom of 12 sqm, kitchen 10 sqm"`;
    }

    if (!hasArea && hasRooms) {
        const totalRoomArea = data.rooms.reduce((sum, r) => sum + (r.minAreaSqm * (r.quantity || 1)), 0);
        return `I have your rooms (${formatRoomList(data.rooms)}).

What should be the total area of the house? It needs to be at least ${Math.ceil(totalRoomArea * 1.15)} sqm to fit these rooms comfortably.`;
    }

    return "Please tell me about your house requirements.";
}

/**
 * Format room list for display
 */
function formatRoomList(rooms) {
    return rooms.map(r => {
        const label = ROOM_TYPES[r.type]?.label || r.type.replace(/_/g, ' ');
        return `${r.quantity}x ${label} (${r.minAreaSqm} sqm)`;
    }).join(', ');
}

/**
 * Validate requirements before generation
 */
export function validateRequirements(data) {
    const errors = [];

    if (!data.totalAreaSqm || data.totalAreaSqm < 20) {
        errors.push('Total area must be at least 20 sqm');
    }

    if (!data.rooms || data.rooms.length === 0) {
        errors.push('At least one room is required');
    }

    const normalizedRooms = [];
    if (data.rooms) {
        for (const room of data.rooms) {
            const roomType = ROOM_TYPES[room.type];
            if (!roomType) {
                errors.push(`Unknown room type: ${room.type}`);
                continue;
            }

            normalizedRooms.push({
                type: room.type,
                label: roomType.label,
                quantity: room.quantity || 1,
                minAreaSqm: room.minAreaSqm,
                minAreaMm: sqmToSqmm(room.minAreaSqm),
                color: roomType.color,
                priority: roomType.priority,
                adjacentTo: roomType.adjacentTo,
                requiresWindow: roomType.requiresWindow
            });
        }
    }

    const totalRoomArea = normalizedRooms.reduce((sum, r) => sum + (r.minAreaSqm * r.quantity), 0);
    if (data.totalAreaSqm && totalRoomArea > data.totalAreaSqm) {
        errors.push(`Room area (${totalRoomArea} sqm) exceeds total area (${data.totalAreaSqm} sqm)`);
    }

    if (errors.length > 0) {
        return { success: false, error: errors.join('. ') };
    }

    return {
        success: true,
        data: {
            totalAreaSqm: data.totalAreaSqm,
            totalAreaMm: sqmToSqmm(data.totalAreaSqm),
            plotDimensions: data.plotDimensions || null,
            rooms: normalizedRooms
        }
    };
}

/**
 * Reset conversation
 */
export function resetConversation() {
    conversationHistory = [];
}

// Backward compatibility exports
export function processFormInput(formData) {
    return validateRequirements({
        totalAreaSqm: parseFloat(formData.totalArea),
        plotDimensions: formData.plotWidth && formData.plotLength ? {
            width: parseFloat(formData.plotWidth),
            length: parseFloat(formData.plotLength)
        } : null,
        rooms: formData.rooms
    });
}

export function parseChatInput(input, currentStep, currentData) {
    return parseNaturalLanguage(input, currentData);
}

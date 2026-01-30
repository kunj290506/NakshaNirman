/**
 * Intelligent Requirement Agent
 * Natural language understanding for floor plan requirements
 * Auto-generates when complete requirements are provided
 */

import { ROOM_TYPES } from '../utils/constants.js';
import { sqmToSqmm } from '../utils/geometry.js';

// Conversation state
let conversationHistory = [];

/**
 * Convert sq ft to sqm (1 sq ft = 0.0929 sqm)
 */
function sqftToSqm(sqft) {
    return Math.round(sqft * 0.0929 * 10) / 10;
}

/**
 * Main entry point - understand and respond to user message
 */
export function parseNaturalLanguage(input, context = {}) {
    const userMessage = input.trim();
    const lowerMessage = userMessage.toLowerCase();

    conversationHistory.push({ role: 'user', content: userMessage });

    const result = {
        understood: true,
        data: { ...context },
        response: '',
        complete: false,
        wantsToGenerate: false
    };

    // Check if this is a comprehensive request (long message with multiple requirements)
    const isComprehensive = isComprehensiveRequest(lowerMessage);

    // Detect intent
    const intent = detectIntent(lowerMessage);

    if (intent.type === 'greeting') {
        result.response = handleGreeting();
    } else if (intent.type === 'help') {
        result.response = handleHelp();
    } else if (intent.type === 'thanks') {
        result.response = handleThanks();
    } else if (intent.type === 'confirmation') {
        if (hasCompleteRequirements(result.data)) {
            result.wantsToGenerate = true;
        } else {
            result.response = getMissingRequirementsPrompt(result.data);
        }
    } else if (intent.type === 'reset') {
        result.data = {};
        result.response = "Starting fresh. What kind of house would you like to design?";
    } else if (intent.type === 'question') {
        result.response = handleQuestion(lowerMessage);
    } else {
        // Extract requirements from the message
        const extracted = extractRequirements(lowerMessage, result.data);
        result.data = extracted.data;

        if (extracted.foundSomething) {
            // Check if we have complete requirements
            if (hasCompleteRequirements(result.data)) {
                // If comprehensive request, auto-generate
                if (isComprehensive) {
                    result.wantsToGenerate = true;
                    result.response = "I have all your requirements. Generating your floor plan now...";
                } else {
                    result.response = generateProgressResponse(result.data);
                    result.complete = true;
                }
            } else {
                result.response = generateProgressResponse(result.data);
            }
        } else {
            result.response = handleUnknownInput(userMessage);
        }
    }

    conversationHistory.push({ role: 'agent', content: result.response });

    return result;
}

/**
 * Check if this is a comprehensive request with multiple requirements
 */
function isComprehensiveRequest(message) {
    // Long message with area AND room mentions
    const hasArea = /(\d+)\s*(sq\.?\s*ft|sqft|square\s*feet|sq\.?\s*m|sqm|m2)/i.test(message);
    const hasRooms = /(bedroom|bhk|living|kitchen|bathroom|hall)/i.test(message);
    const isLong = message.length > 100;

    // Keywords that suggest "just do it"
    const actionWords = /(build|create|design|make|suggest|generate|plan|layout|need|want)/i.test(message);

    return hasArea && hasRooms && (isLong || actionWords);
}

/**
 * Detect user's intent from message
 */
function detectIntent(message) {
    if (/^(hi|hello|hey|good\s*(morning|afternoon|evening))/i.test(message)) {
        return { type: 'greeting' };
    }

    if (/^(help|how\s*(do|can|to)|what\s*(can|should)|explain)/i.test(message)) {
        return { type: 'help' };
    }

    if (/^(thanks?|thank\s*you|thx|great|awesome)/i.test(message)) {
        return { type: 'thanks' };
    }

    if (/^(yes|yeah|yep|ok|okay|sure|confirm|generate|create|go\s*ahead|proceed|do\s*it)/i.test(message)) {
        return { type: 'confirmation' };
    }

    if (/^(cancel|reset|clear|start\s*over|new)/i.test(message)) {
        return { type: 'reset' };
    }

    if (/^(what|which|how|why|can\s*you)\s/i.test(message) || (message.endsWith('?') && message.length < 50)) {
        return { type: 'question' };
    }

    return { type: 'requirements' };
}

/**
 * Handle greetings
 */
function handleGreeting() {
    return "Hello! I am here to help you design a floor plan. Tell me about your house - the total area and what rooms you need.";
}

/**
 * Handle help requests
 */
function handleHelp() {
    return `I can help you create a floor plan. Here is how:

**Tell me the size:** "1220 sq ft" or "100 sqm"

**Tell me the rooms:** "2BHK with living room, kitchen, 2 bathrooms"

**Or describe everything at once:**
"I have 1200 sq ft and need a 2BHK with living room, kitchen, 2 bathrooms, and parking"

I will generate the floor plan automatically when I have all the details.`;
}

/**
 * Handle thanks
 */
function handleThanks() {
    return "You are welcome! Let me know if you need any changes to the floor plan.";
}

/**
 * Handle questions
 */
function handleQuestion(message) {
    if (/what\s*(rooms?|types?)/i.test(message)) {
        return `Available room types:
- Living Room / Hall
- Bedroom
- Master Bedroom
- Kitchen
- Bathroom
- Dining Room
- Study / Office
- Storage
- Balcony
- Parking

Just tell me what you need!`;
    }

    if (/vastu/i.test(message)) {
        return `For Vastu-friendly layout:
- Main entrance: North or East
- Kitchen: Southeast
- Master bedroom: Southwest
- Living room: North or East
- Bathroom: West or Northwest

Tell me your requirements and I will create a suitable layout.`;
    }

    return "Just describe your requirements - the area and rooms you need - and I will create the floor plan.";
}

/**
 * Handle unknown input
 */
function handleUnknownInput(message) {
    if (message.length < 5) {
        return "Could you provide more details? Tell me the area and rooms you need.";
    }

    return "I need a bit more clarity. Please tell me:\n- Total area (in sq ft or sqm)\n- Rooms you need (bedrooms, kitchen, etc.)";
}

/**
 * Extract requirements from message
 */
function extractRequirements(message, existingData) {
    const data = { ...existingData };
    let foundSomething = false;

    // Extract area (supports sq ft and sqm)
    const area = extractArea(message);
    if (area !== null) {
        data.totalAreaSqm = area;
        foundSomething = true;
    }

    // Extract rooms from BHK or explicit room mentions
    const rooms = extractRooms(message);
    if (rooms.length > 0) {
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
 * Extract area from natural language (supports sq ft and sqm)
 */
function extractArea(message) {
    // Square feet patterns
    const sqftPatterns = [
        /(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft|square\s*feet|sft)/i,
        /(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:sq\.?\s*feet)/i,
        /area\s*(?:of|is|:)?\s*(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft|feet)/i,
        /(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft)?\s*(?:plot|area|land|site)/i
    ];

    for (const pattern of sqftPatterns) {
        const match = message.match(pattern);
        if (match) {
            const sqft = parseFloat(match[1].replace(/,/g, ''));
            if (sqft >= 100 && sqft <= 100000) {
                return sqftToSqm(sqft);
            }
        }
    }

    // Square meter patterns
    const sqmPatterns = [
        /(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m(?:eters?)?|m²|m2)/i,
        /area\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m)/i,
        /(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m)?\s*(?:house|home|flat)/i
    ];

    for (const pattern of sqmPatterns) {
        const match = message.match(pattern);
        if (match) {
            const area = parseFloat(match[1]);
            if (area >= 15 && area <= 10000) {
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

    // Room mappings
    const roomMappings = {
        'living room': 'living_room', 'living': 'living_room', 'hall': 'living_room',
        'drawing room': 'living_room', 'lounge': 'living_room',
        'bedroom': 'bedroom', 'bed room': 'bedroom', 'bed': 'bedroom',
        'master bedroom': 'master_bedroom', 'master bed': 'master_bedroom', 'master': 'master_bedroom',
        'kitchen': 'kitchen', 'modular kitchen': 'kitchen', 'cooking': 'kitchen',
        'bathroom': 'bathroom', 'bath': 'bathroom', 'toilet': 'bathroom',
        'washroom': 'bathroom', 'attached': 'bathroom', 'common bathroom': 'bathroom',
        'attached bathroom': 'bathroom',
        'dining': 'dining_room', 'dining room': 'dining_room', 'dining area': 'dining_room',
        'study': 'study', 'office': 'study', 'work room': 'study',
        'storage': 'storage', 'store': 'storage', 'utility': 'utility',
        'balcony': 'balcony', 'terrace': 'balcony', 'patio': 'balcony',
        'parking': 'parking', 'car parking': 'parking', 'garage': 'parking',
        'car': 'parking', 'vehicle': 'parking', 'pooja': 'pooja_room', 'pooja room': 'pooja_room',
        'puja': 'pooja_room', 'mandir': 'pooja_room'
    };

    // Default room sizes in sqm (will be adjusted based on total area)
    const defaultSizes = {
        'living_room': 18,
        'bedroom': 12,
        'master_bedroom': 15,
        'kitchen': 9,
        'bathroom': 4,
        'dining_room': 10,
        'study': 8,
        'storage': 4,
        'balcony': 6,
        'parking': 12,
        'utility': 4,
        'pooja_room': 4
    };

    // Handle BHK format: 2BHK, 3 BHK, etc.
    const bhkMatch = message.match(/(\d)\s*bhk/i);
    if (bhkMatch) {
        const numBedrooms = parseInt(bhkMatch[1]);

        // Add master bedroom
        if (!foundTypes.has('master_bedroom')) {
            foundTypes.add('master_bedroom');
            rooms.push({
                type: 'master_bedroom',
                quantity: 1,
                minAreaSqm: defaultSizes['master_bedroom']
            });
        }

        // Add regular bedrooms
        if (numBedrooms > 1 && !foundTypes.has('bedroom')) {
            foundTypes.add('bedroom');
            rooms.push({
                type: 'bedroom',
                quantity: numBedrooms - 1,
                minAreaSqm: defaultSizes['bedroom']
            });
        }

        // BHK implies hall and kitchen
        if (!foundTypes.has('living_room')) {
            foundTypes.add('living_room');
            rooms.push({
                type: 'living_room',
                quantity: 1,
                minAreaSqm: defaultSizes['living_room']
            });
        }

        if (!foundTypes.has('kitchen')) {
            foundTypes.add('kitchen');
            rooms.push({
                type: 'kitchen',
                quantity: 1,
                minAreaSqm: defaultSizes['kitchen']
            });
        }
    }

    // Pattern: "[number] [room]" - e.g., "2 bedrooms", "2 bathrooms"
    const patternWithNumber = /(\d+)\s*(bedrooms?|living\s*rooms?|kitchens?|bathrooms?|toilets?|balcon(?:y|ies)|parking|dining)/gi;

    let match;
    while ((match = patternWithNumber.exec(message)) !== null) {
        const quantity = Math.min(parseInt(match[1], 10), 10);
        const roomName = match[2].toLowerCase();
        const roomType = normalizeRoomType(roomName, roomMappings);

        if (roomType && !foundTypes.has(roomType)) {
            foundTypes.add(roomType);
            rooms.push({
                type: roomType,
                quantity: quantity,
                minAreaSqm: defaultSizes[roomType] || 10
            });
        }
    }

    // Pattern: Single room mentions
    const singlePattern = /(living\s*room|hall|drawing\s*room|bedroom|master\s*bedroom|kitchen|bathroom|toilet|washroom|attached|common\s*bathroom|dining|balcony|terrace|parking|garage|pooja|mandir|study|office)/gi;

    while ((match = singlePattern.exec(message)) !== null) {
        const roomName = match[1].toLowerCase();
        const roomType = normalizeRoomType(roomName, roomMappings);

        if (roomType && !foundTypes.has(roomType)) {
            foundTypes.add(roomType);
            rooms.push({
                type: roomType,
                quantity: 1,
                minAreaSqm: defaultSizes[roomType] || 10
            });
        }
    }

    return rooms;
}

/**
 * Normalize room type names
 */
function normalizeRoomType(name, mappings) {
    let cleaned = name.replace(/ies$/, 'y').replace(/s$/, '').trim();

    if (mappings[cleaned]) return mappings[cleaned];
    if (mappings[name]) return mappings[name];

    // Check partial matches
    for (const [key, value] of Object.entries(mappings)) {
        if (name.includes(key) || key.includes(name)) {
            return value;
        }
    }

    return null;
}

/**
 * Extract plot dimensions
 */
function extractPlotDimensions(message) {
    // Pattern: "30x40", "30 x 40", "30 by 40", "30ft x 40ft"
    const patterns = [
        /(\d+)\s*(?:ft|feet|')?\s*(?:x|by|×|X)\s*(\d+)\s*(?:ft|feet|')?/i,
        /(\d+)\s*(?:m|meters?)?\s*(?:x|by|×|X)\s*(\d+)\s*(?:m|meters?)?/i
    ];

    for (const pattern of patterns) {
        const match = message.match(pattern);
        if (match) {
            let width = parseFloat(match[1]);
            let length = parseFloat(match[2]);

            // If it looks like feet (small numbers with no unit usually means feet in India)
            if (width < 100 && length < 100) {
                width = width * 304.8; // Convert feet to mm
                length = length * 304.8;
            } else {
                width = width * 1000; // Assume meters, convert to mm
                length = length * 1000;
            }

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
        return "Please tell me the total area and rooms you need.";
    }

    if (!hasArea) {
        return "What is the total area of your plot/house? (e.g., 1200 sq ft or 100 sqm)";
    }

    if (!hasRooms) {
        return "What rooms do you need? (e.g., 2BHK with kitchen, 2 bathrooms)";
    }

    return "Something is missing. Please provide area and rooms.";
}

/**
 * Generate progress response
 */
function generateProgressResponse(data) {
    const hasArea = data.totalAreaSqm && data.totalAreaSqm > 0;
    const hasRooms = data.rooms && data.rooms.length > 0;

    if (hasArea && hasRooms) {
        const totalRoomArea = data.rooms.reduce((sum, r) => sum + (r.minAreaSqm * (r.quantity || 1)), 0);

        if (totalRoomArea > data.totalAreaSqm * 0.85) {
            return `The rooms need ${totalRoomArea} sqm but only ${data.totalAreaSqm} sqm is available.\n\nPlease reduce room requirements or increase total area.`;
        }

        // Convert back to sq ft for Indian users
        const areaInSqft = Math.round(data.totalAreaSqm / 0.0929);

        return `**Your Requirements:**\n\nTotal Area: ${data.totalAreaSqm} sqm (${areaInSqft} sq ft)\nRooms: ${formatRoomList(data.rooms)}\n\nReady to generate. Click the button below or say "generate".`;
    }

    if (hasArea && !hasRooms) {
        const areaInSqft = Math.round(data.totalAreaSqm / 0.0929);
        return `Noted: ${data.totalAreaSqm} sqm (${areaInSqft} sq ft).\n\nWhat rooms do you need? (e.g., 2BHK, 3 bedrooms, living room, kitchen)`;
    }

    if (!hasArea && hasRooms) {
        return `Rooms noted: ${formatRoomList(data.rooms)}\n\nWhat is the total area? (e.g., 1200 sq ft)`;
    }

    return "Please tell me about your house requirements.";
}

/**
 * Format room list for display
 */
function formatRoomList(rooms) {
    return rooms.map(r => {
        const labels = {
            'living_room': 'Living Room',
            'bedroom': 'Bedroom',
            'master_bedroom': 'Master Bedroom',
            'kitchen': 'Kitchen',
            'bathroom': 'Bathroom',
            'dining_room': 'Dining',
            'study': 'Study',
            'storage': 'Storage',
            'balcony': 'Balcony',
            'parking': 'Parking',
            'utility': 'Utility',
            'pooja_room': 'Pooja Room'
        };
        const label = labels[r.type] || r.type.replace(/_/g, ' ');
        return r.quantity > 1 ? `${r.quantity}x ${label}` : label;
    }).join(', ');
}

/**
 * Validate requirements before generation
 */
export function validateRequirements(data) {
    const errors = [];

    if (!data.totalAreaSqm || data.totalAreaSqm < 15) {
        errors.push('Total area must be at least 15 sqm (160 sq ft)');
    }

    if (!data.rooms || data.rooms.length === 0) {
        errors.push('At least one room is required');
    }

    const normalizedRooms = [];
    if (data.rooms) {
        for (const room of data.rooms) {
            // Handle custom room types
            const roomConfig = ROOM_TYPES[room.type] || {
                label: room.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
                color: '#e2e8f0',
                minAreaSqm: 4,
                defaultAreaSqm: 10,
                priority: 5
            };

            normalizedRooms.push({
                type: room.type,
                label: roomConfig.label || room.type.replace(/_/g, ' '),
                quantity: room.quantity || 1,
                minAreaSqm: room.minAreaSqm || roomConfig.defaultAreaSqm,
                minAreaMm: sqmToSqmm(room.minAreaSqm || roomConfig.defaultAreaSqm),
                color: roomConfig.color,
                priority: roomConfig.priority || 5,
                adjacentTo: roomConfig.adjacentTo || [],
                requiresWindow: roomConfig.requiresWindow !== false
            });
        }
    }

    const totalRoomArea = normalizedRooms.reduce((sum, r) => sum + (r.minAreaSqm * r.quantity), 0);
    if (data.totalAreaSqm && totalRoomArea > data.totalAreaSqm * 0.9) {
        // Auto-adjust room sizes to fit
        const scaleFactor = (data.totalAreaSqm * 0.75) / totalRoomArea;
        normalizedRooms.forEach(room => {
            room.minAreaSqm = Math.max(room.minAreaSqm * scaleFactor, 3);
            room.minAreaMm = sqmToSqmm(room.minAreaSqm);
        });
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

// Backward compatibility
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

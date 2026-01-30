/**
 * Intelligent Requirement Agent
 * Uses Gemini AI for natural language understanding
 * Falls back to regex-based parsing if AI unavailable
 */

import { ROOM_TYPES } from '../utils/constants.js';
import { sqmToSqmm } from '../utils/geometry.js';
import { processWithAI, isAIAvailable } from '../services/aiService.js';

/**
 * Process user input with AI or fallback to regex
 */
export async function parseNaturalLanguage(input, context = {}) {
    const userMessage = input.trim();

    // Try AI first if available
    if (isAIAvailable()) {
        const aiResult = await processWithAI(userMessage, context);

        if (aiResult.success && aiResult.data) {
            const data = aiResult.data;

            // Merge AI response with existing context
            const mergedContext = {
                ...context,
                totalAreaSqm: data.totalAreaSqm || context.totalAreaSqm,
                rooms: mergeRooms(context.rooms || [], data.rooms || []),
                plotDimensions: data.plotDimensions || context.plotDimensions
            };

            return {
                understood: data.understood,
                data: mergedContext,
                response: data.response,
                complete: !data.needsMoreInfo && hasCompleteRequirements(mergedContext),
                wantsToGenerate: data.readyToGenerate && hasCompleteRequirements(mergedContext)
            };
        }
    }

    // Fallback to regex-based parsing
    return parseWithRegex(userMessage, context);
}

/**
 * Merge new rooms with existing rooms
 */
function mergeRooms(existing, newRooms) {
    const merged = [...existing];

    for (const newRoom of newRooms) {
        const existingIndex = merged.findIndex(r => r.type === newRoom.type);
        if (existingIndex >= 0) {
            merged[existingIndex] = {
                ...merged[existingIndex],
                ...newRoom,
                quantity: newRoom.quantity || merged[existingIndex].quantity,
                minAreaSqm: newRoom.areaSqm || newRoom.minAreaSqm || merged[existingIndex].minAreaSqm
            };
        } else {
            merged.push({
                type: newRoom.type,
                quantity: newRoom.quantity || 1,
                minAreaSqm: newRoom.areaSqm || newRoom.minAreaSqm || getDefaultRoomSize(newRoom.type)
            });
        }
    }

    return merged;
}

/**
 * Get default room size for a type
 */
function getDefaultRoomSize(type) {
    const defaults = {
        living_room: 18, bedroom: 12, master_bedroom: 15,
        kitchen: 9, bathroom: 4, dining_room: 10,
        study: 8, storage: 4, balcony: 6, parking: 14,
        utility: 4, pooja_room: 4
    };
    return defaults[type] || 10;
}

/**
 * Fallback regex-based parsing
 */
function parseWithRegex(input, context) {
    const message = input.toLowerCase();
    const data = { ...context };
    let foundSomething = false;
    let response = '';

    // Check for greetings
    if (/^(hi|hello|hey|good\s*(morning|afternoon|evening))/.test(message)) {
        return {
            understood: true,
            data,
            response: "Hello! I'm your Floor Plan Agent. Tell me about your house - the total area and rooms you need. For example: 'I have 1200 sq ft and need a 3BHK with 2 bathrooms'",
            complete: false,
            wantsToGenerate: false
        };
    }

    // Check for help
    if (/^(help|how|what can)/.test(message)) {
        return {
            understood: true,
            data,
            response: "I can help you design a floor plan! Just tell me:\n\n1. Total area (e.g., '1200 sq ft' or '100 sqm')\n2. Rooms you need (e.g., '3BHK', '2 bedrooms, 1 kitchen')\n\nI'll create a layout with proper room placement and a central corridor.",
            complete: false,
            wantsToGenerate: false
        };
    }

    // Check for confirmation
    if (/^(yes|yeah|ok|okay|sure|generate|create|build|go ahead|proceed)/.test(message)) {
        if (hasCompleteRequirements(data)) {
            return {
                understood: true,
                data,
                response: "Generating your floor plan...",
                complete: true,
                wantsToGenerate: true
            };
        }
    }

    // Extract area (sq ft or sqm)
    const sqftMatch = message.match(/(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft|square\s*feet)/i);
    const sqmMatch = message.match(/(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m2|square\s*m)/i);

    if (sqftMatch) {
        const sqft = parseFloat(sqftMatch[1].replace(/,/g, ''));
        data.totalAreaSqm = Math.round(sqft * 0.0929 * 10) / 10;
        foundSomething = true;
    } else if (sqmMatch) {
        data.totalAreaSqm = parseFloat(sqmMatch[1]);
        foundSomething = true;
    }

    // Extract BHK
    const bhkMatch = message.match(/(\d)\s*bhk/i);
    if (bhkMatch) {
        const numBedrooms = parseInt(bhkMatch[1]);
        if (!data.rooms) data.rooms = [];

        // Add master bedroom
        if (!data.rooms.find(r => r.type === 'master_bedroom')) {
            data.rooms.push({ type: 'master_bedroom', quantity: 1, minAreaSqm: 15 });
        }

        // Add regular bedrooms
        if (numBedrooms > 1 && !data.rooms.find(r => r.type === 'bedroom')) {
            data.rooms.push({ type: 'bedroom', quantity: numBedrooms - 1, minAreaSqm: 12 });
        }

        // Add living room and kitchen (part of BHK)
        if (!data.rooms.find(r => r.type === 'living_room')) {
            data.rooms.push({ type: 'living_room', quantity: 1, minAreaSqm: 18 });
        }
        if (!data.rooms.find(r => r.type === 'kitchen')) {
            data.rooms.push({ type: 'kitchen', quantity: 1, minAreaSqm: 9 });
        }

        foundSomething = true;
    }

    // Extract individual room mentions
    const roomPatterns = [
        { pattern: /(\d+)\s*(?:bed\s*rooms?|bedrooms?)/i, type: 'bedroom' },
        { pattern: /(\d+)\s*(?:bath\s*rooms?|bathrooms?|toilets?)/i, type: 'bathroom' },
        { pattern: /living\s*room|hall|drawing\s*room/i, type: 'living_room', qty: 1 },
        { pattern: /kitchen/i, type: 'kitchen', qty: 1 },
        { pattern: /dining/i, type: 'dining_room', qty: 1 },
        { pattern: /balcony/i, type: 'balcony', qty: 1 },
        { pattern: /parking|garage/i, type: 'parking', qty: 1 },
        { pattern: /study|office/i, type: 'study', qty: 1 },
        { pattern: /pooja|puja|mandir/i, type: 'pooja_room', qty: 1 }
    ];

    if (!data.rooms) data.rooms = [];

    for (const { pattern, type, qty } of roomPatterns) {
        const match = message.match(pattern);
        if (match) {
            const quantity = qty || parseInt(match[1]) || 1;
            const existing = data.rooms.find(r => r.type === type);
            if (!existing) {
                data.rooms.push({ type, quantity, minAreaSqm: getDefaultRoomSize(type) });
                foundSomething = true;
            }
        }
    }

    // Generate response
    if (foundSomething) {
        if (hasCompleteRequirements(data)) {
            const areaFt = Math.round(data.totalAreaSqm / 0.0929);
            const roomList = data.rooms.map(r => `${r.quantity}x ${r.type.replace(/_/g, ' ')}`).join(', ');
            response = `Got it! ${areaFt} sq ft with ${roomList}.\n\nReady to generate your floor plan. Click the button below or say "generate".`;
        } else if (data.totalAreaSqm && (!data.rooms || data.rooms.length === 0)) {
            response = `Noted: ${Math.round(data.totalAreaSqm / 0.0929)} sq ft.\n\nWhat rooms do you need? (e.g., "3BHK with 2 bathrooms")`;
        } else if (data.rooms && data.rooms.length > 0 && !data.totalAreaSqm) {
            response = `Got the rooms. What's the total area? (e.g., "1200 sq ft")`;
        } else {
            response = "Please provide more details about your requirements.";
        }
    } else {
        response = "I didn't quite catch that. Please tell me the total area (e.g., '1200 sq ft') and rooms you need (e.g., '3BHK, 2 bathrooms').";
    }

    return {
        understood: foundSomething,
        data,
        response,
        complete: hasCompleteRequirements(data),
        wantsToGenerate: false
    };
}

/**
 * Check if requirements are complete
 */
function hasCompleteRequirements(data) {
    return data.totalAreaSqm > 0 && data.rooms && data.rooms.length > 0;
}

/**
 * Validate and normalize requirements for generation
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
                minAreaSqm: room.minAreaSqm || room.areaSqm || roomConfig.defaultAreaSqm,
                minAreaMm: sqmToSqmm(room.minAreaSqm || room.areaSqm || roomConfig.defaultAreaSqm),
                color: roomConfig.color,
                priority: roomConfig.priority || 5,
                adjacentTo: roomConfig.adjacentTo || [],
                requiresWindow: roomConfig.requiresWindow !== false
            });
        }
    }

    // Scale rooms to fit if needed
    const totalRoomArea = normalizedRooms.reduce((sum, r) => sum + (r.minAreaSqm * r.quantity), 0);
    if (data.totalAreaSqm && totalRoomArea > data.totalAreaSqm * 0.85) {
        const scale = (data.totalAreaSqm * 0.75) / totalRoomArea;
        normalizedRooms.forEach(room => {
            room.minAreaSqm = Math.max(room.minAreaSqm * scale, 3);
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
 * Reset conversation (for AI service)
 */
export function resetConversation() {
    // Import dynamically to avoid circular dependency
    import('../services/aiService.js').then(ai => ai.resetConversation());
}

/**
 * Requirement Agent - Advanced Natural Language Processing
 * Parses user input and extracts floor plan requirements
 */

import { ROOM_TYPES, MIN_ROOM_AREA_SQM } from '../utils/constants.js';
import { sqmToSqmm } from '../utils/geometry.js';

/**
 * Parse natural language input to extract requirements
 * @param {string} input - User's message
 * @param {Object} context - Current conversation context
 * @returns {Object} Parsed result
 */
export function parseNaturalLanguage(input, context = {}) {
    const result = {
        understood: false,
        data: { ...context },
        response: '',
        complete: false
    };

    // Normalize input
    const normalizedInput = input.toLowerCase().trim();

    // Extract total area
    const extractedArea = extractArea(normalizedInput);
    if (extractedArea !== null) {
        result.data.totalAreaSqm = extractedArea;
        result.understood = true;
    }

    // Extract rooms
    const extractedRooms = extractRooms(normalizedInput);
    if (extractedRooms.length > 0) {
        // Merge or replace rooms
        if (result.data.rooms && result.data.rooms.length > 0) {
            // Merge: add new rooms, update existing
            extractedRooms.forEach(newRoom => {
                const existingIndex = result.data.rooms.findIndex(r => r.type === newRoom.type);
                if (existingIndex >= 0) {
                    result.data.rooms[existingIndex] = newRoom;
                } else {
                    result.data.rooms.push(newRoom);
                }
            });
        } else {
            result.data.rooms = extractedRooms;
        }
        result.understood = true;
    }

    // Extract plot dimensions if mentioned
    const plotDims = extractPlotDimensions(normalizedInput);
    if (plotDims) {
        result.data.plotDimensions = plotDims;
        result.understood = true;
    }

    // Check for confirmation or action requests
    if (isConfirmation(normalizedInput)) {
        result.wantsToGenerate = true;
        result.understood = true;
    }

    // Generate appropriate response
    const responseData = generateResponse(result.data, result.understood, normalizedInput);
    result.response = responseData.text;
    result.complete = responseData.complete;
    result.needsMoreInfo = responseData.needsMore;

    return result;
}

/**
 * Extract area from text
 */
function extractArea(input) {
    // Patterns for area extraction
    const patterns = [
        // "100 sqm", "100 sq m", "100 m2", "100m²", "100 square meters"
        /(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m(?:eters?)?|m²|m2|square\s*m(?:eters?)?)/i,
        // "area of 100", "area is 100", "area: 100"
        /(?:total\s*)?area\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)/i,
        // "100 sqm house/home/flat/apartment"
        /(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2)?\s*(?:house|home|flat|apartment|floor\s*plan)/i,
        // "house of 100 sqm", "home with 100 sqm"
        /(?:house|home|flat|apartment)\s*(?:of|with|having)?\s*(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2)?/i,
        // "need 100 sqm", "want 100 sqm", "looking for 100 sqm"
        /(?:need|want|looking\s*for|require|build)\s*(?:a\s*)?(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2)?/i,
        // Just a number if it's reasonable for area (50-500)
        /^(\d+)$/
    ];

    for (const pattern of patterns) {
        const match = input.match(pattern);
        if (match) {
            const area = parseFloat(match[1]);
            // Validate reasonable area range
            if (area >= 20 && area <= 10000) {
                return area;
            }
        }
    }

    return null;
}

/**
 * Extract rooms from text
 */
function extractRooms(input) {
    const rooms = [];
    const foundTypes = new Set();

    // Room type mapping with aliases
    const roomAliases = {
        'living room': 'living_room',
        'livingroom': 'living_room',
        'living': 'living_room',
        'lounge': 'living_room',
        'hall': 'living_room',
        'drawing room': 'living_room',
        'bedroom': 'bedroom',
        'bed room': 'bedroom',
        'bed': 'bedroom',
        'master bedroom': 'master_bedroom',
        'master bed': 'master_bedroom',
        'master': 'master_bedroom',
        'main bedroom': 'master_bedroom',
        'kitchen': 'kitchen',
        'kitchenette': 'kitchen',
        'bathroom': 'bathroom',
        'bath room': 'bathroom',
        'bath': 'bathroom',
        'toilet': 'bathroom',
        'washroom': 'bathroom',
        'restroom': 'bathroom',
        'wc': 'bathroom',
        'dining room': 'dining_room',
        'diningroom': 'dining_room',
        'dining': 'dining_room',
        'study': 'study',
        'study room': 'study',
        'office': 'study',
        'home office': 'study',
        'work room': 'study',
        'storage': 'storage',
        'store room': 'storage',
        'store': 'storage',
        'utility': 'utility',
        'utility room': 'utility',
        'laundry': 'utility',
        'balcony': 'balcony',
        'balconies': 'balcony',
        'terrace': 'balcony',
        'patio': 'balcony'
    };

    // Pattern 1: "2 bedrooms of 12 sqm" or "2 bedrooms 12 sqm each"
    const patternWithArea = /(\d+)\s*(living\s*rooms?|bedrooms?|master\s*bedrooms?|kitchens?|bathrooms?|toilets?|dining\s*rooms?|stud(?:y|ies)|storage|utility|balcon(?:y|ies)|hall|lounge|office)(?:\s*(?:of|at|with|,|each))?\s*(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2)?(?:\s*each)?/gi;

    let match;
    while ((match = patternWithArea.exec(input)) !== null) {
        const quantity = parseInt(match[1], 10);
        const roomName = match[2].toLowerCase().trim();
        const area = parseFloat(match[3]);

        const roomType = normalizeRoomType(roomName, roomAliases);
        if (roomType && !foundTypes.has(roomType)) {
            foundTypes.add(roomType);
            rooms.push({
                type: roomType,
                quantity: Math.min(quantity, 10),
                minAreaSqm: Math.max(area, ROOM_TYPES[roomType]?.minAreaSqm || 4)
            });
        }
    }

    // Pattern 2: "2 bedrooms, 1 kitchen" (without area)
    const patternWithoutArea = /(\d+)\s*(living\s*rooms?|bedrooms?|master\s*bedrooms?|kitchens?|bathrooms?|toilets?|dining\s*rooms?|stud(?:y|ies)|storage|utility|balcon(?:y|ies)|hall|lounge|office)/gi;

    while ((match = patternWithoutArea.exec(input)) !== null) {
        const quantity = parseInt(match[1], 10);
        const roomName = match[2].toLowerCase().trim();

        const roomType = normalizeRoomType(roomName, roomAliases);
        if (roomType && !foundTypes.has(roomType)) {
            foundTypes.add(roomType);
            rooms.push({
                type: roomType,
                quantity: Math.min(quantity, 10),
                minAreaSqm: ROOM_TYPES[roomType]?.defaultAreaSqm || 10
            });
        }
    }

    // Pattern 3: Single room mentions "a living room", "kitchen", "bedroom"
    const singleRoomPattern = /(?:a|one|the|with|and|,)\s*(living\s*room|bedroom|master\s*bedroom|kitchen|bathroom|toilet|dining\s*room|study|storage|utility|balcony|hall|lounge|office)/gi;

    while ((match = singleRoomPattern.exec(input)) !== null) {
        const roomName = match[1].toLowerCase().trim();

        const roomType = normalizeRoomType(roomName, roomAliases);
        if (roomType && !foundTypes.has(roomType)) {
            foundTypes.add(roomType);
            rooms.push({
                type: roomType,
                quantity: 1,
                minAreaSqm: ROOM_TYPES[roomType]?.defaultAreaSqm || 10
            });
        }
    }

    // Pattern 4: Room with area but no quantity "bedroom of 12 sqm"
    const roomAreaPattern = /(living\s*room|bedroom|master\s*bedroom|kitchen|bathroom|toilet|dining\s*room|study|storage|utility|balcony|hall|lounge|office)\s*(?:of|at|with|,)?\s*(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2)/gi;

    while ((match = roomAreaPattern.exec(input)) !== null) {
        const roomName = match[1].toLowerCase().trim();
        const area = parseFloat(match[2]);

        const roomType = normalizeRoomType(roomName, roomAliases);
        if (roomType && !foundTypes.has(roomType)) {
            foundTypes.add(roomType);
            rooms.push({
                type: roomType,
                quantity: 1,
                minAreaSqm: Math.max(area, ROOM_TYPES[roomType]?.minAreaSqm || 4)
            });
        }
    }

    return rooms;
}

/**
 * Normalize room type to our standard keys
 */
function normalizeRoomType(roomName, aliases) {
    // Remove trailing 's' for plurals
    const singular = roomName.replace(/ies$/, 'y').replace(/s$/, '').trim();

    // Check direct match first
    if (aliases[singular]) return aliases[singular];
    if (aliases[roomName]) return aliases[roomName];

    // Check if it's already a valid type
    if (ROOM_TYPES[singular]) return singular;
    if (ROOM_TYPES[roomName]) return roomName;

    // Try with underscores
    const withUnderscore = singular.replace(/\s+/g, '_');
    if (ROOM_TYPES[withUnderscore]) return withUnderscore;

    return null;
}

/**
 * Extract plot dimensions
 */
function extractPlotDimensions(input) {
    // "plot of 10x12", "10m x 12m plot", "10 by 12 meters"
    const patterns = [
        /(?:plot|land|site)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*(?:m|meters?)?\s*(?:x|by|×)\s*(\d+(?:\.\d+)?)\s*(?:m|meters?)?/i,
        /(\d+(?:\.\d+)?)\s*(?:m|meters?)?\s*(?:x|by|×)\s*(\d+(?:\.\d+)?)\s*(?:m|meters?)?\s*(?:plot|land|site)/i
    ];

    for (const pattern of patterns) {
        const match = input.match(pattern);
        if (match) {
            const width = parseFloat(match[1]) * 1000; // Convert m to mm
            const length = parseFloat(match[2]) * 1000;
            if (width >= 3000 && length >= 3000) {
                return { width, length };
            }
        }
    }

    return null;
}

/**
 * Check if user wants to confirm/generate
 */
function isConfirmation(input) {
    const confirmWords = [
        'yes', 'ok', 'okay', 'sure', 'confirm', 'generate', 'create', 'build',
        'go ahead', 'proceed', 'make it', 'do it', 'looks good', 'perfect',
        'that is correct', 'that\'s correct', 'correct', 'right', 'exactly'
    ];

    return confirmWords.some(word => input.includes(word));
}

/**
 * Generate contextual response
 */
function generateResponse(data, understood, input) {
    const hasArea = data.totalAreaSqm && data.totalAreaSqm > 0;
    const hasRooms = data.rooms && data.rooms.length > 0;

    // If we didn't understand anything
    if (!understood) {
        return {
            text: "I could not understand that. Please specify your requirements clearly.\n\nFor example:\n- \"I need a 100 sqm house\"\n- \"2 bedrooms of 12 sqm each, 1 kitchen, 1 bathroom\"\n- \"living room 20 sqm, master bedroom 15 sqm\"",
            needsMore: true,
            complete: false
        };
    }

    // If we have area but no rooms
    if (hasArea && !hasRooms) {
        return {
            text: `I have noted the total area: ${data.totalAreaSqm} square meters.\n\nNow please tell me what rooms you need. For example:\n- \"2 bedrooms of 12 sqm each\"\n- \"1 living room, 1 kitchen, 2 bathrooms\"`,
            needsMore: true,
            complete: false
        };
    }

    // If we have rooms but no area
    if (!hasArea && hasRooms) {
        const totalRoomArea = data.rooms.reduce((sum, r) => sum + (r.minAreaSqm * (r.quantity || 1)), 0);
        return {
            text: `I have captured your rooms. The total room area is ${totalRoomArea} square meters.\n\nWhat should be the total area of your house? It should be at least ${Math.ceil(totalRoomArea * 1.15)} sqm to allow for walls and circulation.`,
            needsMore: true,
            complete: false
        };
    }

    // If we have both area and rooms
    if (hasArea && hasRooms) {
        const totalRoomArea = data.rooms.reduce((sum, r) => sum + (r.minAreaSqm * (r.quantity || 1)), 0);

        // Check if rooms fit in area
        if (totalRoomArea > data.totalAreaSqm) {
            return {
                text: `There is a problem: your rooms need ${totalRoomArea} sqm, but you only have ${data.totalAreaSqm} sqm total area.\n\nPlease either:\n- Increase the total area to at least ${Math.ceil(totalRoomArea * 1.1)} sqm\n- Reduce the room sizes`,
                needsMore: true,
                complete: false
            };
        }

        return {
            text: `I have all your requirements:\n\n**Total Area:** ${data.totalAreaSqm} sqm\n**Rooms:** ${formatRoomList(data.rooms)}\n**Room Area:** ${totalRoomArea} sqm\n\nPlease review the summary below and click "Generate Floor Plan" to proceed.`,
            complete: true,
            needsMore: false
        };
    }

    // Default fallback
    return {
        text: "Please tell me about your floor plan requirements. Start with the total area or the rooms you need.",
        needsMore: true,
        complete: false
    };
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
 * Validate complete requirements
 */
export function validateRequirements(data) {
    const errors = [];

    if (!data.totalAreaSqm || data.totalAreaSqm < 20) {
        errors.push('Total area must be at least 20 square meters');
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

            if (room.minAreaSqm < roomType.minAreaSqm) {
                errors.push(`${roomType.label} requires at least ${roomType.minAreaSqm} sqm`);
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
        return { success: false, error: errors[0] };
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

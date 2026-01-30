/**
 * Requirement Agent (Natural Language Version)
 * Parses natural language input and extracts floor plan requirements
 */

import { ROOM_TYPES, MIN_ROOM_AREA_SQM, MIN_PLOT_DIMENSION, MAX_PLOT_DIMENSION } from '../utils/constants.js';
import { sqmToSqmm } from '../utils/geometry.js';

/**
 * Parse natural language input to extract requirements
 * @param {string} input - User's natural language message
 * @param {Object} context - Current conversation context
 * @returns {Object} Parsed result with extracted data
 */
export function parseNaturalLanguage(input, context = {}) {
    const lowerInput = input.toLowerCase();
    const result = {
        understood: false,
        data: { ...context },
        response: '',
        complete: false
    };

    // Extract total area
    const areaMatch = lowerInput.match(/(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|square\s*meters?|m2)/i) ||
        lowerInput.match(/(?:area|size|total|house)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)/i);

    if (areaMatch) {
        const area = parseFloat(areaMatch[1]);
        if (area >= 20 && area <= 10000) {
            result.data.totalAreaSqm = area;
            result.understood = true;
        }
    }

    // Extract rooms
    const extractedRooms = extractRooms(lowerInput);
    if (extractedRooms.length > 0) {
        // Merge with existing rooms or replace
        result.data.rooms = extractedRooms;
        result.understood = true;
    }

    // Extract plot dimensions
    const plotMatch = lowerInput.match(/(?:plot|land|site)\s*(?:of|is|:)?\s*(\d+)\s*(?:x|by|×)\s*(\d+)/i);
    if (plotMatch) {
        const width = parseInt(plotMatch[1]) * 1000; // Assume meters, convert to mm
        const height = parseInt(plotMatch[2]) * 1000;
        if (width >= MIN_PLOT_DIMENSION && height >= MIN_PLOT_DIMENSION) {
            result.data.plotDimensions = { width, length: height };
            result.understood = true;
        }
    }

    // Generate response based on what we have
    result.response = generateResponse(result.data);
    result.complete = isComplete(result.data);

    return result;
}

/**
 * Extract rooms from natural language
 */
function extractRooms(input) {
    const rooms = [];

    // Common room patterns
    const patterns = [
        // "2 bedrooms of 12 sqm" or "2 bedrooms 12 sqm each"
        /(\d+)\s*(living\s*rooms?|bedrooms?|master\s*bedrooms?|kitchens?|bathrooms?|toilets?|dining\s*rooms?|stud(?:y|ies)|storage|utility|balcon(?:y|ies))\s*(?:of|at|with|each)?\s*(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2)?/gi,
        // "bedroom of 12 sqm" (quantity 1 implied)
        /(?:a|one|1)?\s*(living\s*room|bedroom|master\s*bedroom|kitchen|bathroom|toilet|dining\s*room|study|storage|utility|balcony)\s*(?:of|at|with)?\s*(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2)?/gi,
    ];

    // Pattern 1: quantity + type + area
    let match;
    const pattern1 = /(\d+)\s*(living\s*rooms?|bedrooms?|master\s*bedrooms?|kitchens?|bathrooms?|toilets?|dining\s*rooms?|stud(?:y|ies)|storage|utility|balcon(?:y|ies))\s*(?:of|at|with|each|,)?\s*(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2)?(?:\s*each)?/gi;

    while ((match = pattern1.exec(input)) !== null) {
        const quantity = parseInt(match[1], 10);
        const roomType = mapRoomType(match[2]);
        const area = parseFloat(match[3]);

        if (roomType && area > 0) {
            rooms.push({
                type: roomType,
                quantity: quantity,
                minAreaSqm: area
            });
        }
    }

    // Pattern 2: Rooms without explicit area (use defaults)
    if (rooms.length === 0) {
        const pattern2 = /(\d+)\s*(living\s*rooms?|bedrooms?|master\s*bedrooms?|kitchens?|bathrooms?|toilets?|dining\s*rooms?|stud(?:y|ies)|storage|utility|balcon(?:y|ies))/gi;

        while ((match = pattern2.exec(input)) !== null) {
            const quantity = parseInt(match[1], 10);
            const roomType = mapRoomType(match[2]);

            if (roomType) {
                rooms.push({
                    type: roomType,
                    quantity: quantity,
                    minAreaSqm: ROOM_TYPES[roomType]?.defaultAreaSqm || 10
                });
            }
        }
    }

    // Pattern 3: Single room mentions "a living room", "kitchen"
    if (rooms.length === 0) {
        const singleRoomPattern = /(?:a|one|the)?\s*(living\s*room|bedroom|master\s*bedroom|kitchen|bathroom|toilet|dining\s*room|study|storage|utility|balcony)/gi;

        while ((match = singleRoomPattern.exec(input)) !== null) {
            const roomType = mapRoomType(match[1]);

            if (roomType && !rooms.find(r => r.type === roomType)) {
                rooms.push({
                    type: roomType,
                    quantity: 1,
                    minAreaSqm: ROOM_TYPES[roomType]?.defaultAreaSqm || 10
                });
            }
        }
    }

    return rooms;
}

/**
 * Map natural language room names to our room type keys
 */
function mapRoomType(raw) {
    const normalized = raw.toLowerCase().replace(/\s+/g, '_').replace(/ies$/, 'y').replace(/s$/, '');

    const mappings = {
        'living_room': 'living_room',
        'livingroom': 'living_room',
        'living': 'living_room',
        'lounge': 'living_room',
        'bedroom': 'bedroom',
        'bed_room': 'bedroom',
        'master_bedroom': 'master_bedroom',
        'masterbed': 'master_bedroom',
        'master': 'master_bedroom',
        'kitchen': 'kitchen',
        'bathroom': 'bathroom',
        'bath': 'bathroom',
        'toilet': 'bathroom',
        'washroom': 'bathroom',
        'dining_room': 'dining_room',
        'diningroom': 'dining_room',
        'dining': 'dining_room',
        'study': 'study',
        'office': 'study',
        'storage': 'storage',
        'store': 'storage',
        'utility': 'utility',
        'balcony': 'balcony',
        'balcon': 'balcony'
    };

    return mappings[normalized] || null;
}

/**
 * Generate agent response based on current data
 */
function generateResponse(data) {
    const hasArea = data.totalAreaSqm && data.totalAreaSqm > 0;
    const hasRooms = data.rooms && data.rooms.length > 0;

    if (!hasArea && !hasRooms) {
        return {
            text: "I couldn't understand specific requirements from that. Could you tell me:\n\n• **Total area** of your house (e.g., \"100 sqm\")\n• **Rooms needed** (e.g., \"2 bedrooms of 12 sqm, 1 kitchen\")",
            needsMore: true
        };
    }

    if (hasArea && !hasRooms) {
        return {
            text: `Great! I've noted **${data.totalAreaSqm} m²** total area.\n\nNow, what rooms do you need? For example:\n• "2 bedrooms of 12 sqm each"\n• "1 living room, 1 kitchen, 1 bathroom"`,
            needsMore: true
        };
    }

    if (!hasArea && hasRooms) {
        const totalRoomArea = data.rooms.reduce((sum, r) => sum + (r.minAreaSqm * r.quantity), 0);
        return {
            text: `I've captured your rooms. What's the **total area** of your house?\n\n(Your rooms need at least ${totalRoomArea} m²)`,
            needsMore: true
        };
    }

    // Both area and rooms present
    return {
        text: "I've got all your requirements! Here's the summary:",
        complete: true
    };
}

/**
 * Check if requirements are complete
 */
function isComplete(data) {
    return (
        data.totalAreaSqm > 0 &&
        data.rooms &&
        data.rooms.length > 0
    );
}

/**
 * Validate complete requirements
 * @param {Object} data - Parsed requirements data
 * @returns {Object} {success: boolean, data?: Object, error?: string}
 */
export function validateRequirements(data) {
    const errors = [];

    // Validate total area
    if (!data.totalAreaSqm || data.totalAreaSqm <= 0) {
        errors.push('Total area is required');
    } else if (data.totalAreaSqm < 20) {
        errors.push('Total area must be at least 20 m²');
    }

    // Validate rooms
    if (!data.rooms || data.rooms.length === 0) {
        errors.push('At least one room is required');
    }

    // Check room areas
    const normalizedRooms = [];
    if (data.rooms) {
        for (const room of data.rooms) {
            const roomType = ROOM_TYPES[room.type];
            if (!roomType) {
                errors.push(`Unknown room type: ${room.type}`);
                continue;
            }

            if (room.minAreaSqm < roomType.minAreaSqm) {
                errors.push(`${roomType.label} requires at least ${roomType.minAreaSqm} m²`);
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

    // Check total room area
    const totalRoomArea = normalizedRooms.reduce((sum, r) => sum + (r.minAreaSqm * r.quantity), 0);
    if (data.totalAreaSqm && totalRoomArea > data.totalAreaSqm) {
        errors.push(`Total room area (${totalRoomArea} m²) exceeds available area (${data.totalAreaSqm} m²)`);
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

// Keep old function for backward compatibility
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

/**
 * Intelligent Requirement Agent
 * Uses Gemini AI for natural language understanding
 * Enhanced fallback parsing for typos, broken English, and informal language
 */

import { ROOM_TYPES } from '../utils/constants.js';
import { sqmToSqmm } from '../utils/geometry.js';
import { processWithAI, isAIAvailable } from '../services/aiService.js';

/**
 * Normalize text - fix common typos and variations
 */
function normalizeText(input) {
    let text = input.toLowerCase().trim();
    
    // Fix common typos and variations
    const corrections = {
        // Area units
        'squre': 'square', 'sqaure': 'square', 'sqr': 'square',
        'sq ft': 'sqft', 'sq.ft': 'sqft', 'sft': 'sqft', 'sqfeet': 'sqft',
        'sq m': 'sqm', 'sq.m': 'sqm', 'sqmeter': 'sqm', 'sqmetre': 'sqm',
        
        // Rooms
        'bedrrom': 'bedroom', 'bedrom': 'bedroom', 'bdrm': 'bedroom', 'bed room': 'bedroom',
        'batroom': 'bathroom', 'bathrom': 'bathroom', 'bath room': 'bathroom', 'washroom': 'bathroom',
        'tolet': 'toilet', 'toilat': 'toilet',
        'kichen': 'kitchen', 'kitchn': 'kitchen', 'kithcen': 'kitchen',
        'livng': 'living', 'livign': 'living', 'lving': 'living',
        'dning': 'dining', 'dinning': 'dining', 'dinig': 'dining',
        'stusy': 'study', 'stduy': 'study',
        'balcny': 'balcony', 'balkony': 'balcony',
        'poja': 'pooja', 'puja': 'pooja', 'mandir': 'pooja',
        
        // Actions
        'wnat': 'want', 'wnt': 'want', 'watn': 'want',
        'ned': 'need', 'nedd': 'need',
        'plase': 'please', 'pls': 'please',
        'genrate': 'generate', 'genarate': 'generate', 'cretae': 'create',
        'dsign': 'design', 'desgn': 'design', 'desgin': 'design',
        'hose': 'house', 'huose': 'house', 'hous': 'house',
        
        // Plot
        'plat': 'plot', 'plote': 'plot',
        'iregular': 'irregular', 'irregualr': 'irregular',
        'boundry': 'boundary', 'boundery': 'boundary',
        
        // Directions
        'nort': 'north', 'norht': 'north',
        'sout': 'south', 'souht': 'south',
        'est': 'east', 'esat': 'east',
        'wset': 'west', 'wets': 'west',
        
        // Numbers as words
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    };
    
    for (const [typo, correct] of Object.entries(corrections)) {
        text = text.replace(new RegExp(typo, 'gi'), correct);
    }
    
    return text;
}

/**
 * Process user input with AI or fallback to regex
 */
export async function parseNaturalLanguage(input, context = {}) {
    const userMessage = input.trim();
    const normalizedMessage = normalizeText(userMessage);

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
                plotDimensions: data.plotDimensions || context.plotDimensions,
                plotShape: data.plotShape || context.plotShape,
                plotBoundary: data.plotBoundary || context.plotBoundary,
                structureType: data.suggested_structure || context.structureType
            };

            return {
                understood: data.understood,
                data: mergedContext,
                response: data.response,
                thought_process: data.thought_process,
                complete: !data.needsMoreInfo && hasCompleteRequirements(mergedContext),
                wantsToGenerate: data.readyToGenerate && hasCompleteRequirements(mergedContext)
            };
        }
    }

    // Fallback to enhanced regex-based parsing
    return parseWithRegex(normalizedMessage, context, userMessage);
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
        utility: 4, pooja_room: 4, shop: 15
    };
    return defaults[type] || 10;
}

/**
 * Enhanced fallback regex-based parsing with typo tolerance
 */
function parseWithRegex(normalizedMessage, context, originalMessage) {
    const message = normalizedMessage;
    const data = { ...context };
    let foundSomething = false;
    let response = '';

    // Check for greetings
    if (/^(hi|hello|hey|good\s*(morning|afternoon|evening)|helo|hii)/.test(message)) {
        return {
            understood: true,
            data,
            response: "Hello! I'm your Floor Plan Agent. Tell me about your house - the total area and rooms you need. For example: 'I want 1200 sq ft with 3 bedrooms and 2 bathrooms'",
            complete: false,
            wantsToGenerate: false
        };
    }

    // Check for help
    if (/^(help|how|what can|hlp|halp)/.test(message)) {
        return {
            understood: true,
            data,
            response: "I can help you design a floor plan! Just tell me:\n\n1. Total area (e.g., '1200 sq ft' or '100 sqm')\n2. Rooms you need (e.g., '3BHK', '2 bedrooms, 1 kitchen')\n\nI'll create a layout with proper room placement.",
            complete: false,
            wantsToGenerate: false
        };
    }

    // Check for confirmation/generate commands
    if (/^(yes|yeah|ok|okay|sure|generate|create|build|go|proceed|make|do it|lets go|start)/.test(message)) {
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

    // Extract plot dimensions in MILLIMETERS (2080mm x 1675mm, 2080 x 1675 mm)
    const mmPlotMatch = message.match(/(\d+)\s*(?:mm)?\s*(?:x|by|into|\*)\s*(\d+)\s*(?:mm)/i);
    if (mmPlotMatch) {
        const width = parseFloat(mmPlotMatch[1]);
        const length = parseFloat(mmPlotMatch[2]);
        data.plotDimensions = { width, length };
        data.totalAreaSqm = (width * length) / 1000000; // mm2 to sqm
        data.plotShape = 'irregular'; // mm dimensions usually indicate detailed/irregular plots
        foundSomething = true;
    }
    
    // Extract plot dimensions in METERS (20m x 16m, 20 x 16 m)
    const mPlotMatch = message.match(/(\d+(?:\.\d+)?)\s*(?:m|meter|metre)?\s*(?:x|by|into|\*)\s*(\d+(?:\.\d+)?)\s*(?:m|meter|metre)/i);
    if (mPlotMatch && !data.plotDimensions) {
        const width = parseFloat(mPlotMatch[1]) * 1000; // m to mm
        const length = parseFloat(mPlotMatch[2]) * 1000;
        data.plotDimensions = { width, length };
        data.totalAreaSqm = (width * length) / 1000000;
        foundSomething = true;
    }

    // Extract plot dimensions in FEET (30x40, 30 by 40, 30 into 40 feet)
    const ftPlotMatch = message.match(/(\d+)\s*(?:x|by|into|\*)\s*(\d+)\s*(?:ft|feet|foot|plot|site)?/i);
    if (ftPlotMatch && !data.plotDimensions) {
        const width = parseFloat(ftPlotMatch[1]) * 304.8; // feet to mm
        const length = parseFloat(ftPlotMatch[2]) * 304.8;
        data.plotDimensions = { width, length };
        data.totalAreaSqm = (width * length) / 1000000; // mm2 to sqm
        foundSomething = true;
    }
    
    // Extract overall/bounding dimensions mentioned in text
    const boundingMatch = message.match(/(?:overall|bounding|main|plot)\s*(?:dimension|size)?[:\s]*(\d+)\s*(?:mm)?\s*(?:x|by)\s*(\d+)\s*(?:mm)?/i);
    if (boundingMatch && !data.plotDimensions) {
        const width = parseFloat(boundingMatch[1]);
        const length = parseFloat(boundingMatch[2]);
        data.plotDimensions = { width, length };
        data.totalAreaSqm = (width * length) / 1000000;
        data.plotShape = 'irregular';
        foundSomething = true;
    }

    // Extract area (sq ft or sqm) - enhanced patterns
    const sqftPatterns = [
        /(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:sqft|square\s*feet|sq\s*ft)/i,
        /(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:square\s*foot|sft)/i,
        /area\s*(?:is|of)?\s*(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:ft|feet)?/i
    ];
    
    for (const pattern of sqftPatterns) {
        const match = message.match(pattern);
        if (match) {
            const sqft = parseFloat(match[1].replace(/,/g, ''));
            data.totalAreaSqm = Math.round(sqft * 0.0929 * 10) / 10;
            foundSomething = true;
            break;
        }
    }
    
    // sqm patterns
    const sqmPatterns = [
        /(\d+(?:\.\d+)?)\s*(?:sqm|sq\s*m|m2|square\s*m)/i,
        /(\d+(?:\.\d+)?)\s*(?:meter|metre)/i
    ];
    
    for (const pattern of sqmPatterns) {
        const match = message.match(pattern);
        if (match && !data.totalAreaSqm) {
            data.totalAreaSqm = parseFloat(match[1]);
            foundSomething = true;
            break;
        }
    }

    // Extract Plot Shape
    if (/(l[- ]?shape|corner\s*plot|l\s*type)/i.test(message)) {
        data.plotShape = 'L-shaped';
        foundSomething = true;
    } else if (/(irregular|odd\s*shape|not\s*rectangle|curved)/i.test(message)) {
        data.plotShape = 'irregular';
        foundSomething = true;
    } else if (/(wide|horizontal)/i.test(message)) {
        data.plotShape = 'wide';
        foundSomething = true;
    } else if (/(narrow|vertical)/i.test(message)) {
        data.plotShape = 'narrow';
        foundSomething = true;
    }

    // Extract BHK (1bhk, 2 bhk, 3-bhk, etc.)
    const bhkMatch = message.match(/(\d)\s*[-]?\s*bhk/i);
    if (bhkMatch) {
        const numBedrooms = parseInt(bhkMatch[1]);
        if (!data.rooms) data.rooms = [];

        // Add bedrooms
        const existingBedrooms = data.rooms.filter(r => r.type === 'bedroom' || r.type === 'master_bedroom');
        if (existingBedrooms.length === 0) {
            data.rooms.push({ type: 'master_bedroom', quantity: 1, minAreaSqm: 15 });
            if (numBedrooms > 1) {
                data.rooms.push({ type: 'bedroom', quantity: numBedrooms - 1, minAreaSqm: 12 });
            }
        }

        // Add living room and kitchen (part of BHK)
        if (!data.rooms.find(r => r.type === 'living_room')) {
            data.rooms.push({ type: 'living_room', quantity: 1, minAreaSqm: 18 });
        }
        if (!data.rooms.find(r => r.type === 'kitchen')) {
            data.rooms.push({ type: 'kitchen', quantity: 1, minAreaSqm: 9 });
        }
        // Add default bathrooms for BHK
        if (!data.rooms.find(r => r.type === 'bathroom')) {
            data.rooms.push({ type: 'bathroom', quantity: Math.max(1, numBedrooms - 1), minAreaSqm: 4 });
        }

        foundSomething = true;
    }

    // Extract individual room mentions - enhanced patterns with typo tolerance
    const roomPatterns = [
        { pattern: /(\d+)\s*(?:bed\s*rooms?|bedrooms?|br|beds?)/i, type: 'bedroom' },
        { pattern: /(\d+)\s*(?:bath\s*rooms?|bathrooms?|toilets?|wc|washrooms?)/i, type: 'bathroom' },
        { pattern: /living\s*room|hall|drawing\s*room|sitting\s*room|lounge/i, type: 'living_room', qty: 1 },
        { pattern: /kitchen|cooking\s*area|modular/i, type: 'kitchen', qty: 1 },
        { pattern: /dining|eating\s*area/i, type: 'dining_room', qty: 1 },
        { pattern: /balcony|verandah|sit\s*out|terrace/i, type: 'balcony', qty: 1 },
        { pattern: /parking|garage|car\s*space/i, type: 'parking', qty: 1 },
        { pattern: /study|office|work\s*room|home\s*office/i, type: 'study', qty: 1 },
        { pattern: /pooja|puja|mandir|temple|prayer/i, type: 'pooja_room', qty: 1 },
        { pattern: /shop|store|commercial|dukan|showroom/i, type: 'shop', qty: 1 },
        { pattern: /utility|wash\s*area|laundry/i, type: 'utility', qty: 1 },
        { pattern: /storage|store\s*room|godown/i, type: 'storage', qty: 1 },
        { pattern: /stair|staircase|steps/i, type: 'staircase', qty: 1 }
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
            } else if (!qty && match[1]) {
                // Update quantity if explicitly specified
                existing.quantity = parseInt(match[1]);
            }
        }
    }

    // Extract position preferences
    const positionPatterns = [
        { pattern: /kitchen\s*(?:in|on|at)?\s*(?:the)?\s*(north|south|east|west|corner)/i, room: 'kitchen' },
        { pattern: /bedroom\s*(?:in|on|at)?\s*(?:the)?\s*(north|south|east|west|back|front)/i, room: 'bedroom' },
        { pattern: /living\s*(?:in|on|at)?\s*(?:the)?\s*(front|entrance|main)/i, room: 'living_room' }
    ];

    for (const { pattern, room } of positionPatterns) {
        const match = message.match(pattern);
        if (match) {
            const roomObj = data.rooms?.find(r => r.type === room);
            if (roomObj) {
                roomObj.position = match[1].toLowerCase();
            }
        }
    }

    // Generate helpful response
    if (foundSomething) {
        if (hasCompleteRequirements(data)) {
            const areaFt = Math.round(data.totalAreaSqm / 0.0929);
            const roomList = data.rooms.map(r => `${r.quantity}x ${r.type.replace(/_/g, ' ')}`).join(', ');
            response = `Got it! ${areaFt} sq ft with ${roomList}.\n\nReady to generate your floor plan. Say "generate" or "create" to proceed.`;
        } else if (data.totalAreaSqm && (!data.rooms || data.rooms.length === 0)) {
            response = `Noted: ${Math.round(data.totalAreaSqm / 0.0929)} sq ft.\n\nWhat rooms do you need? (e.g., "3BHK with 2 bathrooms" or "2 bedrooms, kitchen, living room")`;
        } else if (data.rooms && data.rooms.length > 0 && !data.totalAreaSqm) {
            const roomList = data.rooms.map(r => `${r.quantity}x ${r.type.replace(/_/g, ' ')}`).join(', ');
            response = `Got the rooms: ${roomList}.\n\nWhat's the total area? (e.g., "1200 sq ft" or "100 sqm" or "30x40 plot")`;
        } else {
            response = "I understood some of your requirements. Please also tell me the area and rooms you need.";
        }
    } else {
        // Try to give helpful suggestions based on what they might have meant
        response = "I'm trying to understand your requirements. Please tell me:\n\n" +
            "1. Plot size: '30x40 feet' or '1200 sq ft' or '100 sqm'\n" +
            "2. Rooms needed: '3BHK' or '2 bedrooms, 1 kitchen, 1 bathroom'\n\n" +
            "Example: 'I want 1200 sqft house with 3 bedrooms and 2 bathrooms'";
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
            plotShape: data.plotShape || 'rectangular',
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

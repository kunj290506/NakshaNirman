/**
 * Geometry Agent
 * Uses Real-World Reference Templates to generate floor plans
 */

import {
    WALL_THICKNESS,
    MIN_DOOR_WIDTH
} from '../utils/constants.js';
import {
    getSharedEdge
} from '../utils/geometry.js';
import { FLOOR_PLAN_TEMPLATES } from '../data/floorPlanTemplates.js';

/**
 * Geometry Agent
 * Uses Real-World Reference Templates to generate floor plans
 */
export function generateLayout(planData) {
    const { plot, rooms } = planData;

    const bounds = {
        x: WALL_THICKNESS,
        y: WALL_THICKNESS,
        width: plot.usableWidth,
        height: plot.usableHeight
    };

    // 1. Analyze Requirements to find best Template
    const bedroomCount = rooms.filter(r => r.type === 'bedroom' || r.type === 'master_bedroom').length;

    let matchingTemplate = null;

    if (bedroomCount >= 3) {
        matchingTemplate = FLOOR_PLAN_TEMPLATES.find(t => t.id === '3bhk_modern_luxury');
    } else {
        matchingTemplate = FLOOR_PLAN_TEMPLATES.find(t => t.id === '2bhk_standard_type_a');
    }

    // Fallback if no specific template found (though we covered 2 and 3 BHK)
    if (!matchingTemplate) matchingTemplate = FLOOR_PLAN_TEMPLATES[0];

    // 2. Adapt Template to Plot
    // We map the normalized coordinates (0-1) to valid bounds

    const placedRooms = [];
    const templateSlots = [...matchingTemplate.structure];

    // We need to map "User Requested Rooms" to "Template Slots"
    // Heuristic: Match by type exactly first, then fill empty slots

    const requests = [...rooms];

    templateSlots.forEach(slot => {
        // Find a matching request
        const matchIndex = requests.findIndex(r => r.type === slot.type);

        let roomData = null;

        if (matchIndex !== -1) {
            // Found exact match (e.g., User wants Kitchen, Template has Kitchen slot)
            roomData = requests[matchIndex];
            requests.splice(matchIndex, 1); // Remove from requests
        } else {
            // Template has a slot (e.g. Utility) but user didn't ask for it
            // We keep it as a "Bonus Room" or merge it?
            // Let's keep it to maintain the "Real Layout" shape
            roomData = { type: slot.type, label: slot.type.replace('_', ' ').toUpperCase() };
        }

        // Scale Coordinate
        const x = bounds.x + (slot.x * bounds.width);
        const y = bounds.y + (slot.y * bounds.height);
        const w = slot.w * bounds.width;
        const h = slot.h * bounds.height;

        placedRooms.push({
            ...roomData,
            x: x,
            y: y,
            width: w,
            height: h,
            // Keep original properties if it was a real request, else defaults
            ...roomData
        });
    });

    // Handle "Leftover" requests (e.g. Parking, Puja) that weren't in template
    // Ideally we should have a "Flex Zone" in templates, but for now we might skip or warn
    // OR just place them in the 'Setback' area or override a generic slot

    // 3. Generate Doors (Logic remains similar: verify adjacency)
    const corridor = placedRooms.find(r => r.type === 'corridor');

    const roomsWithDoors = placedRooms.map((room, i) => {
        if (room.type === 'corridor') return { ...room, door: null }; // No door for the corridor itself usually

        const target = corridor || placedRooms.find(r => r.type === 'living_room'); // Default to living
        const door = generateDoorPosition(room, placedRooms, target, bounds); // Reuse existing door logic or simple edge check
        return { ...room, door };
    });

    return {
        success: true,
        data: {
            status: 'success',
            boundary: { width: plot.width, height: plot.height },
            rooms: roomsWithDoors
        }
    };
}

/**
 * Generate door position (Reused/Simplified)
 */
function generateDoorPosition(room, allRooms, target, bounds) {
    const doorWidth = MIN_DOOR_WIDTH;

    // Try to find shared edge with Target (Hall/Corridor)
    if (target && target !== room) {
        const shared = getSharedEdge(room, target);
        if (shared && shared.length > doorWidth) {
            if (shared.direction === 'vertical') {
                return {
                    direction: shared.x === room.x ? 'west' : 'east',
                    x: shared.x,
                    y: (shared.y1 + shared.y2) / 2 - doorWidth / 2,
                    width: doorWidth
                };
            } else {
                return {
                    direction: shared.y === room.y ? 'south' : 'north',
                    x: (shared.x1 + shared.x2) / 2 - doorWidth / 2,
                    y: shared.y,
                    width: doorWidth
                };
            }
        }
    }

    // Default: internal edge check logic or just place on "inner" side
    // Simplified: Place door on the side closest to center of house
    const cx = bounds.x + bounds.width / 2;
    const cy = bounds.y + bounds.height / 2;
    const rcx = room.x + room.width / 2;
    const rcy = room.y + room.height / 2;

    if (Math.abs(cx - rcx) > Math.abs(cy - rcy)) {
        // More horizontal offset -> Vertical Door (East/West)
        return {
            direction: rcx < cx ? 'east' : 'west',
            x: rcx < cx ? room.x + room.width : room.x,
            y: room.y + room.height / 2 - doorWidth / 2,
            width: doorWidth
        };
    } else {
        // Vertical offset -> Horizontal Door (North/South)
        return {
            direction: rcy < cy ? 'south' : 'north',
            x: room.x + room.width / 2 - doorWidth / 2,
            y: rcy < cy ? room.y + room.height : room.y,
            width: doorWidth
        };
    }
}

/**
 * Group rooms by category
 */
function groupRooms(rooms) {
    const groups = {
        public: [],      // Living, Dining
        service: [],     // Kitchen, Utility, Parking
        private: [],     // Bedrooms
        wet: [],         // Bathrooms
        other: []        // Balcony, Study, etc.
    };

    rooms.forEach(room => {
        switch (room.type) {
            case 'living_room':
            case 'dining_room':
                groups.public.push(room);
                break;
            case 'kitchen':
            case 'utility':
            case 'parking':
                groups.service.push(room);
                break;
            case 'bedroom':
            case 'master_bedroom':
                groups.private.push(room);
                break;
            case 'bathroom':
                groups.wet.push(room);
                break;
            default:
                groups.other.push(room);
        }
    });

    return groups;
}

/**
 * Create grid layout with central corridor
 */
function createGridLayout(groups, bounds) {
    const placed = [];
    const allRooms = [
        ...groups.public,
        ...groups.service,
        ...groups.private,
        ...groups.wet,
        ...groups.other
    ];

    // Calculate total area needed
    const totalRoomArea = allRooms.reduce((sum, r) =>
        sum + (r.suggestedWidth * r.suggestedHeight), 0);
    const availableArea = bounds.width * bounds.height;

    // Determine layout type based on aspect ratio
    const isWide = bounds.width > bounds.height;

    // Create corridor in center
    let corridor;
    let leftZone, rightZone, topZone, bottomZone;

    if (isWide) {
        // Horizontal layout: corridor runs vertically in middle
        const corridorX = bounds.x + (bounds.width - CORRIDOR_WIDTH) / 2;
        corridor = {
            type: 'corridor',
            label: 'Lobby',
            x: corridorX,
            y: bounds.y,
            width: CORRIDOR_WIDTH,
            height: bounds.height,
            color: '#f0f0f0'
        };

        leftZone = { x: bounds.x, y: bounds.y, width: corridorX - bounds.x, height: bounds.height };
        rightZone = { x: corridorX + CORRIDOR_WIDTH, y: bounds.y, width: bounds.x + bounds.width - corridorX - CORRIDOR_WIDTH, height: bounds.height };
    } else {
        // Vertical layout: corridor runs horizontally in middle
        const corridorY = bounds.y + (bounds.height - CORRIDOR_WIDTH) / 2;
        corridor = {
            type: 'corridor',
            label: 'Lobby',
            x: bounds.x,
            y: corridorY,
            width: bounds.width,
            height: CORRIDOR_WIDTH,
            color: '#f0f0f0'
        };

        topZone = { x: bounds.x, y: bounds.y, width: bounds.width, height: corridorY - bounds.y };
        bottomZone = { x: bounds.x, y: corridorY + CORRIDOR_WIDTH, width: bounds.width, height: bounds.y + bounds.height - corridorY - CORRIDOR_WIDTH };
    }

    // Place rooms in zones
    if (isWide) {
        // Left side: Public + Service rooms
        const leftRooms = [...groups.public, ...groups.service, ...groups.other];
        placeRoomsInZone(leftRooms, leftZone, placed);

        // Right side: Private + Wet rooms
        const rightRooms = [...groups.private, ...groups.wet];
        placeRoomsInZone(rightRooms, rightZone, placed);
    } else {
        // Top: Public + Service rooms (front of house)
        const topRooms = [...groups.public, ...groups.service];
        placeRoomsInZone(topRooms, topZone, placed);

        // Bottom: Private + Wet + Other (back of house)
        const bottomRooms = [...groups.private, ...groups.wet, ...groups.other];
        placeRoomsInZone(bottomRooms, bottomZone, placed);
    }

    if (placed.length < allRooms.length) {
        // Some rooms couldn't be placed, try without corridor
        return createSimpleGridLayout(allRooms, bounds);
    }

    return { success: true, rooms: placed, corridor };
}

/**
 * Place rooms in a zone using grid
 */
function placeRoomsInZone(rooms, zone, placed) {
    if (rooms.length === 0) return;

    // Calculate scale factor to fit rooms in zone
    const totalArea = rooms.reduce((sum, r) => sum + r.suggestedWidth * r.suggestedHeight, 0);
    const zoneArea = zone.width * zone.height;
    const scale = Math.sqrt(zoneArea * 0.95 / totalArea);

    // Sort by size (largest first)
    const sortedRooms = [...rooms].sort((a, b) =>
        (b.suggestedWidth * b.suggestedHeight) - (a.suggestedWidth * a.suggestedHeight)
    );

    // Calculate number of columns based on zone shape
    const avgRoomWidth = sortedRooms.reduce((sum, r) => sum + r.suggestedWidth * scale, 0) / rooms.length;
    const cols = Math.max(1, Math.round(zone.width / avgRoomWidth));
    const colWidth = zone.width / cols;

    let currentCol = 0;
    let currentY = zone.y;
    let colHeights = new Array(cols).fill(zone.y);

    for (const room of sortedRooms) {
        // Scale room to fit
        let w = Math.floor(room.suggestedWidth * scale);
        let h = Math.floor(room.suggestedHeight * scale);

        // Ensure minimum size
        w = Math.max(2000, Math.min(w, zone.width));
        h = Math.max(1500, Math.min(h, zone.height / 2));

        // Find column with most space
        let bestCol = 0;
        let minHeight = colHeights[0];
        for (let c = 0; c < cols; c++) {
            if (colHeights[c] < minHeight) {
                minHeight = colHeights[c];
                bestCol = c;
            }
        }

        // Check if room spans multiple columns
        let roomCols = Math.ceil(w / colWidth);
        roomCols = Math.min(roomCols, cols - bestCol);

        const actualWidth = roomCols * colWidth;
        const x = zone.x + bestCol * colWidth;
        const y = colHeights[bestCol];

        // Check if fits in zone
        if (y + h <= zone.y + zone.height) {
            placed.push({
                ...room,
                x: x,
                y: y,
                width: actualWidth,
                height: h
            });

            // Update column heights
            for (let c = bestCol; c < bestCol + roomCols; c++) {
                colHeights[c] = y + h;
            }
        }
    }
}

/**
 * Simple grid layout without corridor (fallback)
 */
function createSimpleGridLayout(rooms, bounds) {
    const placed = [];

    // Calculate grid dimensions
    const n = rooms.length;
    const cols = Math.ceil(Math.sqrt(n * bounds.width / bounds.height));
    const rows = Math.ceil(n / cols);

    const cellWidth = bounds.width / cols;
    const cellHeight = bounds.height / rows;

    // Sort rooms by priority
    const sortedRooms = [...rooms].sort((a, b) => (a.priority || 5) - (b.priority || 5));

    let i = 0;
    for (let row = 0; row < rows && i < sortedRooms.length; row++) {
        for (let col = 0; col < cols && i < sortedRooms.length; col++) {
            const room = sortedRooms[i];

            placed.push({
                ...room,
                x: bounds.x + col * cellWidth,
                y: bounds.y + row * cellHeight,
                width: cellWidth,
                height: cellHeight
            });

            i++;
        }
    }

    return { success: true, rooms: placed, corridor: null };
}

/**
 * Generate door position (opening to corridor or adjacent room)
 */
function generateDoorPosition(room, allRooms, corridor, bounds) {
    const doorWidth = MIN_DOOR_WIDTH;

    // If corridor exists, door opens to corridor
    if (corridor) {
        // Check which side of room faces corridor
        if (room.x + room.width >= corridor.x && room.x + room.width <= corridor.x + corridor.width / 2) {
            // Room is to the left of corridor
            return {
                direction: 'east',
                x: room.x + room.width,
                y: room.y + (room.height - doorWidth) / 2,
                width: doorWidth
            };
        } else if (room.x >= corridor.x + corridor.width / 2 && room.x <= corridor.x + corridor.width) {
            // Room is to the right of corridor  
            return {
                direction: 'west',
                x: room.x,
                y: room.y + (room.height - doorWidth) / 2,
                width: doorWidth
            };
        } else if (room.y + room.height >= corridor.y && room.y + room.height <= corridor.y + corridor.height / 2) {
            // Room is above corridor
            return {
                direction: 'south',
                x: room.x + (room.width - doorWidth) / 2,
                y: room.y + room.height,
                width: doorWidth
            };
        } else if (room.y >= corridor.y + corridor.height / 2 && room.y <= corridor.y + corridor.height) {
            // Room is below corridor
            return {
                direction: 'north',
                x: room.x + (room.width - doorWidth) / 2,
                y: room.y,
                width: doorWidth
            };
        }
    }

    // Default: find shared edge with another room
    for (const other of allRooms) {
        if (other === room) continue;

        const shared = getSharedEdge(room, other);
        if (shared && shared.length >= doorWidth) {
            if (shared.direction === 'vertical') {
                return {
                    direction: shared.x === room.x ? 'west' : 'east',
                    x: shared.x,
                    y: (shared.y1 + shared.y2) / 2 - doorWidth / 2,
                    width: doorWidth
                };
            } else {
                return {
                    direction: shared.y === room.y ? 'south' : 'north',
                    x: (shared.x1 + shared.x2) / 2 - doorWidth / 2,
                    y: shared.y,
                    width: doorWidth
                };
            }
        }
    }

    // Default door on bottom edge
    return {
        direction: 'south',
        x: room.x + (room.width - doorWidth) / 2,
        y: room.y,
        width: doorWidth
    };
}

export function createErrorResponse(reason) {
    return {
        status: 'error',
        error_reason: reason,
        floor_plan: null
    };
}

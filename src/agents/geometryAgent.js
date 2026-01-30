/**
 * Geometry Agent
 * Intelligent room placement with architectural adjacency rules
 * - Bathroom adjacent to bedrooms
 * - Kitchen adjacent to dining
 * - Living room at front
 * - Private rooms (bedrooms) at back
 */

import {
    WALL_THICKNESS,
    MIN_DOOR_WIDTH,
    MAX_ASPECT_RATIO
} from '../utils/constants.js';
import {
    rectanglesOverlap,
    getAspectRatio,
    getSharedEdge
} from '../utils/geometry.js';

// Room adjacency rules - which rooms should be near each other
const ADJACENCY_RULES = {
    'master_bedroom': ['bathroom'],
    'bedroom': ['bathroom'],
    'kitchen': ['dining_room', 'utility'],
    'dining_room': ['kitchen', 'living_room'],
    'living_room': ['dining_room', 'balcony'],
    'bathroom': ['bedroom', 'master_bedroom'],
    'parking': [],
    'balcony': ['living_room', 'bedroom'],
    'study': ['bedroom'],
    'pooja_room': ['living_room', 'kitchen']
};

// Room zones - front (public) to back (private)
const ROOM_ZONES = {
    'parking': 0,      // Front - entrance area
    'living_room': 1,  // Front - public
    'dining_room': 1,  // Front - public
    'kitchen': 2,      // Middle
    'utility': 2,      // Middle
    'pooja_room': 2,   // Middle
    'study': 2,        // Middle
    'bathroom': 3,     // Back area
    'bedroom': 3,      // Back - private
    'master_bedroom': 4, // Far back - most private
    'balcony': 2       // Side/back
};

/**
 * Generate room layout with intelligent placement
 */
export function generateLayout(planData) {
    const { plot, rooms } = planData;

    const bounds = {
        x: WALL_THICKNESS,
        y: WALL_THICKNESS,
        width: plot.usableWidth,
        height: plot.usableHeight
    };

    // Sort rooms by zone (front to back placement)
    const sortedRooms = [...rooms].sort((a, b) => {
        const zoneA = ROOM_ZONES[a.type] ?? 3;
        const zoneB = ROOM_ZONES[b.type] ?? 3;
        if (zoneA !== zoneB) return zoneA - zoneB;
        // Within same zone, larger rooms first
        return (b.suggestedWidth * b.suggestedHeight) - (a.suggestedWidth * a.suggestedHeight);
    });

    // Scale rooms if needed to fit
    const scaledRooms = scaleRoomsToFit(sortedRooms, bounds);

    // Place rooms with adjacency awareness
    const result = placeRoomsLogically(scaledRooms, bounds);

    if (!result.success) {
        return {
            success: false,
            error: result.error || 'Unable to fit all rooms'
        };
    }

    // Generate doors
    const roomsWithDoors = result.rooms.map((room, i) => {
        const door = generateDoorPosition(room, result.rooms, i, bounds);
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
 * Scale rooms proportionally to fit available space
 */
function scaleRoomsToFit(rooms, bounds) {
    const totalArea = rooms.reduce((sum, r) => sum + r.suggestedWidth * r.suggestedHeight, 0);
    const availableArea = bounds.width * bounds.height;

    if (totalArea <= availableArea * 0.8) {
        return rooms;
    }

    const scale = Math.sqrt((availableArea * 0.7) / totalArea);
    return rooms.map(r => ({
        ...r,
        suggestedWidth: Math.max(2000, Math.floor(r.suggestedWidth * scale)),
        suggestedHeight: Math.max(2000, Math.floor(r.suggestedHeight * scale))
    }));
}

/**
 * Place rooms with logical adjacency
 */
function placeRoomsLogically(rooms, bounds) {
    const placed = [];
    const grid = createGrid(bounds);

    // Divide bounds into zones (front, middle, back)
    const zoneHeight = bounds.height / 3;
    const zones = [
        { y: bounds.y, h: zoneHeight },                    // Front
        { y: bounds.y + zoneHeight, h: zoneHeight },       // Middle
        { y: bounds.y + zoneHeight * 2, h: zoneHeight }    // Back
    ];

    for (const room of rooms) {
        let roomPlaced = false;
        const zone = ROOM_ZONES[room.type] ?? 2;

        // Determine target Y range based on zone
        let targetStartY, targetEndY;
        if (zone <= 1) {
            // Front zone
            targetStartY = bounds.y;
            targetEndY = bounds.y + bounds.height * 0.4;
        } else if (zone === 2) {
            // Middle zone
            targetStartY = bounds.y + bounds.height * 0.3;
            targetEndY = bounds.y + bounds.height * 0.7;
        } else {
            // Back zone (private)
            targetStartY = bounds.y + bounds.height * 0.5;
            targetEndY = bounds.y + bounds.height;
        }

        // Try to place adjacent to required rooms
        const adjacentTo = ADJACENCY_RULES[room.type] || [];
        for (const adjType of adjacentTo) {
            if (roomPlaced) break;

            const adjRoom = placed.find(p => p.type === adjType);
            if (adjRoom) {
                const pos = findAdjacentPosition(room, adjRoom, bounds, placed);
                if (pos) {
                    placed.push({
                        ...room,
                        x: pos.x,
                        y: pos.y,
                        width: pos.width,
                        height: pos.height
                    });
                    roomPlaced = true;
                }
            }
        }

        // If not placed by adjacency, place in target zone
        if (!roomPlaced) {
            const pos = findPositionInZone(room, bounds, placed, targetStartY, targetEndY);
            if (pos) {
                placed.push({
                    ...room,
                    x: pos.x,
                    y: pos.y,
                    width: pos.width,
                    height: pos.height
                });
                roomPlaced = true;
            }
        }

        // Fallback: find any available space
        if (!roomPlaced) {
            const pos = findAnyPosition(room, bounds, placed);
            if (pos) {
                placed.push({
                    ...room,
                    x: pos.x,
                    y: pos.y,
                    width: pos.width,
                    height: pos.height
                });
                roomPlaced = true;
            }
        }

        if (!roomPlaced) {
            return {
                success: false,
                error: `Cannot place ${room.label} - no suitable space found`
            };
        }
    }

    return { success: true, rooms: placed };
}

/**
 * Find position adjacent to another room
 */
function findAdjacentPosition(room, adjRoom, bounds, placed) {
    const w = room.suggestedWidth;
    const h = room.suggestedHeight;
    const gap = WALL_THICKNESS;

    // Try positions around the adjacent room
    const positions = [
        // Right of adjacent room
        { x: adjRoom.x + adjRoom.width + gap, y: adjRoom.y, w, h },
        // Left of adjacent room
        { x: adjRoom.x - w - gap, y: adjRoom.y, w, h },
        // Below adjacent room
        { x: adjRoom.x, y: adjRoom.y + adjRoom.height + gap, w, h },
        // Above adjacent room
        { x: adjRoom.x, y: adjRoom.y - h - gap, w, h },
    ];

    // Also try rotated dimensions
    if (getAspectRatio(h, w) <= MAX_ASPECT_RATIO) {
        positions.push(
            { x: adjRoom.x + adjRoom.width + gap, y: adjRoom.y, w: h, h: w },
            { x: adjRoom.x - h - gap, y: adjRoom.y, w: h, h: w },
            { x: adjRoom.x, y: adjRoom.y + adjRoom.height + gap, w: h, h: w },
            { x: adjRoom.x, y: adjRoom.y - w - gap, w: h, h: w }
        );
    }

    for (const pos of positions) {
        if (isValidPosition(pos.x, pos.y, pos.w, pos.h, bounds, placed)) {
            return { x: pos.x, y: pos.y, width: pos.w, height: pos.h };
        }
    }

    return null;
}

/**
 * Find position within a Y zone
 */
function findPositionInZone(room, bounds, placed, minY, maxY) {
    const step = 250;
    const orientations = [
        { w: room.suggestedWidth, h: room.suggestedHeight },
        { w: room.suggestedHeight, h: room.suggestedWidth }
    ];

    for (const orient of orientations) {
        if (getAspectRatio(orient.w, orient.h) > MAX_ASPECT_RATIO) continue;

        // Search within zone
        for (let y = minY; y + orient.h <= maxY && y + orient.h <= bounds.y + bounds.height; y += step) {
            for (let x = bounds.x; x + orient.w <= bounds.x + bounds.width; x += step) {
                if (isValidPosition(x, y, orient.w, orient.h, bounds, placed)) {
                    return { x, y, width: orient.w, height: orient.h };
                }
            }
        }
    }

    return null;
}

/**
 * Find any available position (fallback)
 */
function findAnyPosition(room, bounds, placed) {
    const step = 250;
    const scales = [1, 0.9, 0.8, 0.7, 0.6];

    for (const scale of scales) {
        const w = Math.max(2000, Math.floor(room.suggestedWidth * scale));
        const h = Math.max(2000, Math.floor(room.suggestedHeight * scale));

        const orientations = [
            { w, h },
            { w: h, h: w }
        ];

        for (const orient of orientations) {
            if (getAspectRatio(orient.w, orient.h) > MAX_ASPECT_RATIO) continue;

            for (let y = bounds.y; y + orient.h <= bounds.y + bounds.height; y += step) {
                for (let x = bounds.x; x + orient.w <= bounds.x + bounds.width; x += step) {
                    if (isValidPosition(x, y, orient.w, orient.h, bounds, placed)) {
                        return { x, y, width: orient.w, height: orient.h };
                    }
                }
            }
        }
    }

    return null;
}

/**
 * Check if position is valid (within bounds and no overlap)
 */
function isValidPosition(x, y, w, h, bounds, placed) {
    // Check bounds
    if (x < bounds.x || y < bounds.y) return false;
    if (x + w > bounds.x + bounds.width) return false;
    if (y + h > bounds.y + bounds.height) return false;

    // Check overlap with placed rooms
    const rect = { x, y, width: w, height: h };
    for (const p of placed) {
        if (rectanglesOverlap(rect, p)) return false;
    }

    return true;
}

/**
 * Create occupancy grid
 */
function createGrid(bounds) {
    const cellSize = 250;
    const cols = Math.ceil(bounds.width / cellSize);
    const rows = Math.ceil(bounds.height / cellSize);
    return { cellSize, cols, rows };
}

/**
 * Generate door position
 */
function generateDoorPosition(room, allRooms, roomIndex, bounds) {
    const doorWidth = MIN_DOOR_WIDTH;

    // Find shared edges with other rooms for internal doors
    for (let i = 0; i < allRooms.length; i++) {
        if (i === roomIndex) continue;

        const other = allRooms[i];
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

    // Default door positions based on room location
    const edges = [];

    if (room.y > bounds.y + 100) {
        edges.push({ dir: 'south', x: room.x + (room.width - doorWidth) / 2, y: room.y });
    }
    if (room.y + room.height < bounds.y + bounds.height - 100) {
        edges.push({ dir: 'north', x: room.x + (room.width - doorWidth) / 2, y: room.y + room.height });
    }
    if (room.x > bounds.x + 100) {
        edges.push({ dir: 'west', x: room.x, y: room.y + (room.height - doorWidth) / 2 });
    }
    if (room.x + room.width < bounds.x + bounds.width - 100) {
        edges.push({ dir: 'east', x: room.x + room.width, y: room.y + (room.height - doorWidth) / 2 });
    }

    const edge = edges[0] || { dir: 'south', x: room.x + (room.width - doorWidth) / 2, y: room.y };
    return { direction: edge.dir, x: edge.x, y: edge.y, width: doorWidth };
}

export function createErrorResponse(reason) {
    return {
        status: 'error',
        error_reason: reason,
        floor_plan: null
    };
}

/**
 * Geometry Agent
 * Generates room coordinates using bin-packing algorithm
 */

import {
    WALL_THICKNESS,
    MIN_DOOR_WIDTH,
    MAX_ASPECT_RATIO
} from '../utils/constants.js';
import {
    rectanglesOverlap,
    isWithinBounds,
    getAspectRatio,
    getSharedEdge
} from '../utils/geometry.js';

/**
 * Generate room layout coordinates
 * @param {Object} planData - Data from Planning Agent
 * @returns {Object} {success: boolean, data?: Object, error?: string}
 */
export function generateLayout(planData) {
    const { plot, rooms } = planData;

    // Create usable area bounds (inside perimeter walls)
    const bounds = {
        x: WALL_THICKNESS,
        y: WALL_THICKNESS,
        width: plot.usableWidth,
        height: plot.usableHeight
    };

    // Use a simple row-based bin packing algorithm
    const placedRooms = [];
    const result = packRooms(rooms, bounds, placedRooms);

    if (!result.success) {
        return {
            success: false,
            error: result.error || 'Unable to fit all rooms in the available space'
        };
    }

    // Generate door positions for each room
    const roomsWithDoors = result.rooms.map((room, index) => {
        const door = generateDoorPosition(room, result.rooms, index, bounds);
        return { ...room, door };
    });

    return {
        success: true,
        data: {
            status: 'success',
            boundary: {
                width: plot.width,
                height: plot.height
            },
            rooms: roomsWithDoors
        }
    };
}

/**
 * Pack rooms into available space using row-based algorithm
 * @param {Array} rooms - Rooms to place (sorted by priority)
 * @param {Object} bounds - Available area bounds
 * @param {Array} placedRooms - Already placed rooms
 * @returns {Object} {success: boolean, rooms?: Array, error?: string}
 */
function packRooms(rooms, bounds, placedRooms = []) {
    const placed = [...placedRooms];

    // Track available rows
    let currentRowY = bounds.y;
    let currentRowX = bounds.x;
    let currentRowHeight = 0;

    for (const room of rooms) {
        let roomPlaced = false;

        // Try current position in row
        const placement = {
            ...room,
            x: currentRowX,
            y: currentRowY,
            width: room.suggestedWidth,
            height: room.suggestedHeight
        };

        // Check if room fits in current row
        if (currentRowX + room.suggestedWidth <= bounds.x + bounds.width &&
            currentRowY + room.suggestedHeight <= bounds.y + bounds.height) {

            // Check for overlaps
            let hasOverlap = false;
            for (const p of placed) {
                if (rectanglesOverlap(placement, p)) {
                    hasOverlap = true;
                    break;
                }
            }

            if (!hasOverlap) {
                placed.push(placement);
                currentRowX += room.suggestedWidth + WALL_THICKNESS;
                currentRowHeight = Math.max(currentRowHeight, room.suggestedHeight);
                roomPlaced = true;
            }
        }

        // If not placed, try new row
        if (!roomPlaced) {
            currentRowY += currentRowHeight + WALL_THICKNESS;
            currentRowX = bounds.x;
            currentRowHeight = room.suggestedHeight;

            const newPlacement = {
                ...room,
                x: currentRowX,
                y: currentRowY,
                width: room.suggestedWidth,
                height: room.suggestedHeight
            };

            // Check if new row position is valid
            if (newPlacement.y + newPlacement.height <= bounds.y + bounds.height) {
                placed.push(newPlacement);
                currentRowX += room.suggestedWidth + WALL_THICKNESS;
                roomPlaced = true;
            }
        }

        // If still not placed, try alternative dimensions (rotate)
        if (!roomPlaced) {
            const rotatedPlacement = tryRotatedPlacement(room, bounds, placed, currentRowY);
            if (rotatedPlacement) {
                placed.push(rotatedPlacement);
                roomPlaced = true;
            }
        }

        // If all attempts fail, try to find any free space
        if (!roomPlaced) {
            const freePlacement = findFreeSpace(room, bounds, placed);
            if (freePlacement) {
                placed.push(freePlacement);
                roomPlaced = true;
            }
        }

        if (!roomPlaced) {
            return {
                success: false,
                error: `Unable to place ${room.label} - insufficient space`
            };
        }
    }

    return {
        success: true,
        rooms: placed
    };
}

/**
 * Try placing room with rotated dimensions
 */
function tryRotatedPlacement(room, bounds, placed, startY) {
    const rotatedWidth = room.suggestedHeight;
    const rotatedHeight = room.suggestedWidth;

    // Check aspect ratio still valid after rotation
    if (getAspectRatio(rotatedWidth, rotatedHeight) > MAX_ASPECT_RATIO) {
        return null;
    }

    // Try at various positions
    for (let y = bounds.y; y + rotatedHeight <= bounds.y + bounds.height; y += 500) {
        for (let x = bounds.x; x + rotatedWidth <= bounds.x + bounds.width; x += 500) {
            const placement = {
                ...room,
                x,
                y,
                width: rotatedWidth,
                height: rotatedHeight
            };

            let hasOverlap = false;
            for (const p of placed) {
                if (rectanglesOverlap(placement, p)) {
                    hasOverlap = true;
                    break;
                }
            }

            if (!hasOverlap) {
                return placement;
            }
        }
    }

    return null;
}

/**
 * Find any free space for a room
 */
function findFreeSpace(room, bounds, placed) {
    const stepSize = 500; // 500mm steps

    // Try original dimensions
    for (let y = bounds.y; y + room.suggestedHeight <= bounds.y + bounds.height; y += stepSize) {
        for (let x = bounds.x; x + room.suggestedWidth <= bounds.x + bounds.width; x += stepSize) {
            const placement = {
                ...room,
                x,
                y,
                width: room.suggestedWidth,
                height: room.suggestedHeight
            };

            let hasOverlap = false;
            for (const p of placed) {
                if (rectanglesOverlap(placement, p)) {
                    hasOverlap = true;
                    break;
                }
            }

            if (!hasOverlap) {
                return placement;
            }
        }
    }

    return null;
}

/**
 * Generate door position for a room
 * @param {Object} room - Room with placement
 * @param {Array} allRooms - All rooms
 * @param {number} roomIndex - Current room index
 * @param {Object} bounds - Plot bounds
 * @returns {Object} Door position {x, y, width, direction}
 */
function generateDoorPosition(room, allRooms, roomIndex, bounds) {
    const doorWidth = MIN_DOOR_WIDTH;

    // Find potential door positions (edges not on outer walls)
    const edges = [];

    // Check each edge
    // Bottom edge
    if (room.y > bounds.y + 100) {
        edges.push({
            direction: 'south',
            x: room.x + (room.width - doorWidth) / 2,
            y: room.y,
            width: doorWidth
        });
    }

    // Top edge
    if (room.y + room.height < bounds.y + bounds.height - 100) {
        edges.push({
            direction: 'north',
            x: room.x + (room.width - doorWidth) / 2,
            y: room.y + room.height,
            width: doorWidth
        });
    }

    // Left edge
    if (room.x > bounds.x + 100) {
        edges.push({
            direction: 'west',
            x: room.x,
            y: room.y + (room.height - doorWidth) / 2,
            width: doorWidth
        });
    }

    // Right edge
    if (room.x + room.width < bounds.x + bounds.width - 100) {
        edges.push({
            direction: 'east',
            x: room.x + room.width,
            y: room.y + (room.height - doorWidth) / 2,
            width: doorWidth
        });
    }

    // Prefer edges that are adjacent to other rooms
    for (const edge of edges) {
        for (let i = 0; i < allRooms.length; i++) {
            if (i === roomIndex) continue;

            const other = allRooms[i];
            const shared = getSharedEdge(room, other);
            if (shared) {
                // Place door on shared edge
                if (shared.direction === 'vertical') {
                    return {
                        direction: edge.x < room.x + room.width / 2 ? 'west' : 'east',
                        x: shared.x,
                        y: (shared.y1 + shared.y2) / 2 - doorWidth / 2,
                        width: doorWidth
                    };
                } else {
                    return {
                        direction: edge.y < room.y + room.height / 2 ? 'south' : 'north',
                        x: (shared.x1 + shared.x2) / 2 - doorWidth / 2,
                        y: shared.y,
                        width: doorWidth
                    };
                }
            }
        }
    }

    // Default to first available edge
    return edges[0] || {
        direction: 'south',
        x: room.x + (room.width - doorWidth) / 2,
        y: room.y,
        width: doorWidth
    };
}

/**
 * Create error response
 * @param {string} reason - Error reason
 * @returns {Object} Error response JSON
 */
export function createErrorResponse(reason) {
    return {
        status: 'error',
        error_reason: reason,
        floor_plan: null
    };
}

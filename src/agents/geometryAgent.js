/**
 * Geometry Agent
 * Generates room coordinates using adaptive bin-packing algorithm
 * Automatically adjusts room sizes to fit available space
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

    // Calculate total requested area vs available
    const totalRequestedArea = rooms.reduce((sum, r) => sum + (r.suggestedWidth * r.suggestedHeight), 0);
    const availableArea = bounds.width * bounds.height;

    // If rooms need more space than available, scale them down
    let roomsToPlace = rooms;
    if (totalRequestedArea > availableArea * 0.85) {
        const scaleFactor = Math.sqrt((availableArea * 0.75) / totalRequestedArea);
        roomsToPlace = rooms.map(room => ({
            ...room,
            suggestedWidth: Math.max(Math.floor(room.suggestedWidth * scaleFactor), 2000),
            suggestedHeight: Math.max(Math.floor(room.suggestedHeight * scaleFactor), 2000)
        }));
    }

    // Use adaptive bin packing
    const result = adaptivePackRooms(roomsToPlace, bounds);

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
 * Adaptive room packing - tries multiple strategies
 */
function adaptivePackRooms(rooms, bounds) {
    // Strategy 1: Try with current sizes
    let result = gridPackRooms(rooms, bounds);
    if (result.success) return result;

    // Strategy 2: Reduce all rooms by 10%
    const scaled90 = scaleRooms(rooms, 0.9);
    result = gridPackRooms(scaled90, bounds);
    if (result.success) return result;

    // Strategy 3: Reduce all rooms by 20%
    const scaled80 = scaleRooms(rooms, 0.8);
    result = gridPackRooms(scaled80, bounds);
    if (result.success) return result;

    // Strategy 4: Reduce all rooms by 30%
    const scaled70 = scaleRooms(rooms, 0.7);
    result = gridPackRooms(scaled70, bounds);
    if (result.success) return result;

    // Strategy 5: Use minimum viable sizes
    const minSized = rooms.map(room => ({
        ...room,
        suggestedWidth: Math.max(2500, room.suggestedWidth * 0.6),
        suggestedHeight: Math.max(2000, room.suggestedHeight * 0.6)
    }));
    result = gridPackRooms(minSized, bounds);
    if (result.success) return result;

    // All strategies failed
    return { success: false, error: result.error };
}

/**
 * Scale rooms by a factor
 */
function scaleRooms(rooms, factor) {
    return rooms.map(room => ({
        ...room,
        suggestedWidth: Math.max(2000, Math.floor(room.suggestedWidth * factor)),
        suggestedHeight: Math.max(2000, Math.floor(room.suggestedHeight * factor))
    }));
}

/**
 * Grid-based room packing - more efficient than row-based
 */
function gridPackRooms(rooms, bounds) {
    const placed = [];
    const grid = createOccupancyGrid(bounds, 250); // 250mm grid cells

    for (const room of rooms) {
        let roomPlaced = false;

        // Try both orientations
        const orientations = [
            { w: room.suggestedWidth, h: room.suggestedHeight },
            { w: room.suggestedHeight, h: room.suggestedWidth }
        ];

        for (const orient of orientations) {
            if (roomPlaced) break;

            // Skip if aspect ratio is too extreme
            if (getAspectRatio(orient.w, orient.h) > MAX_ASPECT_RATIO) continue;

            // Find first free position
            const pos = findFreePosition(grid, bounds, orient.w, orient.h, placed);
            if (pos) {
                const placement = {
                    ...room,
                    x: pos.x,
                    y: pos.y,
                    width: orient.w,
                    height: orient.h
                };
                placed.push(placement);
                markOccupied(grid, placement, bounds);
                roomPlaced = true;
            }
        }

        // Try smaller sizes if still not placed
        if (!roomPlaced) {
            const smallerSizes = [0.8, 0.7, 0.6];
            for (const scale of smallerSizes) {
                if (roomPlaced) break;

                const w = Math.max(2000, Math.floor(room.suggestedWidth * scale));
                const h = Math.max(2000, Math.floor(room.suggestedHeight * scale));

                const pos = findFreePosition(grid, bounds, w, h, placed);
                if (pos) {
                    const placement = {
                        ...room,
                        x: pos.x,
                        y: pos.y,
                        width: w,
                        height: h
                    };
                    placed.push(placement);
                    markOccupied(grid, placement, bounds);
                    roomPlaced = true;
                }
            }
        }

        if (!roomPlaced) {
            return {
                success: false,
                error: `Unable to place ${room.label} - insufficient space`
            };
        }
    }

    return { success: true, rooms: placed };
}

/**
 * Create occupancy grid for efficient placement
 */
function createOccupancyGrid(bounds, cellSize) {
    const cols = Math.ceil(bounds.width / cellSize);
    const rows = Math.ceil(bounds.height / cellSize);
    return {
        cells: Array(rows).fill(null).map(() => Array(cols).fill(false)),
        cellSize,
        cols,
        rows,
        offsetX: bounds.x,
        offsetY: bounds.y
    };
}

/**
 * Find free position in grid
 */
function findFreePosition(grid, bounds, width, height, placed) {
    const cellW = Math.ceil(width / grid.cellSize);
    const cellH = Math.ceil(height / grid.cellSize);

    for (let row = 0; row <= grid.rows - cellH; row++) {
        for (let col = 0; col <= grid.cols - cellW; col++) {
            // Check if all cells are free
            let allFree = true;
            for (let r = row; r < row + cellH && allFree; r++) {
                for (let c = col; c < col + cellW && allFree; c++) {
                    if (grid.cells[r][c]) allFree = false;
                }
            }

            if (allFree) {
                const x = grid.offsetX + col * grid.cellSize;
                const y = grid.offsetY + row * grid.cellSize;

                // Verify no overlap with placed rooms
                const rect = { x, y, width, height };
                let hasOverlap = false;
                for (const p of placed) {
                    if (rectanglesOverlap(rect, p)) {
                        hasOverlap = true;
                        break;
                    }
                }

                if (!hasOverlap && x + width <= bounds.x + bounds.width && y + height <= bounds.y + bounds.height) {
                    return { x, y };
                }
            }
        }
    }

    return null;
}

/**
 * Mark cells as occupied
 */
function markOccupied(grid, room, bounds) {
    const startCol = Math.floor((room.x - grid.offsetX) / grid.cellSize);
    const startRow = Math.floor((room.y - grid.offsetY) / grid.cellSize);
    const endCol = Math.ceil((room.x + room.width - grid.offsetX) / grid.cellSize);
    const endRow = Math.ceil((room.y + room.height - grid.offsetY) / grid.cellSize);

    for (let r = Math.max(0, startRow); r < Math.min(grid.rows, endRow); r++) {
        for (let c = Math.max(0, startCol); c < Math.min(grid.cols, endCol); c++) {
            grid.cells[r][c] = true;
        }
    }
}

/**
 * Generate door position for a room
 */
function generateDoorPosition(room, allRooms, roomIndex, bounds) {
    const doorWidth = MIN_DOOR_WIDTH;

    // Find potential door positions (edges not on outer walls)
    const edges = [];

    // Check each edge - prefer internal edges
    if (room.y > bounds.y + 100) {
        edges.push({
            direction: 'south',
            x: room.x + (room.width - doorWidth) / 2,
            y: room.y,
            width: doorWidth
        });
    }

    if (room.y + room.height < bounds.y + bounds.height - 100) {
        edges.push({
            direction: 'north',
            x: room.x + (room.width - doorWidth) / 2,
            y: room.y + room.height,
            width: doorWidth
        });
    }

    if (room.x > bounds.x + 100) {
        edges.push({
            direction: 'west',
            x: room.x,
            y: room.y + (room.height - doorWidth) / 2,
            width: doorWidth
        });
    }

    if (room.x + room.width < bounds.x + bounds.width - 100) {
        edges.push({
            direction: 'east',
            x: room.x + room.width,
            y: room.y + (room.height - doorWidth) / 2,
            width: doorWidth
        });
    }

    // Prefer edges adjacent to other rooms
    for (const edge of edges) {
        for (let i = 0; i < allRooms.length; i++) {
            if (i === roomIndex) continue;

            const other = allRooms[i];
            const shared = getSharedEdge(room, other);
            if (shared) {
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

    // Default to first available edge or south
    return edges[0] || {
        direction: 'south',
        x: room.x + (room.width - doorWidth) / 2,
        y: room.y,
        width: doorWidth
    };
}

/**
 * Create error response
 */
export function createErrorResponse(reason) {
    return {
        status: 'error',
        error_reason: reason,
        floor_plan: null
    };
}

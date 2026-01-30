/**
 * Planning Agent
 * Performs architectural feasibility checks and space allocation
 */

import {
    WALL_THICKNESS,
    MIN_CORRIDOR_WIDTH,
    MAX_ASPECT_RATIO,
    ROOM_TYPES
} from '../utils/constants.js';
import {
    sqmToSqmm,
    calculatePlotDimensions,
    calculateOptimalDimensions
} from '../utils/geometry.js';

/**
 * Check feasibility and prepare room allocation plan
 * @param {Object} requirements - Validated requirements from Requirement Agent
 * @returns {Object} {success: boolean, data?: Object, error?: string}
 */
export function checkFeasibility(requirements) {
    const errors = [];

    // Determine plot dimensions
    let plotWidth, plotHeight;
    if (requirements.plotDimensions) {
        plotWidth = requirements.plotDimensions.width;
        plotHeight = requirements.plotDimensions.length;
    } else {
        // Calculate from total area (assume square plot with some buffer for walls)
        const dims = calculatePlotDimensions(requirements.totalAreaSqm);
        plotWidth = dims.width;
        plotHeight = dims.height;
    }

    // Calculate usable area (subtract perimeter walls)
    const usableWidth = plotWidth - (2 * WALL_THICKNESS);
    const usableHeight = plotHeight - (2 * WALL_THICKNESS);
    const usableAreaMm = usableWidth * usableHeight;
    const usableAreaSqm = usableAreaMm / 1_000_000;

    // Expand rooms by quantity and calculate total required area
    const expandedRooms = [];
    let totalRequiredAreaMm = 0;

    requirements.rooms.forEach(room => {
        for (let i = 0; i < room.quantity; i++) {
            const roomAreaMm = sqmToSqmm(room.minAreaSqm);

            // Calculate room dimensions
            const dims = calculateOptimalDimensions(roomAreaMm, MAX_ASPECT_RATIO);

            expandedRooms.push({
                id: `${room.type}_${i + 1}`,
                type: room.type,
                label: room.quantity > 1 ? `${room.label} ${i + 1}` : room.label,
                minAreaMm: roomAreaMm,
                minAreaSqm: room.minAreaSqm,
                suggestedWidth: dims.width,
                suggestedHeight: dims.height,
                color: room.color,
                priority: room.priority,
                adjacentTo: room.adjacentTo,
                requiresWindow: room.requiresWindow
            });

            totalRequiredAreaMm += roomAreaMm;
        }
    });

    // Check if total room area fits
    if (totalRequiredAreaMm > usableAreaMm) {
        const totalSqm = (totalRequiredAreaMm / 1_000_000).toFixed(1);
        return {
            success: false,
            error: `Total room area (${totalSqm} m²) exceeds usable area (${usableAreaSqm.toFixed(1)} m²) after accounting for walls`
        };
    }

    // Sort rooms by priority (public to private)
    expandedRooms.sort((a, b) => a.priority - b.priority);

    // Validate individual room dimensions
    for (const room of expandedRooms) {
        // Check minimum dimension (at least corridor width)
        const minDim = Math.min(room.suggestedWidth, room.suggestedHeight);
        if (minDim < MIN_CORRIDOR_WIDTH) {
            errors.push(`${room.label} is too narrow (minimum ${MIN_CORRIDOR_WIDTH}mm required)`);
        }

        // Check if room fits in plot
        if (room.suggestedWidth > usableWidth || room.suggestedHeight > usableHeight) {
            errors.push(`${room.label} dimensions exceed plot size`);
        }
    }

    if (errors.length > 0) {
        return {
            success: false,
            error: errors[0]
        };
    }

    // Calculate circulation space
    const circulationAreaMm = usableAreaMm - totalRequiredAreaMm;
    const circulationPercent = (circulationAreaMm / usableAreaMm) * 100;

    // Warn if circulation space is very low
    if (circulationPercent < 10) {
        // Not an error, just a warning we could surface later
        console.warn(`Low circulation space: ${circulationPercent.toFixed(1)}%`);
    }

    return {
        success: true,
        data: {
            plot: {
                width: plotWidth,
                height: plotHeight,
                usableWidth,
                usableHeight
            },
            rooms: expandedRooms,
            stats: {
                totalRoomAreaMm: totalRequiredAreaMm,
                totalRoomAreaSqm: totalRequiredAreaMm / 1_000_000,
                usableAreaMm,
                usableAreaSqm,
                circulationAreaMm,
                circulationPercent
            }
        }
    };
}

/**
 * Validate room placement against constraints
 * @param {Object} room - Room with placement {x, y, width, height}
 * @param {Object} plot - Plot dimensions
 * @param {Array} placedRooms - Already placed rooms
 * @returns {Object} {valid: boolean, error?: string}
 */
export function validatePlacement(room, plot, placedRooms) {
    // Check bounds
    if (room.x < WALL_THICKNESS || room.y < WALL_THICKNESS) {
        return { valid: false, error: 'Room extends beyond plot boundary' };
    }

    if (room.x + room.width > plot.width - WALL_THICKNESS ||
        room.y + room.height > plot.height - WALL_THICKNESS) {
        return { valid: false, error: 'Room extends beyond plot boundary' };
    }

    // Check overlaps with placed rooms
    for (const placed of placedRooms) {
        if (roomsOverlap(room, placed)) {
            return { valid: false, error: `Room overlaps with ${placed.label}` };
        }
    }

    // Check aspect ratio
    const ratio = Math.max(room.width, room.height) / Math.min(room.width, room.height);
    if (ratio > MAX_ASPECT_RATIO) {
        return { valid: false, error: 'Room aspect ratio exceeds maximum (1:2)' };
    }

    return { valid: true };
}

/**
 * Check if two rooms overlap
 */
function roomsOverlap(r1, r2) {
    return !(
        r1.x + r1.width <= r2.x ||
        r2.x + r2.width <= r1.x ||
        r1.y + r1.height <= r2.y ||
        r2.y + r2.height <= r1.y
    );
}

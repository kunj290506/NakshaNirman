/**
 * CAD Floor Planner - Geometry Utilities
 */

/**
 * Check if two rectangles overlap
 * @param {Object} r1 - First rectangle {x, y, width, height}
 * @param {Object} r2 - Second rectangle {x, y, width, height}
 * @returns {boolean} True if rectangles overlap
 */
export function rectanglesOverlap(r1, r2) {
    return !(
        r1.x + r1.width <= r2.x ||
        r2.x + r2.width <= r1.x ||
        r1.y + r1.height <= r2.y ||
        r2.y + r2.height <= r1.y
    );
}

/**
 * Check if rectangle is within bounds
 * @param {Object} rect - Rectangle {x, y, width, height}
 * @param {Object} bounds - Boundary {width, height}
 * @returns {boolean} True if rectangle is within bounds
 */
export function isWithinBounds(rect, bounds) {
    return (
        rect.x >= 0 &&
        rect.y >= 0 &&
        rect.x + rect.width <= bounds.width &&
        rect.y + rect.height <= bounds.height
    );
}

/**
 * Calculate aspect ratio of a rectangle
 * @param {number} width 
 * @param {number} height 
 * @returns {number} Aspect ratio (always >= 1)
 */
export function getAspectRatio(width, height) {
    if (height === 0) return Infinity;
    const ratio = width / height;
    return ratio >= 1 ? ratio : 1 / ratio;
}

/**
 * Calculate area in square millimeters
 * @param {number} width - Width in mm
 * @param {number} height - Height in mm
 * @returns {number} Area in mm²
 */
export function calculateArea(width, height) {
    return width * height;
}

/**
 * Convert square meters to square millimeters
 * @param {number} sqm - Area in square meters
 * @returns {number} Area in square millimeters
 */
export function sqmToSqmm(sqm) {
    return sqm * 1_000_000;
}

/**
 * Convert square millimeters to square meters
 * @param {number} sqmm - Area in square millimeters
 * @returns {number} Area in square meters
 */
export function sqmmToSqm(sqmm) {
    return sqmm / 1_000_000;
}

/**
 * Convert meters to millimeters
 * @param {number} m - Length in meters
 * @returns {number} Length in millimeters
 */
export function mToMm(m) {
    return m * 1000;
}

/**
 * Convert millimeters to meters
 * @param {number} mm - Length in millimeters
 * @returns {number} Length in meters
 */
export function mmToM(mm) {
    return mm / 1000;
}

/**
 * Calculate optimal dimensions for a given area with aspect ratio constraints
 * @param {number} areaSqmm - Area in square millimeters
 * @param {number} maxRatio - Maximum aspect ratio (default 2)
 * @returns {Object} {width, height} in millimeters
 */
export function calculateOptimalDimensions(areaSqmm, maxRatio = 2) {
    // Start with a square
    const side = Math.sqrt(areaSqmm);

    // Round to nearest 100mm for cleaner dimensions
    const width = Math.ceil(side / 100) * 100;
    const height = Math.ceil(areaSqmm / width / 100) * 100;

    // Ensure aspect ratio is within limits
    const ratio = getAspectRatio(width, height);
    if (ratio > maxRatio) {
        // Adjust to meet ratio constraint
        const newHeight = Math.ceil(width / maxRatio / 100) * 100;
        return { width, height: newHeight };
    }

    return { width, height };
}

/**
 * Calculate plot dimensions from total area
 * @param {number} totalAreaSqm - Total area in square meters
 * @returns {Object} {width, height} in millimeters
 */
export function calculatePlotDimensions(totalAreaSqm) {
    const areaSqmm = sqmToSqmm(totalAreaSqm);
    const side = Math.sqrt(areaSqmm);

    // Round to nearest 500mm for cleaner plot dimensions
    const dimension = Math.ceil(side / 500) * 500;

    return { width: dimension, height: dimension };
}

/**
 * Check if two rectangles are adjacent (share an edge)
 * @param {Object} r1 - First rectangle {x, y, width, height}
 * @param {Object} r2 - Second rectangle {x, y, width, height}
 * @param {number} tolerance - Tolerance for adjacency check (default 1mm)
 * @returns {boolean} True if rectangles are adjacent
 */
export function areAdjacent(r1, r2, tolerance = 1) {
    // Check if they share a vertical edge
    const shareVertical = (
        (Math.abs(r1.x + r1.width - r2.x) <= tolerance || Math.abs(r2.x + r2.width - r1.x) <= tolerance) &&
        !(r1.y + r1.height <= r2.y || r2.y + r2.height <= r1.y)
    );

    // Check if they share a horizontal edge
    const shareHorizontal = (
        (Math.abs(r1.y + r1.height - r2.y) <= tolerance || Math.abs(r2.y + r2.height - r1.y) <= tolerance) &&
        !(r1.x + r1.width <= r2.x || r2.x + r2.width <= r1.x)
    );

    return shareVertical || shareHorizontal;
}

/**
 * Get the shared edge between two adjacent rectangles
 * @param {Object} r1 - First rectangle
 * @param {Object} r2 - Second rectangle
 * @returns {Object|null} Edge info {direction, start, end, position} or null
 */
export function getSharedEdge(r1, r2) {
    const tolerance = 1;

    // r1's right edge touches r2's left edge
    if (Math.abs(r1.x + r1.width - r2.x) <= tolerance) {
        const start = Math.max(r1.y, r2.y);
        const end = Math.min(r1.y + r1.height, r2.y + r2.height);
        if (end > start) {
            return { direction: 'vertical', x: r1.x + r1.width, y1: start, y2: end };
        }
    }

    // r2's right edge touches r1's left edge
    if (Math.abs(r2.x + r2.width - r1.x) <= tolerance) {
        const start = Math.max(r1.y, r2.y);
        const end = Math.min(r1.y + r1.height, r2.y + r2.height);
        if (end > start) {
            return { direction: 'vertical', x: r1.x, y1: start, y2: end };
        }
    }

    // r1's top edge touches r2's bottom edge
    if (Math.abs(r1.y + r1.height - r2.y) <= tolerance) {
        const start = Math.max(r1.x, r2.x);
        const end = Math.min(r1.x + r1.width, r2.x + r2.width);
        if (end > start) {
            return { direction: 'horizontal', y: r1.y + r1.height, x1: start, x2: end };
        }
    }

    // r2's top edge touches r1's bottom edge
    if (Math.abs(r2.y + r2.height - r1.y) <= tolerance) {
        const start = Math.max(r1.x, r2.x);
        const end = Math.min(r1.x + r1.width, r2.x + r2.width);
        if (end > start) {
            return { direction: 'horizontal', y: r1.y, x1: start, x2: end };
        }
    }

    return null;
}

/**
 * Round a number to specified decimal places
 * @param {number} num - Number to round
 * @param {number} decimals - Decimal places
 * @returns {number} Rounded number
 */
export function roundTo(num, decimals = 0) {
    const factor = Math.pow(10, decimals);
    return Math.round(num * factor) / factor;
}

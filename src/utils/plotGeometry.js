/**
 * Plot Geometry Utilities
 * Handles irregular plot shapes, polygon validation, and boundary calculations
 */

/**
 * Detect plot shape from boundary points
 * @param {Array} boundary - Array of [x, y] coordinates
 * @returns {string} Shape type: 'rectangular', 'L-shaped', 'T-shaped', 'irregular'
 */
export function detectPlotShape(boundary) {
    if (!boundary || boundary.length < 3) return 'invalid';
    
    if (boundary.length === 4) {
        // Check if rectangular
        if (isRectangular(boundary)) return 'rectangular';
    }
    
    if (boundary.length === 6) {
        // Could be L-shaped
        if (isLShaped(boundary)) return 'L-shaped';
    }
    
    if (boundary.length === 8) {
        // Could be T-shaped or H-shaped
        if (isTShaped(boundary)) return 'T-shaped';
    }
    
    return 'irregular';
}

/**
 * Check if boundary forms a rectangle
 */
function isRectangular(boundary) {
    if (boundary.length !== 4) return false;
    
    // Check if all angles are 90 degrees
    for (let i = 0; i < 4; i++) {
        const p1 = boundary[i];
        const p2 = boundary[(i + 1) % 4];
        const p3 = boundary[(i + 2) % 4];
        
        const angle = calculateAngle(p1, p2, p3);
        if (Math.abs(angle - 90) > 1) return false; // 1 degree tolerance
    }
    
    return true;
}

/**
 * Check if boundary forms an L-shape
 */
function isLShaped(boundary) {
    if (boundary.length !== 6) return false;
    
    // L-shape should have 4 right angles and 2 reflex angles (270°)
    let rightAngles = 0;
    let reflexAngles = 0;
    
    for (let i = 0; i < 6; i++) {
        const p1 = boundary[i];
        const p2 = boundary[(i + 1) % 6];
        const p3 = boundary[(i + 2) % 6];
        
        const angle = calculateAngle(p1, p2, p3);
        if (Math.abs(angle - 90) < 5) rightAngles++;
        if (Math.abs(angle - 270) < 5) reflexAngles++;
    }
    
    return rightAngles === 4 && reflexAngles === 2;
}

/**
 * Check if boundary forms a T-shape
 */
function isTShaped(boundary) {
    // Simplified check - T-shape typically has 8 vertices
    return boundary.length === 8;
}

/**
 * Calculate angle between three points (in degrees)
 */
function calculateAngle(p1, p2, p3) {
    const v1 = { x: p1[0] - p2[0], y: p1[1] - p2[1] };
    const v2 = { x: p3[0] - p2[0], y: p3[1] - p2[1] };
    
    const dot = v1.x * v2.x + v1.y * v2.y;
    const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
    const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
    
    const cosAngle = dot / (mag1 * mag2);
    const angleRad = Math.acos(Math.max(-1, Math.min(1, cosAngle)));
    
    return angleRad * (180 / Math.PI);
}

/**
 * Calculate area of polygon using shoelace formula
 * @param {Array} boundary - Array of [x, y] coordinates
 * @returns {number} Area in square mm
 */
export function calculatePolygonArea(boundary) {
    if (!boundary || boundary.length < 3) return 0;
    
    let area = 0;
    for (let i = 0; i < boundary.length; i++) {
        const j = (i + 1) % boundary.length;
        area += boundary[i][0] * boundary[j][1];
        area -= boundary[j][0] * boundary[i][1];
    }
    
    return Math.abs(area / 2);
}

/**
 * Calculate bounding box of polygon
 * @param {Array} boundary - Array of [x, y] coordinates
 * @returns {Object} {minX, minY, maxX, maxY, width, height}
 */
export function calculateBoundingBox(boundary) {
    if (!boundary || boundary.length === 0) {
        return { minX: 0, minY: 0, maxX: 0, maxY: 0, width: 0, height: 0 };
    }
    
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;
    
    for (const [x, y] of boundary) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
    }
    
    return {
        minX,
        minY,
        maxX,
        maxY,
        width: maxX - minX,
        height: maxY - minY
    };
}

/**
 * Check if point is inside polygon (ray casting algorithm)
 * @param {Array} point - [x, y]
 * @param {Array} boundary - Array of [x, y] coordinates
 * @returns {boolean}
 */
export function isPointInPolygon(point, boundary) {
    const [x, y] = point;
    let inside = false;
    
    for (let i = 0, j = boundary.length - 1; i < boundary.length; j = i++) {
        const [xi, yi] = boundary[i];
        const [xj, yj] = boundary[j];
        
        const intersect = ((yi > y) !== (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        
        if (intersect) inside = !inside;
    }
    
    return inside;
}

/**
 * Check if rectangle fits within polygon boundary
 * @param {Object} rect - {x, y, width, height}
 * @param {Array} boundary - Array of [x, y] coordinates
 * @returns {boolean}
 */
export function rectangleFitsInPolygon(rect, boundary) {
    // Check all four corners
    const corners = [
        [rect.x, rect.y],
        [rect.x + rect.width, rect.y],
        [rect.x + rect.width, rect.y + rect.height],
        [rect.x, rect.y + rect.height]
    ];
    
    return corners.every(corner => isPointInPolygon(corner, boundary));
}

/**
 * Normalize boundary to start from origin
 * @param {Array} boundary - Array of [x, y] coordinates
 * @returns {Array} Normalized boundary
 */
export function normalizeBoundary(boundary) {
    if (!boundary || boundary.length === 0) return [];
    
    const bbox = calculateBoundingBox(boundary);
    
    return boundary.map(([x, y]) => [
        x - bbox.minX,
        y - bbox.minY
    ]);
}

/**
 * Convert rectangular dimensions to boundary polygon
 * @param {number} width - Width in mm
 * @param {number} height - Height in mm
 * @returns {Array} Boundary as array of [x, y] coordinates
 */
export function rectangleToBoundary(width, height) {
    return [
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ];
}

/**
 * Validate boundary polygon
 * @param {Array} boundary - Array of [x, y] coordinates
 * @returns {Object} {valid: boolean, error?: string}
 */
export function validateBoundary(boundary) {
    if (!boundary || !Array.isArray(boundary)) {
        return { valid: false, error: 'Boundary must be an array' };
    }
    
    if (boundary.length < 3) {
        return { valid: false, error: 'Boundary must have at least 3 points' };
    }
    
    // Check if all points are valid
    for (const point of boundary) {
        if (!Array.isArray(point) || point.length !== 2) {
            return { valid: false, error: 'Each point must be [x, y]' };
        }
        if (typeof point[0] !== 'number' || typeof point[1] !== 'number') {
            return { valid: false, error: 'Coordinates must be numbers' };
        }
    }
    
    // Check if polygon is closed (first and last point should be same or close)
    const first = boundary[0];
    const last = boundary[boundary.length - 1];
    const distance = Math.sqrt(
        Math.pow(first[0] - last[0], 2) + Math.pow(first[1] - last[1], 2)
    );
    
    if (distance > 10) { // 10mm tolerance
        return { valid: false, error: 'Boundary polygon is not closed' };
    }
    
    // Check for self-intersection (simplified check)
    // A proper implementation would check all edge pairs
    
    return { valid: true };
}

/**
 * Decompose irregular polygon into rectangular regions
 * Used for room placement in non-rectangular plots
 * @param {Array} boundary - Array of [x, y] coordinates
 * @returns {Array} Array of rectangular regions {x, y, width, height}
 */
export function decomposePolygonToRectangles(boundary) {
    // Simplified implementation - returns bounding box
    // A full implementation would use partition algorithms
    const bbox = calculateBoundingBox(boundary);
    
    return [{
        x: bbox.minX,
        y: bbox.minY,
        width: bbox.width,
        height: bbox.height
    }];
}

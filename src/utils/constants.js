/**
 * CAD Floor Planner - Constants
 * All dimensions in millimeters unless otherwise specified
 */

// Wall & Corridor Constraints
export const WALL_THICKNESS = 230;        // mm
export const MIN_CORRIDOR_WIDTH = 900;    // mm
export const MIN_DOOR_WIDTH = 800;        // mm
export const MAX_DOOR_WIDTH = 1200;       // mm

// Room Constraints
export const MIN_ROOM_AREA_SQM = 4;       // Minimum room size in m²
export const MAX_ASPECT_RATIO = 2;        // Maximum width:height or height:width ratio
export const MIN_ASPECT_RATIO = 1;        // Minimum (square) ratio

// Plot Constraints
export const MIN_PLOT_DIMENSION = 3000;   // mm (3m minimum side)
export const MAX_PLOT_DIMENSION = 100000; // mm (100m maximum side)

// Room Type Definitions with default properties
export const ROOM_TYPES = {
    living_room: {
        label: 'Living Room',
        defaultAreaSqm: 18,
        minAreaSqm: 12,
        color: '#bfdbfe',
        priority: 1,        // Lower = placed first (public to private)
        adjacentTo: ['dining_room', 'kitchen'],
        requiresWindow: true
    },
    dining_room: {
        label: 'Dining Room',
        defaultAreaSqm: 12,
        minAreaSqm: 8,
        color: '#fecaca',
        priority: 2,
        adjacentTo: ['living_room', 'kitchen'],
        requiresWindow: false
    },
    kitchen: {
        label: 'Kitchen',
        defaultAreaSqm: 10,
        minAreaSqm: 6,
        color: '#fde68a',
        priority: 3,
        adjacentTo: ['dining_room'],
        requiresWindow: true
    },
    bedroom: {
        label: 'Bedroom',
        defaultAreaSqm: 12,
        minAreaSqm: 9,
        color: '#c7d2fe',
        priority: 4,
        adjacentTo: [],
        requiresWindow: true
    },
    master_bedroom: {
        label: 'Master Bedroom',
        defaultAreaSqm: 16,
        minAreaSqm: 12,
        color: '#ddd6fe',
        priority: 4,
        adjacentTo: ['bathroom'],
        requiresWindow: true
    },
    bathroom: {
        label: 'Bathroom',
        defaultAreaSqm: 5,
        minAreaSqm: 3,
        color: '#a5f3fc',
        priority: 5,
        adjacentTo: ['bedroom', 'master_bedroom'],
        requiresWindow: false
    },
    study: {
        label: 'Study',
        defaultAreaSqm: 9,
        minAreaSqm: 6,
        color: '#d9f99d',
        priority: 4,
        adjacentTo: [],
        requiresWindow: true
    },
    storage: {
        label: 'Storage',
        defaultAreaSqm: 4,
        minAreaSqm: 2,
        color: '#e2e8f0',
        priority: 6,
        adjacentTo: [],
        requiresWindow: false
    },
    utility: {
        label: 'Utility',
        defaultAreaSqm: 6,
        minAreaSqm: 4,
        color: '#e5e7eb',
        priority: 5,
        adjacentTo: ['kitchen'],
        requiresWindow: false
    },
    balcony: {
        label: 'Balcony',
        defaultAreaSqm: 6,
        minAreaSqm: 3,
        color: '#bbf7d0',
        priority: 6,
        adjacentTo: ['living_room', 'bedroom'],
        requiresWindow: false
    }
};

// Room type keys for dropdown
export const ROOM_TYPE_OPTIONS = Object.entries(ROOM_TYPES).map(([key, value]) => ({
    value: key,
    label: value.label
}));

// Conversion factors
export const SQM_TO_SQMM = 1_000_000;    // 1 m² = 1,000,000 mm²
export const M_TO_MM = 1000;              // 1 m = 1000 mm

// DXF Layer Names
export const DXF_LAYERS = {
    WALLS: 'WALLS',
    DOORS: 'DOORS',
    ROOMS: 'ROOMS',
    TEXT: 'TEXT',
    DIMENSIONS: 'DIMENSIONS',
    GRID: 'GRID'
};

// Canvas rendering
export const CANVAS_PADDING = 50;         // px
export const CANVAS_GRID_SIZE = 1000;     // mm (1m grid)
export const CANVAS_MIN_ZOOM = 0.1;
export const CANVAS_MAX_ZOOM = 5;
export const CANVAS_ZOOM_STEP = 0.2;

/**
 * CAD Floor Planner - Constants
 * All dimensions in millimeters unless otherwise specified
 */

// Wall & Corridor Constraints (Indian Residential Standards)
export const WALL_THICKNESS = 230;        // mm (9 inches standard)
export const EXTERNAL_WALL_THICKNESS = 230; // mm
export const INTERNAL_WALL_THICKNESS = 115; // mm (4.5 inches partition)
export const MIN_CORRIDOR_WIDTH = 900;    // mm
export const MIN_DOOR_WIDTH = 800;        // mm
export const MAX_DOOR_WIDTH = 1200;       // mm
export const WINDOW_HEIGHT = 1200;        // mm
export const WINDOW_SILL_HEIGHT = 750;    // mm
export const MIN_WINDOW_WIDTH = 900;      // mm

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
        priority: 1,
        privacyLevel: 'public',
        adjacentTo: ['dining_room', 'kitchen'],
        requiresWindow: true
    },
    dining_room: {
        label: 'Dining Room',
        defaultAreaSqm: 12,
        minAreaSqm: 8,
        color: '#fecaca',
        priority: 2,
        privacyLevel: 'public',
        adjacentTo: ['living_room', 'kitchen'],
        requiresWindow: false
    },
    kitchen: {
        label: 'Kitchen',
        defaultAreaSqm: 10,
        minAreaSqm: 6,
        color: '#fde68a',
        priority: 3,
        privacyLevel: 'semi-private',
        adjacentTo: ['dining_room'],
        requiresWindow: true
    },
    bedroom: {
        label: 'Bedroom',
        defaultAreaSqm: 12,
        minAreaSqm: 9,
        color: '#c7d2fe',
        priority: 4,
        privacyLevel: 'private',
        adjacentTo: [],
        requiresWindow: true
    },
    master_bedroom: {
        label: 'Master Bedroom',
        defaultAreaSqm: 16,
        minAreaSqm: 12,
        color: '#ddd6fe',
        priority: 4,
        privacyLevel: 'private',
        adjacentTo: ['bathroom'],
        requiresWindow: true
    },
    bathroom: {
        label: 'Bathroom',
        defaultAreaSqm: 5,
        minAreaSqm: 3,
        color: '#a5f3fc',
        priority: 5,
        privacyLevel: 'private',
        adjacentTo: ['bedroom', 'master_bedroom'],
        requiresWindow: false
    },
    study: {
        label: 'Study',
        defaultAreaSqm: 9,
        minAreaSqm: 6,
        color: '#d9f99d',
        priority: 4,
        privacyLevel: 'semi-private',
        adjacentTo: [],
        requiresWindow: true
    },
    storage: {
        label: 'Storage',
        defaultAreaSqm: 4,
        minAreaSqm: 2,
        color: '#e2e8f0',
        priority: 6,
        privacyLevel: 'semi-private',
        adjacentTo: [],
        requiresWindow: false
    },
    utility: {
        label: 'Utility',
        defaultAreaSqm: 6,
        minAreaSqm: 4,
        color: '#e5e7eb',
        priority: 5,
        privacyLevel: 'semi-private',
        adjacentTo: ['kitchen'],
        requiresWindow: false
    },
    balcony: {
        label: 'Balcony',
        defaultAreaSqm: 6,
        minAreaSqm: 3,
        color: '#bbf7d0',
        priority: 6,
        privacyLevel: 'semi-private',
        adjacentTo: ['living_room', 'bedroom'],
        requiresWindow: false
    },
    parking: {
        label: 'Parking',
        defaultAreaSqm: 14,
        minAreaSqm: 10,
        color: '#cbd5e1',
        priority: 7,
        privacyLevel: 'public',
        adjacentTo: [],
        requiresWindow: false
    },
    pooja_room: {
        label: 'Pooja Room',
        defaultAreaSqm: 4,
        minAreaSqm: 2,
        color: '#fcd34d',
        priority: 5,
        privacyLevel: 'semi-private',
        adjacentTo: ['living_room'],
        requiresWindow: false
    },
    shop: {
        label: 'Shop',
        defaultAreaSqm: 20,
        minAreaSqm: 12,
        color: '#fbbf24',
        priority: 1,
        privacyLevel: 'public',
        adjacentTo: [],
        requiresWindow: true,
        requiresStreetAccess: true
    },
    staircase: {
        label: 'Staircase',
        defaultAreaSqm: 8,
        minAreaSqm: 5,
        color: '#9ca3af',
        priority: 3,
        privacyLevel: 'public',
        adjacentTo: [],
        requiresWindow: false
    }
};

// Privacy Hierarchy (for Indian Residential Design)
export const PRIVACY_LEVELS = {
    PUBLIC: 'public',           // Living room, dining, shop, entrance
    SEMI_PRIVATE: 'semi-private', // Kitchen, study, pooja room
    PRIVATE: 'private'          // Bedrooms, bathrooms
};

// Room type keys for dropdown
export const ROOM_TYPE_OPTIONS = Object.entries(ROOM_TYPES).map(([key, value]) => ({
    value: key,
    label: value.label
}));

// Conversion factors
export const SQM_TO_SQMM = 1_000_000;    // 1 m² = 1,000,000 mm²
export const M_TO_MM = 1000;              // 1 m = 1000 mm

// DXF Layer Names (AutoCAD Compatible)
export const DXF_LAYERS = {
    WALLS: 'WALLS',
    DOORS: 'DOORS',
    WINDOWS: 'WINDOWS',
    FURNITURE: 'FURNITURE',
    SANITARY: 'SANITARY',
    ROOMS: 'ROOMS',
    TEXT: 'TEXT',
    DIMENSIONS: 'DIMENSIONS',
    GRID: 'GRID',
    ANNOTATIONS: 'ANNOTATIONS'
};

// Furniture Dimensions (Indian Standards - All in mm)
export const FURNITURE = {
    // Living Room
    SOFA_3SEATER: { width: 2100, depth: 900, label: 'Sofa (3-Seater)' },
    SOFA_2SEATER: { width: 1500, depth: 900, label: 'Sofa (2-Seater)' },
    TV_UNIT: { width: 1800, depth: 450, label: 'TV Unit' },
    COFFEE_TABLE: { width: 1200, depth: 600, label: 'Coffee Table' },
    
    // Dining Room
    DINING_TABLE_4: { width: 1200, depth: 900, label: 'Dining Table (4-Seater)' },
    DINING_TABLE_6: { width: 1800, depth: 900, label: 'Dining Table (6-Seater)' },
    DINING_CHAIR: { width: 450, depth: 450, label: 'Dining Chair' },
    
    // Bedroom
    DOUBLE_BED: { width: 1900, depth: 1500, label: 'Double Bed' },
    QUEEN_BED: { width: 2100, depth: 1800, label: 'Queen Bed' },
    KING_BED: { width: 2100, depth: 2100, label: 'King Bed' },
    WARDROBE: { width: 1800, depth: 600, label: 'Wardrobe' },
    SIDE_TABLE: { width: 450, depth: 450, label: 'Side Table' },
    STUDY_TABLE: { width: 1200, depth: 600, label: 'Study Table' },
    
    // Kitchen
    KITCHEN_PLATFORM: { depth: 600, height: 850, label: 'Kitchen Platform' },
    REFRIGERATOR: { width: 600, depth: 600, label: 'Refrigerator' },
    STOVE_4BURNER: { width: 600, depth: 600, label: 'Stove (4-Burner)' },
    SINK_SINGLE: { width: 600, depth: 450, label: 'Kitchen Sink' },
    SINK_DOUBLE: { width: 900, depth: 450, label: 'Double Sink' }
};

// Sanitary Fixtures (Indian Standards - All in mm)
export const SANITARY_FIXTURES = {
    WC_INDIAN: { width: 450, depth: 600, label: 'Indian WC' },
    WC_WESTERN: { width: 500, depth: 700, label: 'Western WC' },
    WASH_BASIN: { width: 500, depth: 400, label: 'Wash Basin' },
    SHOWER_AREA: { width: 900, depth: 900, label: 'Shower Area' },
    BATHTUB: { width: 1500, depth: 700, label: 'Bathtub' },
    URINAL: { width: 400, depth: 350, label: 'Urinal' }
};

// Staircase Specifications (Indian Residential)
export const STAIRCASE_SPECS = {
    MIN_WIDTH: 900,           // mm
    TREAD_DEPTH: 250,         // mm (going)
    RISER_HEIGHT: 175,        // mm (rise)
    TYPICAL_FLIGHT: 3000,     // mm height for one floor
    LANDING_DEPTH: 900        // mm
};

// Canvas rendering
export const CANVAS_PADDING = 50;         // px
export const CANVAS_GRID_SIZE = 1000;     // mm (1m grid)
export const CANVAS_MIN_ZOOM = 0.1;
export const CANVAS_MAX_ZOOM = 5;
export const CANVAS_ZOOM_STEP = 0.2;

/**
 * Real-World Floor Plan Dataset
 * sourced from architectural standards and common residential layouts.
 * 
 * Each template represents a "Real Life" valid floor plan geometry.
 */

export const FLOOR_PLAN_TEMPLATES = [
    {
        id: '2bhk_standard_type_a',
        name: 'Standard 2BHK Apartment (30x40)',
        source: 'Real World Residential',
        minAreaSqm: 60,
        maxAreaSqm: 110,
        bedroomCount: 2,
        gridRatio: 1.33, // 4:3 aspect ratio
        // Normalized coordinates (0 to 1 scale)
        rooms: [
            { type: 'living_room', x: 0.05, y: 0.05, w: 0.55, h: 0.45, label: 'LIVING' },
            { type: 'dining_room', x: 0.60, y: 0.05, w: 0.35, h: 0.45, label: 'DINING' },
            { type: 'kitchen', x: 0.60, y: 0.50, w: 0.35, h: 0.25, label: 'KITCHEN' },
            { type: 'utility', x: 0.60, y: 0.75, w: 0.35, h: 0.20, label: 'UTIL' },
            { type: 'master_bedroom', x: 0.05, y: 0.50, w: 0.40, h: 0.35, label: 'M.BED' },
            { type: 'bathroom', x: 0.45, y: 0.50, w: 0.15, h: 0.20, label: 'TOILET' }, // attached
            { type: 'bedroom', x: 0.05, y: 0.85, w: 0.35, h: 0.10, label: 'BED-2' }, // Placeholder, usually bigger, let's adjust
            // Better 2BHK Coordinates (Standard 30x40 Box)
            // Left Side: Living (Front), Mas. Bed (Back)
            // Right Side: Kitchen+Dining (Front), Bed 2 (Back)
            // Center/Core: Bathrooms
        ],
        // Let's redefine with a cleaner block structure
        structure: [
            // ROW 1 (Front)
            { type: 'living_room', x: 0, y: 0, w: 0.5, h: 0.4 },
            { type: 'kitchen', x: 0.5, y: 0, w: 0.3, h: 0.4 },
            { type: 'dining_room', x: 0.8, y: 0, w: 0.2, h: 0.4 },

            // ROW 2 (Middle - Services/Corridor)
            { type: 'corridor', x: 0.4, y: 0.4, w: 0.6, h: 0.1 }, // Hallway
            { type: 'bathroom', x: 0, y: 0.4, w: 0.2, h: 0.2 },   // Common Bath
            { type: 'bathroom', x: 0.2, y: 0.4, w: 0.2, h: 0.2 }, // Attached Bath (for M.Bed below)

            // ROW 3 (Back - Private)
            { type: 'master_bedroom', x: 0, y: 0.6, w: 0.5, h: 0.4 },
            { type: 'bedroom', x: 0.5, y: 0.6, w: 0.5, h: 0.4 }
        ]
    },
    {
        id: '3bhk_modern_luxury',
        name: 'Luxury 3BHK Layout',
        minAreaSqm: 100,
        maxAreaSqm: 200,
        bedroomCount: 3,
        structure: [
            // Left Wing (Active)
            { type: 'living_room', x: 0, y: 0, w: 0.4, h: 0.5 },
            { type: 'dining_room', x: 0.4, y: 0.1, w: 0.3, h: 0.4 },
            { type: 'balcony', x: 0, y: 0.5, w: 0.4, h: 0.15 },

            // Top Right (Service)
            { type: 'kitchen', x: 0.7, y: 0, w: 0.3, h: 0.3 },
            { type: 'utility', x: 0.7, y: 0.3, w: 0.3, h: 0.1 },

            // Bottom (Private)
            { type: 'master_bedroom', x: 0, y: 0.65, w: 0.4, h: 0.35 },
            { type: 'bathroom', x: 0.4, y: 0.65, w: 0.15, h: 0.2 }, // Master Bath

            { type: 'bedroom', x: 0.55, y: 0.5, w: 0.45, h: 0.25 }, // Bed 2
            { type: 'bedroom', x: 0.55, y: 0.75, w: 0.45, h: 0.25 }, // Bed 3

            { type: 'bathroom', x: 0.4, y: 0.85, w: 0.15, h: 0.15 }, // Common Bath

            { type: 'corridor', x: 0.4, y: 0.5, w: 0.15, h: 0.15 } // Central Hub
        ]
    }
];

export const GENERIC_TEMPLATES = {
    // Template for standard rectangular plots
    rectangular: (rows, cols) => {
        // ... generator function based on typical grid patterns
    }
};

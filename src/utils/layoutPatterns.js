/**
 * Architectural Layout Patterns
 * Defines logical room arrangements based on real-world floor plans
 */

import { WALL_THICKNESS } from './constants.js';

/**
 * Standard Room Aspect Ratios (width:height)
 * Real-world rooms are rarely square or extremely narrow
 */
export const ROOM_ASPECTS = {
    living_room: { min: 1.2, max: 1.8, target: 1.5 },   // Rectangular
    dining_room: { min: 1.0, max: 1.4, target: 1.2 },   // Slightly rectangular
    bedroom: { min: 1.0, max: 1.3, target: 1.1 },       // Nearly square
    master_bedroom: { min: 1.1, max: 1.4, target: 1.25 }, // Rectangular
    kitchen: { min: 1.0, max: 1.5, target: 1.2 },       // Rectangular (platform space)
    bathroom: { min: 1.2, max: 2.0, target: 1.6 },      // Narrow/Rectangular
    parking: { min: 1.5, max: 2.5, target: 1.8 },       // Car shape
    corridor: { min: 3.0, max: 10.0, target: 5.0 }      // Long
};

/**
 * Adjacency Matrix - Physics-like attraction forces
 * Higher number = stronger desire to be close
 */
export const ADJACENCY_WEIGHTS = {
    'living_room': { 'dining_room': 10, 'balcony': 8, 'parking': 5, 'bedroom': 2 },
    'dining_room': { 'kitchen': 10, 'living_room': 10, 'utility': 5 },
    'kitchen': { 'dining_room': 10, 'utility': 8, 'store': 8 },
    'bedroom': { 'bathroom': 10, 'balcony': 5, 'living_room': 2 },
    'master_bedroom': { 'bathroom': 10, 'balcony': 6, 'living_room': 1 },
    'bathroom': { 'bedroom': 10, 'master_bedroom': 10, 'living_room': 1 },
};

/**
 * Try to fit a room based on architectural constraints
 */
export function calculateOptimalDimensions(areaMm2, type) {
    const aspect = ROOM_ASPECTS[type] || { min: 1.0, max: 1.5, target: 1.2 };

    // width * height = area
    // width / height = ratio
    // width = height * ratio
    // height * ratio * height = area
    // height^2 = area / ratio

    const h = Math.sqrt(areaMm2 / aspect.target);
    const w = h * aspect.target;

    return {
        width: Math.round(w),
        height: Math.round(h)
    };
}

/**
 * Central Hall Architecture Pattern
 * Generates a layout where Living/Dining is central, and other rooms surround it.
 * 
 * Layout Strategy:
 * 1. Place Living+Dining in the center (The Core)
 * 2. Place Master Bedroom (Bottom-Left/Right corner)
 * 3. Place Kitchen (Top-Left/Right corner)
 * 4. Place other Bedrooms (Remaining corners)
 * 5. Fill gaps with Bathrooms/Utility
 */
export function generateCentralHallLayout(rooms, availableWidth, availableHeight) {
    // This is a simplified "Block Layout" generator
    // It divides the canvas into a 3x3 or 3x2 grid logically, but flexibility sized

    const layout = [];
    const usedArea = { x: 0, y: 0, w: 0, h: 0 };

    // 1. Identify Core Rooms
    const living = rooms.find(r => r.type === 'living_room');
    const dining = rooms.find(r => r.type === 'dining_room');
    const kitchen = rooms.find(r => r.type === 'kitchen');
    const master = rooms.find(r => r.type === 'master_bedroom');
    const bedrooms = rooms.filter(r => r.type === 'bedroom');
    const bathrooms = rooms.filter(r => r.type === 'bathroom');

    // Define the "Spine" width (Living + Dining)
    const spineWidth = Math.min(availableWidth * 0.4, 4500); // Max 4.5m wide hall
    const spineX = (availableWidth - spineWidth) / 2;

    let currentY = 0;

    // PLACEMENT 1: Living Room (Front Center)
    if (living) {
        layout.push({
            ...living,
            x: spineX,
            y: 0,
            width: spineWidth,
            height: living.suggestedHeight // Preserve area-based height
        });
        currentY += living.suggestedHeight;
    }

    // PLACEMENT 2: Dining Room (Center Behind Living)
    if (dining) {
        layout.push({
            ...dining,
            x: spineX,
            y: currentY,
            width: spineWidth,
            height: dining.suggestedHeight
        });
        currentY += dining.suggestedHeight;
    } else {
        // If no dining separate, living acts as hall, keep currentY
    }

    const spineHeight = currentY;

    // Side Zones
    const leftZoneX = 0;
    const leftZoneW = spineX;
    const rightZoneX = spineX + spineWidth;
    const rightZoneW = availableWidth - rightZoneX;

    let leftY = 0;
    let rightY = 0;

    // PLACEMENT 3: Place Rooms in Side Zones
    // Strategy: Alternate heavy rooms (Bedrooms) and service rooms (Kitchen/Bath)

    // Left Front: Parking or Bedroom type
    const parking = rooms.find(r => r.type === 'parking');
    if (parking) {
        layout.push({
            ...parking,
            x: leftZoneX,
            y: leftY,
            width: leftZoneW,
            height: parking.suggestedHeight
        });
        leftY += parking.suggestedHeight;
    }

    // Right Front: Maybe a Bedroom or Kitchen
    if (kitchen) {
        layout.push({
            ...kitchen,
            x: rightZoneX,
            y: rightY,
            width: rightZoneW,
            height: kitchen.suggestedHeight
        });
        rightY += kitchen.suggestedHeight;
    }

    // Master Bedroom (Best spot: Rear corner - usually South West in Vastu/Indian context, but here just Rear Left)
    if (master) {
        layout.push({
            ...master,
            x: leftZoneX,
            y: leftY,
            width: leftZoneW,
            height: master.suggestedHeight
        });

        // Try to place attached bath immediately
        const attBath = bathrooms.shift();
        if (attBath) {
            // Place bath "inside" the master bedroom zone visually or adjacent
            // For block layout, place below master
            layout.push({
                ...attBath,
                x: leftZoneX,
                y: leftY + master.suggestedHeight,
                width: leftZoneW / 2, // Half width
                height: attBath.suggestedHeight
            });
            leftY += master.suggestedHeight + attBath.suggestedHeight;
        } else {
            leftY += master.suggestedHeight;
        }
    }

    // Other Bedrooms (Right side)
    bedrooms.forEach(bed => {
        layout.push({
            ...bed,
            x: rightZoneX,
            y: rightY,
            width: rightZoneW,
            height: bed.suggestedHeight
        });

        // Flexible Bathrooms for these bedrooms
        const commonBath = bathrooms.shift();
        if (commonBath) {
            layout.push({
                ...commonBath,
                x: rightZoneX,
                y: rightY + bed.suggestedHeight,
                width: rightZoneW / 2,
                height: commonBath.suggestedHeight
            });
            rightY += bed.suggestedHeight + commonBath.suggestedHeight;
        } else {
            rightY += bed.suggestedHeight;
        }
    });

    // Handle remaining bathrooms (Common)
    bathrooms.forEach(bath => {
        // Place in any remaining gap, e.g., end of left zone
        layout.push({
            ...bath,
            x: leftZoneX,
            y: leftY,
            width: leftZoneW / 2,
            height: bath.suggestedHeight
        });
        leftY += bath.suggestedHeight;
    });

    // PLACEMENT: Other rooms (Study, Pooja, etc.)
    const others = rooms.filter(r => !layout.includes(r) && !bathrooms.includes(r));
    others.forEach(room => {
        // Just append to shorter side
        if (leftY < rightY) {
            layout.push({ ...room, x: leftZoneX, y: leftY, width: leftZoneW, height: room.suggestedHeight });
            leftY += room.suggestedHeight;
        } else {
            layout.push({ ...room, x: rightZoneX, y: rightY, width: rightZoneW, height: room.suggestedHeight });
            rightY += room.suggestedHeight;
        }
    });

    return layout;
}

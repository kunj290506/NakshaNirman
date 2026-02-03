/**
 * Furniture Placement Engine
 * Automatically generates realistic furniture layouts for all room types
 */

import { FURNITURE, SANITARY_FIXTURES, ROOM_TYPES } from './constants.js';

/**
 * Generate furniture layout for a room
 * @param {Object} room - Room object with type, x, y, width, height
 * @returns {Array} Array of furniture items with positions
 */
export function generateFurnitureLayout(room) {
    const furniture = [];
    
    switch (room.type) {
        case 'living_room':
            furniture.push(...placeLivingRoomFurniture(room));
            break;
        case 'bedroom':
        case 'master_bedroom':
            furniture.push(...placeBedroomFurniture(room));
            break;
        case 'kitchen':
            furniture.push(...placeKitchenFurniture(room));
            break;
        case 'dining_room':
            furniture.push(...placeDiningRoomFurniture(room));
            break;
        case 'bathroom':
            furniture.push(...placeBathroomFixtures(room));
            break;
        case 'study':
            furniture.push(...placeStudyFurniture(room));
            break;
        default:
            break;
    }
    
    return furniture;
}

/**
 * Place living room furniture
 */
function placeLivingRoomFurniture(room) {
    const items = [];
    const { x, y, width, height } = room;
    const clearance = 600; // mm clearance from walls
    
    // Place 3-seater sofa on longest wall
    const sofa = FURNITURE.SOFA_3SEATER;
    if (width >= sofa.width + clearance && height >= sofa.depth + clearance) {
        items.push({
            type: 'SOFA_3SEATER',
            x: x + clearance,
            y: y + clearance,
            width: sofa.width,
            depth: sofa.depth,
            rotation: 0,
            label: sofa.label
        });
        
        // Coffee table in front
        const coffeeTable = FURNITURE.COFFEE_TABLE;
        items.push({
            type: 'COFFEE_TABLE',
            x: x + clearance + (sofa.width - coffeeTable.width) / 2,
            y: y + clearance + sofa.depth + 400,
            width: coffeeTable.width,
            depth: coffeeTable.depth,
            rotation: 0,
            label: coffeeTable.label
        });
    }
    
    // TV unit on opposite wall
    const tvUnit = FURNITURE.TV_UNIT;
    if (height >= tvUnit.depth + clearance + 1500) {
        items.push({
            type: 'TV_UNIT',
            x: x + clearance,
            y: y + height - clearance - tvUnit.depth,
            width: tvUnit.width,
            depth: tvUnit.depth,
            rotation: 0,
            label: tvUnit.label
        });
    }
    
    return items;
}

/**
 * Place bedroom furniture
 */
function placeBedroomFurniture(room) {
    const items = [];
    const { x, y, width, height } = room;
    const clearance = 600;
    
    // Determine bed size based on room type
    let bed = room.type === 'master_bedroom' ? FURNITURE.KING_BED : FURNITURE.DOUBLE_BED;
    
    // Check if bed fits
    if (width < bed.width + clearance || height < bed.depth + clearance) {
        bed = FURNITURE.DOUBLE_BED; // Fallback to smaller bed
    }
    
    // Place bed centered on back wall
    if (width >= bed.width + clearance && height >= bed.depth + clearance) {
        items.push({
            type: bed === FURNITURE.KING_BED ? 'KING_BED' : 'DOUBLE_BED',
            x: x + (width - bed.width) / 2,
            y: y + height - clearance - bed.depth,
            width: bed.width,
            depth: bed.depth,
            rotation: 0,
            label: bed.label
        });
        
        // Side tables on both sides
        const sideTable = FURNITURE.SIDE_TABLE;
        items.push({
            type: 'SIDE_TABLE',
            x: x + (width - bed.width) / 2 - sideTable.width - 100,
            y: y + height - clearance - bed.depth,
            width: sideTable.width,
            depth: sideTable.depth,
            rotation: 0,
            label: 'Side Table L'
        });
        
        items.push({
            type: 'SIDE_TABLE',
            x: x + (width - bed.width) / 2 + bed.width + 100,
            y: y + height - clearance - bed.depth,
            width: sideTable.width,
            depth: sideTable.depth,
            rotation: 0,
            label: 'Side Table R'
        });
    }
    
    // Wardrobe on side wall
    const wardrobe = FURNITURE.WARDROBE;
    if (width >= wardrobe.width + clearance) {
        items.push({
            type: 'WARDROBE',
            x: x + clearance,
            y: y + clearance,
            width: wardrobe.width,
            depth: wardrobe.depth,
            rotation: 0,
            label: wardrobe.label
        });
    }
    
    return items;
}

/**
 * Place kitchen furniture and fixtures
 */
function placeKitchenFurniture(room) {
    const items = [];
    const { x, y, width, height } = room;
    const platformDepth = FURNITURE.KITCHEN_PLATFORM.depth;
    const clearance = 100;
    
    // L-shaped or linear platform based on room shape
    const isSquareish = Math.abs(width - height) < 1000;
    
    if (isSquareish) {
        // L-shaped kitchen
        // Platform along bottom wall
        items.push({
            type: 'KITCHEN_PLATFORM',
            x: x + clearance,
            y: y + clearance,
            width: width - 2 * clearance,
            depth: platformDepth,
            rotation: 0,
            label: 'Kitchen Platform'
        });
        
        // Platform along right wall
        items.push({
            type: 'KITCHEN_PLATFORM',
            x: x + width - clearance - platformDepth,
            y: y + clearance,
            width: platformDepth,
            depth: height / 2,
            rotation: 90,
            label: 'Kitchen Platform'
        });
        
        // Sink in corner
        items.push({
            type: 'SINK_DOUBLE',
            x: x + width - clearance - platformDepth - FURNITURE.SINK_DOUBLE.width,
            y: y + clearance + 100,
            width: FURNITURE.SINK_DOUBLE.width,
            depth: FURNITURE.SINK_DOUBLE.depth,
            rotation: 0,
            label: FURNITURE.SINK_DOUBLE.label
        });
        
        // Stove on bottom platform
        items.push({
            type: 'STOVE_4BURNER',
            x: x + clearance + 500,
            y: y + clearance + 100,
            width: FURNITURE.STOVE_4BURNER.width,
            depth: FURNITURE.STOVE_4BURNER.depth,
            rotation: 0,
            label: FURNITURE.STOVE_4BURNER.label
        });
        
        // Refrigerator in opposite corner
        items.push({
            type: 'REFRIGERATOR',
            x: x + clearance,
            y: y + height - clearance - FURNITURE.REFRIGERATOR.depth,
            width: FURNITURE.REFRIGERATOR.width,
            depth: FURNITURE.REFRIGERATOR.depth,
            rotation: 0,
            label: FURNITURE.REFRIGERATOR.label
        });
        
    } else {
        // Linear kitchen (along longest wall)
        const alongWidth = width > height;
        
        if (alongWidth) {
            items.push({
                type: 'KITCHEN_PLATFORM',
                x: x + clearance,
                y: y + clearance,
                width: width - 2 * clearance,
                depth: platformDepth,
                rotation: 0,
                label: 'Kitchen Platform'
            });
            
            items.push({
                type: 'SINK_SINGLE',
                x: x + width / 2,
                y: y + clearance + 100,
                width: FURNITURE.SINK_SINGLE.width,
                depth: FURNITURE.SINK_SINGLE.depth,
                rotation: 0,
                label: FURNITURE.SINK_SINGLE.label
            });
            
            items.push({
                type: 'STOVE_4BURNER',
                x: x + clearance + 400,
                y: y + clearance + 100,
                width: FURNITURE.STOVE_4BURNER.width,
                depth: FURNITURE.STOVE_4BURNER.depth,
                rotation: 0,
                label: FURNITURE.STOVE_4BURNER.label
            });
        }
    }
    
    return items;
}

/**
 * Place dining room furniture
 */
function placeDiningRoomFurniture(room) {
    const items = [];
    const { x, y, width, height } = room;
    const clearance = 800; // Need more clearance for chairs
    
    // Determine table size
    const roomArea = (width * height) / 1_000_000; // sqm
    const table = roomArea >= 10 ? FURNITURE.DINING_TABLE_6 : FURNITURE.DINING_TABLE_4;
    
    // Center the table
    if (width >= table.width + clearance && height >= table.depth + clearance) {
        items.push({
            type: roomArea >= 10 ? 'DINING_TABLE_6' : 'DINING_TABLE_4',
            x: x + (width - table.width) / 2,
            y: y + (height - table.depth) / 2,
            width: table.width,
            depth: table.depth,
            rotation: 0,
            label: table.label
        });
        
        // Add chairs (simplified - just show count)
        const chairCount = roomArea >= 10 ? 6 : 4;
        items.push({
            type: 'DINING_CHAIRS',
            count: chairCount,
            associatedWith: 'dining_table',
            label: `${chairCount} Chairs`
        });
    }
    
    return items;
}

/**
 * Place bathroom sanitary fixtures
 */
function placeBathroomFixtures(room) {
    const items = [];
    const { x, y, width, height } = room;
    const clearance = 150;
    
    // WC - back left corner
    const wc = SANITARY_FIXTURES.WC_WESTERN;
    items.push({
        type: 'WC_WESTERN',
        x: x + clearance,
        y: y + height - clearance - wc.depth,
        width: wc.width,
        depth: wc.depth,
        rotation: 0,
        label: wc.label
    });
    
    // Wash basin - front left
    const basin = SANITARY_FIXTURES.WASH_BASIN;
    items.push({
        type: 'WASH_BASIN',
        x: x + clearance,
        y: y + clearance,
        width: basin.width,
        depth: basin.depth,
        rotation: 0,
        label: basin.label
    });
    
    // Shower area - right side
    const shower = SANITARY_FIXTURES.SHOWER_AREA;
    if (width >= shower.width + wc.width + 3 * clearance) {
        items.push({
            type: 'SHOWER_AREA',
            x: x + width - clearance - shower.width,
            y: y + height - clearance - shower.depth,
            width: shower.width,
            depth: shower.depth,
            rotation: 0,
            label: shower.label
        });
    }
    
    return items;
}

/**
 * Place study room furniture
 */
function placeStudyFurniture(room) {
    const items = [];
    const { x, y, width, height } = room;
    const clearance = 600;
    
    // Study table along wall
    const studyTable = FURNITURE.STUDY_TABLE;
    items.push({
        type: 'STUDY_TABLE',
        x: x + clearance,
        y: y + height - clearance - studyTable.depth,
        width: studyTable.width,
        depth: studyTable.depth,
        rotation: 0,
        label: studyTable.label
    });
    
    // Bookshelf (represented as wardrobe)
    const bookshelf = FURNITURE.WARDROBE;
    items.push({
        type: 'BOOKSHELF',
        x: x + width - clearance - bookshelf.depth,
        y: y + clearance,
        width: bookshelf.depth,
        depth: bookshelf.width,
        rotation: 90,
        label: 'Bookshelf'
    });
    
    return items;
}

/**
 * Check if furniture item overlaps with existing items
 */
export function checkOverlap(item, existingItems, buffer = 100) {
    for (const existing of existingItems) {
        if (rectanglesOverlap(
            item.x - buffer, item.y - buffer, item.width + 2*buffer, item.depth + 2*buffer,
            existing.x - buffer, existing.y - buffer, existing.width + 2*buffer, existing.depth + 2*buffer
        )) {
            return true;
        }
    }
    return false;
}

/**
 * Check if two rectangles overlap
 */
function rectanglesOverlap(x1, y1, w1, h1, x2, y2, w2, h2) {
    return !(x1 + w1 < x2 || x2 + w2 < x1 || y1 + h1 < y2 || y2 + h2 < y1);
}

/**
 * Validate furniture fits within room boundaries
 */
export function validateFurnitureFit(furniture, room) {
    return furniture.x >= room.x &&
           furniture.y >= room.y &&
           furniture.x + furniture.width <= room.x + room.width &&
           furniture.y + furniture.depth <= room.y + room.height;
}

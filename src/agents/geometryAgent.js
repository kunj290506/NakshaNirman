/**
 * Geometry Agent
 * Uses Real-World Reference Templates to generate floor plans
 */

import {
    WALL_THICKNESS,
    MIN_DOOR_WIDTH,
    MIN_WINDOW_WIDTH,
    WINDOW_SILL_HEIGHT
} from '../utils/constants.js';
import {
    getSharedEdge
} from '../utils/geometry.js';
import { FLOOR_PLAN_TEMPLATES } from '../data/floorPlanTemplates.js';
import { generateFurnitureLayout } from '../utils/furniturePlacement.js';

/**
 * Geometry Agent
 * Uses Real-World Reference Templates to generate floor plans
 */
/**
 * Geometry Agent - ML Powered
 * Uses Python backend to predict room dimensions
 */
export async function generateLayout(planData) {
    const { plot, rooms } = planData;

    // Bounds check
    const bounds = {
        x: WALL_THICKNESS,
        y: WALL_THICKNESS,
        width: plot.usableWidth,
        height: plot.usableHeight
    };

    // Prepare Features for ML
    const bedroomCount = rooms.filter(r => r.type === 'bedroom' || r.type === 'master_bedroom').length;
    const bathroomCount = rooms.filter(r => r.type === 'bathroom').length;
    const totalAreaSqft = (plot.usableWidth * plot.usableHeight / 1000000) * 10.764; // mm² -> sqft
    const aspectRatio = plot.width / plot.height;

    // Default dimensions
    let predictions = {
        living_room: { width: 4500, height: 4000 },
        kitchen: { width: 3000, height: 3000 }
    };

    try {
        console.log('Fetching ML Layout prediction...');
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                bedrooms: bedroomCount,
                bathrooms: bathroomCount,
                totalAreaSqft: totalAreaSqft,
                aspectRatio: aspectRatio
            })
        });

        if (response.ok) {
            const result = await response.json();
            if (result.success && result.prediction) {
                // prediction returns generic units (likely based on training data which was mm? No, training data was likely feet or pixels from random generator?)
                // Wait, generate_data.py used width = (area * aspect)**0.5. Area was 800-2500. So units are roughly feet.
                // We need to convert feet to mm for our canvas (1 ft = 304.8 mm).

                const p = result.prediction;
                // Convert ML (Feet) to Canvas (mm)
                predictions.living_room = {
                    width: p.living_room.width * 304.8,
                    height: p.living_room.height * 304.8
                };
                predictions.kitchen = {
                    width: p.kitchen.width * 304.8,
                    height: p.kitchen.height * 304.8
                };
                console.log('ML Prediction received:', predictions);
            }
        }
    } catch (e) {
        console.warn('ML Server unavailable, using defaults:', e);
    }

    // --- BINARY SPACE PARTITIONING (BSP) LAYOUT ---
    // Uses ML Predictions to determine split ratios

    // 1. Recursive Split Function
    function splitBox(box, ratio, dir) {
        if (dir === 'vertical') { // Split vertically (Left/Right)
            const w1 = box.width * ratio;
            return [
                { x: box.x, y: box.y, width: w1, height: box.height },
                { x: box.x + w1, y: box.y, width: box.width - w1, height: box.height }
            ];
        } else { // Split horizontally (Top/Bottom)
            const h1 = box.height * ratio;
            return [
                { x: box.x, y: box.y, width: box.width, height: h1 },
                { x: box.x, y: box.y + h1, width: box.width, height: box.height - h1 }
            ];
        }
    }

    const filledRooms = [];

    // PREPARE ZONES
    // Calculate total predicted areas for weighting
    const pLiving = predictions.living_room.width * predictions.living_room.height;
    const pKitchen = predictions.kitchen.width * predictions.kitchen.height;
    const pBed = 3500 * 3000; // Est
    const pBath = 2000 * 2000; // Est

    const bedCount = rooms.filter(r => r.type.includes('bedroom')).length || 1;
    const totalBedArea = bedCount * pBed;
    const totalPublicArea = pLiving + pKitchen + (pBath * 2); // Roughly

    // ROOT SPLIT: Private vs Public
    // If wider than tall, split left/right. Else top/bottom.
    const rootDir = bounds.width > bounds.height ? 'vertical' : 'horizontal';
    const privateRatio = totalBedArea / (totalBedArea + totalPublicArea);

    const [privateZone, publicZone] = splitBox(bounds, 0.4, rootDir); // Fixed 40/60 split for stability

    // --- PRIVATE ZONE (Bedrooms) ---
    // Split recursively for bedrooms
    let beds = rooms.filter(r => r.type.includes('bedroom'));
    if (beds.length === 0) beds = [{ type: 'bedroom', label: 'Bedroom' }];

    let currentBox = privateZone;
    beds.forEach((bed, i) => {
        // Last bed takes remaining
        if (i === beds.length - 1) {
            filledRooms.push({ ...bed, isPlaced: true, ...currentBox });
        } else {
            // Split current box
            const remaining = beds.length - i;
            const splitRatio = 1 / remaining; // e.g. 1/3, then 1/2
            // Alternate split direction for "Rectangular Spirals" or just stack?
            // Stacking is safer for bedrooms (corridor access)
            const splitDir = rootDir === 'vertical' ? 'horizontal' : 'vertical';

            const [bedBox, rest] = splitBox(currentBox, splitRatio, splitDir);
            filledRooms.push({ ...bed, isPlaced: true, ...bedBox });
            currentBox = rest; // Continue splitting the rest
        }
    });

    // --- PUBLIC ZONE (Living | Kitchen | Service) ---
    // Split Public Zone into Living and Service
    // Living takes main chunk
    const serviceRatio = 0.35; // 35% for Kitchen/Bath
    const publicDir = rootDir; // Same direction as root split? No, usually orthogonal
    // If Root was Vert (Left=Priv, Right=Pub), Split Pub Vertically again? Or Horiz?
    // Let's split "Service Strip" off the end.

    const [livingZone, serviceZone] = splitBox(publicZone, 1 - serviceRatio, rootDir);

    // LIVING ROOM
    const lrReq = rooms.find(r => r.type === 'living_room') || { type: 'living_room' };
    filledRooms.push({ ...lrReq, isPlaced: true, ...livingZone });

    // SERVICE ZONE (Kitchen + Bath)
    // Split Kitchen (Top) vs Baths (Bottom) is usually safe
    const kReq = rooms.find(r => r.type === 'kitchen') || { type: 'kitchen' };
    const baths = rooms.filter(r => r.type === 'bathroom');

    // Kitchen takes top 50%
    const serviceDir = rootDir === 'vertical' ? 'horizontal' : 'vertical';
    const [kBox, bathBox] = splitBox(serviceZone, 0.5, serviceDir);

    filledRooms.push({ ...kReq, isPlaced: true, ...kBox });

    // BATHROOMS
    if (baths.length > 0) {
        let currentBathBox = bathBox;
        baths.forEach((bath, i) => {
            if (i === baths.length - 1) {
                filledRooms.push({ ...bath, isPlaced: true, ...currentBathBox });
            } else {
                const bRatio = 1 / (baths.length - i);
                const [b1, rest] = splitBox(currentBathBox, bRatio, serviceDir);
                filledRooms.push({ ...bath, isPlaced: true, ...b1 });
                currentBathBox = rest;
            }
        });
    } else {
        filledRooms.push({ type: 'bathroom', label: 'Bath', isPlaced: true, ...bathBox });
    }

    // Apply Padding/Shrink for walls
    const shrunkRooms = filledRooms.map(r => ({
        ...r,
        x: r.x + 100,
        y: r.y + 100,
        width: r.width - 200,
        height: r.height - 200
    }));

    const placedRooms = shrunkRooms;

    // Generate Doors and Windows
    const corridor = placedRooms.find(r => r.type === 'corridor');
    const living = placedRooms.find(r => r.type === 'living_room');

    const roomsWithDoors = placedRooms.map(room => {
        if (room.type === 'living_room') {
            // Main entry door logic
            const door = {
                direction: 'south',
                x: room.x + room.width / 2 - MIN_DOOR_WIDTH / 2,
                y: room.y,
                width: MIN_DOOR_WIDTH
            };
            const windows = generateWindows(room, bounds);
            return { ...room, door, windows };
        }
        const target = living;
        const door = generateDoorPosition(room, placedRooms, target, bounds);
        const windows = room.requiresWindow ? generateWindows(room, bounds) : [];
        return { ...room, door, windows };
    });

    // Generate furniture for all rooms
    const allFurniture = [];
    roomsWithDoors.forEach(room => {
        const furniture = generateFurnitureLayout(room);
        allFurniture.push(...furniture);
    });

    return {
        success: true,
        data: {
            status: 'success',
            boundary: { width: plot.width, height: plot.height },
            rooms: roomsWithDoors,
            furniture: allFurniture
        }
    };
}

/**
 * Generate window positions for a room
 */
function generateWindows(room, bounds) {
    const windows = [];
    const windowWidth = MIN_WINDOW_WIDTH;
    
    // Add window on external walls only
    // Check if room is on perimeter
    
    // North wall (if on top edge)
    if (room.y + room.height >= bounds.y + bounds.height - WALL_THICKNESS - 100) {
        windows.push({
            wall: 'north',
            x: room.x + room.width / 2 - windowWidth / 2,
            y: room.y + room.height,
            width: windowWidth,
            sillHeight: WINDOW_SILL_HEIGHT
        });
    }
    
    // South wall (if on bottom edge)
    if (room.y <= bounds.y + WALL_THICKNESS + 100) {
        windows.push({
            wall: 'south',
            x: room.x + room.width / 2 - windowWidth / 2,
            y: room.y,
            width: windowWidth,
            sillHeight: WINDOW_SILL_HEIGHT
        });
    }
    
    // East wall (if on right edge)
    if (room.x + room.width >= bounds.x + bounds.width - WALL_THICKNESS - 100) {
        windows.push({
            wall: 'east',
            x: room.x + room.width,
            y: room.y + room.height / 2 - windowWidth / 2,
            width: windowWidth,
            sillHeight: WINDOW_SILL_HEIGHT
        });
    }
    
    // West wall (if on left edge)
    if (room.x <= bounds.x + WALL_THICKNESS + 100) {
        windows.push({
            wall: 'west',
            x: room.x,
            y: room.y + room.height / 2 - windowWidth / 2,
            width: windowWidth,
            sillHeight: WINDOW_SILL_HEIGHT
        });
    }
    
    // Return first 2 windows max per room
    return windows.slice(0, 2);
}

/**
 * Generate door position (Reused/Simplified)
 */
function generateDoorPosition(room, allRooms, target, bounds) {
    const doorWidth = MIN_DOOR_WIDTH;

    // Try to find shared edge with Target (Hall/Corridor)
    if (target && target !== room) {
        const shared = getSharedEdge(room, target);
        if (shared && shared.length > doorWidth) {
            if (shared.direction === 'vertical') {
                return {
                    direction: shared.x === room.x ? 'west' : 'east',
                    x: shared.x,
                    y: (shared.y1 + shared.y2) / 2 - doorWidth / 2,
                    width: doorWidth
                };
            } else {
                return {
                    direction: shared.y === room.y ? 'south' : 'north',
                    x: (shared.x1 + shared.x2) / 2 - doorWidth / 2,
                    y: shared.y,
                    width: doorWidth
                };
            }
        }
    }

    // Default: internal edge check logic or just place on "inner" side
    // Simplified: Place door on the side closest to center of house
    const cx = bounds.x + bounds.width / 2;
    const cy = bounds.y + bounds.height / 2;
    const rcx = room.x + room.width / 2;
    const rcy = room.y + room.height / 2;

    if (Math.abs(cx - rcx) > Math.abs(cy - rcy)) {
        // More horizontal offset -> Vertical Door (East/West)
        return {
            direction: rcx < cx ? 'east' : 'west',
            x: rcx < cx ? room.x + room.width : room.x,
            y: room.y + room.height / 2 - doorWidth / 2,
            width: doorWidth
        };
    } else {
        // Vertical offset -> Horizontal Door (North/South)
        return {
            direction: rcy < cy ? 'south' : 'north',
            x: room.x + room.width / 2 - doorWidth / 2,
            y: rcy < cy ? room.y + room.height : room.y,
            width: doorWidth
        };
    }
}



/**
 * CAD Agent
 * Generates AutoCAD-compatible DXF files
 */

import Drawing from 'dxf-writer';
import { DXF_LAYERS, WALL_THICKNESS, ROOM_TYPES } from '../utils/constants.js';
import { mmToM } from '../utils/geometry.js';

/**
 * Generate DXF file from geometry data
 * @param {Object} geometryData - Output from Geometry Agent
 * @returns {string} DXF file content as string
 */
export function generateDXF(geometryData) {
    const d = new Drawing();

    // Set up layers
    d.addLayer(DXF_LAYERS.WALLS, Drawing.ACI.WHITE, 'CONTINUOUS');
    d.addLayer(DXF_LAYERS.DOORS, Drawing.ACI.CYAN, 'CONTINUOUS');
    d.addLayer(DXF_LAYERS.ROOMS, Drawing.ACI.GREEN, 'CONTINUOUS');
    d.addLayer(DXF_LAYERS.TEXT, Drawing.ACI.YELLOW, 'CONTINUOUS');
    d.addLayer(DXF_LAYERS.DIMENSIONS, Drawing.ACI.MAGENTA, 'CONTINUOUS');

    const { boundary, rooms } = geometryData;

    // Draw outer boundary walls
    drawBoundaryWalls(d, boundary);

    // Draw rooms
    rooms.forEach(room => {
        drawRoom(d, room);
        drawRoomLabel(d, room);
        if (room.door) {
            drawDoor(d, room, room.door);
        }
    });

    // Draw dimensions
    drawDimensions(d, boundary, rooms);

    return d.toDxfString();
}

/**
 * Draw boundary walls (outer perimeter)
 */
function drawBoundaryWalls(d, boundary) {
    d.setActiveLayer(DXF_LAYERS.WALLS);

    const { width, height } = boundary;
    const t = WALL_THICKNESS;

    // Outer rectangle
    d.drawRect(0, 0, width, height);

    // Inner rectangle (creates wall thickness)
    d.drawRect(t, t, width - 2 * t, height - 2 * t);
}

/**
 * Draw a room rectangle
 */
function drawRoom(d, room) {
    d.setActiveLayer(DXF_LAYERS.ROOMS);

    // Draw room outline
    d.drawRect(room.x, room.y, room.width, room.height);

    // Draw inner walls between rooms
    d.setActiveLayer(DXF_LAYERS.WALLS);

    // Top wall
    d.drawLine(room.x, room.y + room.height, room.x + room.width, room.y + room.height);

    // Right wall
    d.drawLine(room.x + room.width, room.y, room.x + room.width, room.y + room.height);
}

/**
 * Draw room label (name and area)
 */
function drawRoomLabel(d, room) {
    d.setActiveLayer(DXF_LAYERS.TEXT);

    const centerX = room.x + room.width / 2;
    const centerY = room.y + room.height / 2;

    // Room name
    const textHeight = Math.min(room.width, room.height) / 10;
    const clampedHeight = Math.max(150, Math.min(300, textHeight));

    d.drawText(centerX, centerY + clampedHeight / 2, clampedHeight, 0, room.label);

    // Area
    const areaSqm = (room.width * room.height) / 1_000_000;
    const areaText = `${areaSqm.toFixed(1)} m²`;
    d.drawText(centerX, centerY - clampedHeight, clampedHeight * 0.7, 0, areaText);
}

/**
 * Draw door symbol
 */
function drawDoor(d, room, door) {
    d.setActiveLayer(DXF_LAYERS.DOORS);

    const { direction, x, y, width } = door;

    // Door opening (line)
    switch (direction) {
        case 'north':
        case 'south':
            d.drawLine(x, y, x + width, y);
            // Door swing arc
            drawDoorArc(d, x, y, width, direction);
            break;
        case 'east':
        case 'west':
            d.drawLine(x, y, x, y + width);
            // Door swing arc
            drawDoorArc(d, x, y, width, direction);
            break;
    }
}

/**
 * Draw door swing arc
 */
function drawDoorArc(d, x, y, width, direction) {
    // Simplified arc representation using lines
    const segments = 8;
    const radius = width;

    let startAngle, endAngle, pivotX, pivotY;

    switch (direction) {
        case 'north':
            pivotX = x;
            pivotY = y;
            startAngle = 0;
            endAngle = Math.PI / 2;
            break;
        case 'south':
            pivotX = x;
            pivotY = y;
            startAngle = -Math.PI / 2;
            endAngle = 0;
            break;
        case 'east':
            pivotX = x;
            pivotY = y;
            startAngle = Math.PI / 2;
            endAngle = Math.PI;
            break;
        case 'west':
            pivotX = x;
            pivotY = y;
            startAngle = 0;
            endAngle = Math.PI / 2;
            break;
    }

    // Draw arc as line segments
    for (let i = 0; i < segments; i++) {
        const angle1 = startAngle + (endAngle - startAngle) * (i / segments);
        const angle2 = startAngle + (endAngle - startAngle) * ((i + 1) / segments);

        const x1 = pivotX + radius * Math.cos(angle1);
        const y1 = pivotY + radius * Math.sin(angle1);
        const x2 = pivotX + radius * Math.cos(angle2);
        const y2 = pivotY + radius * Math.sin(angle2);

        d.drawLine(x1, y1, x2, y2);
    }
}

/**
 * Draw dimensions
 */
function drawDimensions(d, boundary, rooms) {
    d.setActiveLayer(DXF_LAYERS.DIMENSIONS);

    const offset = 500; // Dimension offset from walls
    const textHeight = 200;

    // Overall width dimension
    const widthM = mmToM(boundary.width).toFixed(2);
    d.drawText(
        boundary.width / 2,
        -offset,
        textHeight,
        0,
        `${widthM} m`
    );

    // Overall height dimension  
    const heightM = mmToM(boundary.height).toFixed(2);
    d.drawText(
        -offset,
        boundary.height / 2,
        textHeight,
        90,
        `${heightM} m`
    );

    // Extension lines
    d.drawLine(0, -offset / 2, boundary.width, -offset / 2);
    d.drawLine(-offset / 2, 0, -offset / 2, boundary.height);
}

/**
 * Download DXF file
 * @param {string} dxfContent - DXF file content
 * @param {string} filename - Output filename
 */
export function downloadDXF(dxfContent, filename = 'floor_plan.dxf') {
    const blob = new Blob([dxfContent], { type: 'application/dxf' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
}

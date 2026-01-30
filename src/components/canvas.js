/**
 * Canvas Renderer
 * Professional architectural floor plan rendering
 */

import { WALL_THICKNESS, ROOM_TYPES, CANVAS_PADDING, CANVAS_MIN_ZOOM, CANVAS_MAX_ZOOM } from '../utils/constants.js';
import { mmToM } from '../utils/geometry.js';

export class CanvasRenderer {
    constructor(canvasElement, containerElement) {
        this.canvas = canvasElement;
        this.container = containerElement;
        this.ctx = canvasElement.getContext('2d');

        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.scale = 1;

        this.geometryData = null;

        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        this.resize = this.resize.bind(this);
        this.handleMouseDown = this.handleMouseDown.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);
        this.handleWheel = this.handleWheel.bind(this);

        this.setupEventListeners();
        this.resize();
    }

    setupEventListeners() {
        window.addEventListener('resize', this.resize);
        this.canvas.addEventListener('mousedown', this.handleMouseDown);
        this.canvas.addEventListener('mousemove', this.handleMouseMove);
        this.canvas.addEventListener('mouseup', this.handleMouseUp);
        this.canvas.addEventListener('mouseleave', this.handleMouseUp);
        this.canvas.addEventListener('wheel', this.handleWheel, { passive: false });
    }

    destroy() {
        window.removeEventListener('resize', this.resize);
        this.canvas.removeEventListener('mousedown', this.handleMouseDown);
        this.canvas.removeEventListener('mousemove', this.handleMouseMove);
        this.canvas.removeEventListener('mouseup', this.handleMouseUp);
        this.canvas.removeEventListener('mouseleave', this.handleMouseUp);
        this.canvas.removeEventListener('wheel', this.handleWheel);
    }

    resize() {
        const rect = this.container.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;

        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.canvas.style.width = `${rect.width}px`;
        this.canvas.style.height = `${rect.height}px`;

        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.scale(dpr, dpr);

        this.render();
    }

    handleMouseDown(e) {
        this.isDragging = true;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
        this.canvas.style.cursor = 'grabbing';
    }

    handleMouseMove(e) {
        if (!this.isDragging) return;

        this.panX += e.clientX - this.lastMouseX;
        this.panY += e.clientY - this.lastMouseY;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;

        this.render();
    }

    handleMouseUp() {
        this.isDragging = false;
        this.canvas.style.cursor = 'grab';
    }

    handleWheel(e) {
        e.preventDefault();

        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        const newZoom = Math.max(CANVAS_MIN_ZOOM, Math.min(CANVAS_MAX_ZOOM, this.zoom * delta));

        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        this.panX = mouseX - (mouseX - this.panX) * (newZoom / this.zoom);
        this.panY = mouseY - (mouseY - this.panY) * (newZoom / this.zoom);

        this.zoom = newZoom;
        this.render();
    }

    zoomIn() {
        this.zoom = Math.min(CANVAS_MAX_ZOOM, this.zoom * 1.2);
        this.render();
    }

    zoomOut() {
        this.zoom = Math.max(CANVAS_MIN_ZOOM, this.zoom * 0.8);
        this.render();
    }

    resetView() {
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.fitToView();
        this.render();
    }

    fitToView() {
        if (!this.geometryData) return;

        const rect = this.container.getBoundingClientRect();
        const { boundary } = this.geometryData;

        const padding = 80;
        const scaleX = (rect.width - padding * 2) / boundary.width;
        const scaleY = (rect.height - padding * 2) / boundary.height;

        this.scale = Math.min(scaleX, scaleY);
        this.zoom = 1;

        const scaledWidth = boundary.width * this.scale;
        const scaledHeight = boundary.height * this.scale;

        this.panX = (rect.width - scaledWidth) / 2;
        this.panY = (rect.height - scaledHeight) / 2;
    }

    setGeometry(geometryData) {
        this.geometryData = geometryData;
        this.fitToView();
        this.render();
    }

    clear() {
        this.geometryData = null;
        this.render();
    }

    render() {
        const rect = this.container.getBoundingClientRect();
        const ctx = this.ctx;

        // Clear with white background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, rect.width, rect.height);

        if (!this.geometryData) return;

        ctx.save();
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.scale * this.zoom, this.scale * this.zoom);

        // Draw everything
        this.drawBoundary(ctx);
        this.drawRooms(ctx);
        this.drawFurniture(ctx);
        this.drawDoors(ctx);
        this.drawDimensions(ctx);

        ctx.restore();
    }

    drawBoundary(ctx) {
        const { boundary } = this.geometryData;
        const wallT = WALL_THICKNESS;

        // Outer wall (thick black)
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, boundary.width, wallT); // Top
        ctx.fillRect(0, boundary.height - wallT, boundary.width, wallT); // Bottom
        ctx.fillRect(0, 0, wallT, boundary.height); // Left
        ctx.fillRect(boundary.width - wallT, 0, wallT, boundary.height); // Right
    }

    drawRooms(ctx) {
        const { rooms, boundary } = this.geometryData;

        rooms.forEach(room => {
            // Room fill
            ctx.fillStyle = '#f5f5f5';
            ctx.fillRect(room.x, room.y, room.width, room.height);

            // Room walls (thinner than boundary)
            ctx.strokeStyle = '#333333';
            ctx.lineWidth = 80;
            ctx.strokeRect(room.x, room.y, room.width, room.height);

            // Room label
            const fontSize = Math.min(room.width, room.height) * 0.12;
            ctx.fillStyle = '#000000';
            ctx.font = `bold ${fontSize}px Arial`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            const cx = room.x + room.width / 2;
            const cy = room.y + room.height / 2;

            ctx.fillText(room.label.toUpperCase(), cx, cy - fontSize * 0.5);

            // Area label
            const areaSqm = (room.width * room.height) / 1000000;
            const areaSqft = Math.round(areaSqm * 10.764);
            ctx.font = `${fontSize * 0.7}px Arial`;
            ctx.fillStyle = '#666666';
            ctx.fillText(`${areaSqft} sq.ft`, cx, cy + fontSize * 0.6);
        });
    }

    drawFurniture(ctx) {
        const { rooms } = this.geometryData;

        ctx.strokeStyle = '#444444';
        ctx.lineWidth = 40;

        rooms.forEach(room => {
            const padding = Math.min(room.width, room.height) * 0.1;
            const fw = room.width - padding * 2;
            const fh = room.height - padding * 2;
            const fx = room.x + padding;
            const fy = room.y + padding;

            switch (room.type) {
                case 'bedroom':
                case 'master_bedroom':
                    // Bed
                    const bedW = fw * 0.7;
                    const bedH = fh * 0.5;
                    const bedX = fx + (fw - bedW) / 2;
                    const bedY = fy + fh - bedH - padding;
                    ctx.strokeRect(bedX, bedY, bedW, bedH);
                    // Pillows
                    ctx.strokeRect(bedX + bedW * 0.1, bedY + 50, bedW * 0.35, bedH * 0.25);
                    ctx.strokeRect(bedX + bedW * 0.55, bedY + 50, bedW * 0.35, bedH * 0.25);
                    break;

                case 'living_room':
                    // Sofa
                    const sofaW = fw * 0.6;
                    const sofaH = fh * 0.25;
                    const sofaX = fx + (fw - sofaW) / 2;
                    ctx.strokeRect(sofaX, fy + fh - sofaH - padding, sofaW, sofaH);
                    // Coffee table
                    const tableW = sofaW * 0.4;
                    const tableH = sofaH * 0.5;
                    ctx.strokeRect(fx + (fw - tableW) / 2, fy + fh * 0.4, tableW, tableH);
                    break;

                case 'kitchen':
                    // Counter L-shape
                    const counterD = Math.min(fw, fh) * 0.2;
                    ctx.fillStyle = '#e0e0e0';
                    ctx.fillRect(fx, fy, counterD, fh);
                    ctx.fillRect(fx, fy + fh - counterD, fw, counterD);
                    ctx.strokeRect(fx, fy, counterD, fh);
                    ctx.strokeRect(fx, fy + fh - counterD, fw, counterD);
                    // Sink circle
                    ctx.beginPath();
                    ctx.arc(fx + fw * 0.5, fy + fh - counterD * 0.5, counterD * 0.3, 0, Math.PI * 2);
                    ctx.stroke();
                    break;

                case 'dining_room':
                    // Dining table
                    const dtW = fw * 0.5;
                    const dtH = fh * 0.4;
                    const dtX = fx + (fw - dtW) / 2;
                    const dtY = fy + (fh - dtH) / 2;
                    ctx.strokeRect(dtX, dtY, dtW, dtH);
                    // Chairs
                    const chairS = Math.min(dtW, dtH) * 0.25;
                    ctx.strokeRect(dtX + dtW * 0.2, dtY - chairS - 50, chairS, chairS);
                    ctx.strokeRect(dtX + dtW * 0.6, dtY - chairS - 50, chairS, chairS);
                    ctx.strokeRect(dtX + dtW * 0.2, dtY + dtH + 50, chairS, chairS);
                    ctx.strokeRect(dtX + dtW * 0.6, dtY + dtH + 50, chairS, chairS);
                    break;

                case 'bathroom':
                    // Toilet
                    const toiletW = fw * 0.35;
                    const toiletH = fh * 0.3;
                    ctx.strokeRect(fx + 50, fy + 50, toiletW, toiletH * 0.5);
                    ctx.beginPath();
                    ctx.ellipse(fx + 50 + toiletW * 0.5, fy + 50 + toiletH * 0.7, toiletW * 0.4, toiletH * 0.3, 0, 0, Math.PI * 2);
                    ctx.stroke();
                    // Sink
                    ctx.strokeRect(fx + fw - toiletW - 50, fy + 50, toiletW, toiletH * 0.6);
                    break;

                case 'parking':
                    // Car outline
                    const carW = fw * 0.6;
                    const carH = fh * 0.8;
                    const carX = fx + (fw - carW) / 2;
                    const carY = fy + (fh - carH) / 2;
                    ctx.strokeRect(carX, carY, carW, carH);
                    ctx.beginPath();
                    ctx.moveTo(carX, carY + carH * 0.25);
                    ctx.lineTo(carX + carW, carY + carH * 0.25);
                    ctx.moveTo(carX, carY + carH * 0.75);
                    ctx.lineTo(carX + carW, carY + carH * 0.75);
                    ctx.stroke();
                    break;

                case 'balcony':
                    // Diagonal lines pattern
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(room.x, room.y, room.width, room.height);
                    ctx.clip();
                    ctx.strokeStyle = '#cccccc';
                    ctx.lineWidth = 20;
                    const step = Math.min(room.width, room.height) * 0.15;
                    for (let i = -room.height; i < room.width + room.height; i += step) {
                        ctx.beginPath();
                        ctx.moveTo(room.x + i, room.y);
                        ctx.lineTo(room.x + i + room.height, room.y + room.height);
                        ctx.stroke();
                    }
                    ctx.restore();
                    break;

                case 'study':
                    // Desk
                    const deskW = fw * 0.5;
                    const deskH = fh * 0.25;
                    ctx.strokeRect(fx + (fw - deskW) / 2, fy + fh * 0.2, deskW, deskH);
                    // Chair
                    const chW = deskW * 0.3;
                    ctx.strokeRect(fx + (fw - chW) / 2, fy + fh * 0.55, chW, chW);
                    break;
            }
        });
    }

    drawDoors(ctx) {
        const { rooms } = this.geometryData;

        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 40;

        rooms.forEach(room => {
            if (!room.door) return;

            const { door } = room;
            const dw = door.width * 0.8;

            // Clear door opening
            ctx.fillStyle = '#ffffff';

            switch (door.direction) {
                case 'south':
                    ctx.fillRect(door.x, door.y - 60, dw, 120);
                    // Door swing arc
                    ctx.beginPath();
                    ctx.arc(door.x, door.y, dw, 0, Math.PI / 2);
                    ctx.stroke();
                    break;
                case 'north':
                    ctx.fillRect(door.x, door.y - 60, dw, 120);
                    ctx.beginPath();
                    ctx.arc(door.x + dw, door.y, dw, Math.PI, Math.PI * 1.5);
                    ctx.stroke();
                    break;
                case 'west':
                    ctx.fillRect(door.x - 60, door.y, 120, dw);
                    ctx.beginPath();
                    ctx.arc(door.x, door.y, dw, Math.PI / 2, Math.PI);
                    ctx.stroke();
                    break;
                case 'east':
                    ctx.fillRect(door.x - 60, door.y, 120, dw);
                    ctx.beginPath();
                    ctx.arc(door.x, door.y + dw, dw, -Math.PI / 2, 0);
                    ctx.stroke();
                    break;
            }
        });
    }

    drawDimensions(ctx) {
        const { boundary } = this.geometryData;

        // Convert mm to feet (1 foot = 304.8 mm)
        const widthFt = (boundary.width / 304.8).toFixed(0);
        const heightFt = (boundary.height / 304.8).toFixed(0);

        const fontSize = Math.min(boundary.width, boundary.height) * 0.03;
        ctx.fillStyle = '#000000';
        ctx.font = `${fontSize}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Top dimension
        ctx.fillText(`${widthFt}'`, boundary.width / 2, -fontSize * 1.5);

        // Left dimension (rotated)
        ctx.save();
        ctx.translate(-fontSize * 1.5, boundary.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(`${heightFt}'`, 0, 0);
        ctx.restore();

        // Dimension lines
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 20;

        // Top line with end marks
        ctx.beginPath();
        ctx.moveTo(0, -fontSize * 0.8);
        ctx.lineTo(boundary.width, -fontSize * 0.8);
        ctx.stroke();

        // Left line with end marks
        ctx.beginPath();
        ctx.moveTo(-fontSize * 0.8, 0);
        ctx.lineTo(-fontSize * 0.8, boundary.height);
        ctx.stroke();
    }
}

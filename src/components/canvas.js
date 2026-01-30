/**
 * Canvas Renderer
 * Professional architectural floor plan rendering
 * Features: furniture symbols, door swings, dimensions, black-white style
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

        const scaleX = (rect.width - CANVAS_PADDING * 4) / boundary.width;
        const scaleY = (rect.height - CANVAS_PADDING * 4) / boundary.height;

        this.scale = Math.min(scaleX, scaleY);
        this.zoom = 1;

        const scaledWidth = boundary.width * this.scale;
        const scaledHeight = boundary.height * this.scale;

        this.panX = (rect.width - scaledWidth) / 2;
        this.panY = (rect.height - scaledHeight) / 2 + CANVAS_PADDING;
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

        // Draw in architectural style (no Y flip for simplicity)
        this.drawBoundary(ctx);
        this.drawRooms(ctx);
        this.drawFurniture(ctx);
        this.drawDoors(ctx);
        this.drawLabels(ctx);
        this.drawDimensions(ctx);

        ctx.restore();
    }

    drawBoundary(ctx) {
        const { boundary } = this.geometryData;
        const t = WALL_THICKNESS;
        const lineWidth = 3 / (this.scale * this.zoom);

        // Outer boundary (thick black line)
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = lineWidth * 2;
        ctx.strokeRect(0, 0, boundary.width, boundary.height);

        // Inner walls
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(t, t, boundary.width - 2 * t, boundary.height - 2 * t);
    }

    drawRooms(ctx) {
        const { rooms } = this.geometryData;
        const lineWidth = 2 / (this.scale * this.zoom);

        rooms.forEach(room => {
            // Light gray fill for all rooms (architectural style)
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(room.x, room.y, room.width, room.height);

            // Room walls
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = lineWidth;
            ctx.strokeRect(room.x, room.y, room.width, room.height);
        });
    }

    drawFurniture(ctx) {
        const { rooms } = this.geometryData;
        const lineWidth = 1 / (this.scale * this.zoom);

        ctx.strokeStyle = '#333333';
        ctx.lineWidth = lineWidth;

        rooms.forEach(room => {
            const cx = room.x + room.width / 2;
            const cy = room.y + room.height / 2;
            const w = room.width;
            const h = room.height;

            switch (room.type) {
                case 'bedroom':
                case 'master_bedroom':
                    this.drawBed(ctx, room);
                    break;
                case 'living_room':
                    this.drawSofa(ctx, room);
                    break;
                case 'kitchen':
                    this.drawKitchenCounter(ctx, room);
                    break;
                case 'dining_room':
                    this.drawDiningTable(ctx, room);
                    break;
                case 'bathroom':
                    this.drawBathroom(ctx, room);
                    break;
                case 'parking':
                    this.drawCar(ctx, room);
                    break;
                case 'balcony':
                    this.drawBalconyPattern(ctx, room);
                    break;
                case 'study':
                    this.drawDesk(ctx, room);
                    break;
            }
        });
    }

    drawBed(ctx, room) {
        const padding = Math.min(room.width, room.height) * 0.15;
        const bedW = room.width - padding * 2;
        const bedH = room.height - padding * 2;
        const x = room.x + padding;
        const y = room.y + padding;

        // Bed frame
        ctx.strokeRect(x, y, bedW, bedH);

        // Pillows
        const pillowH = bedH * 0.2;
        ctx.strokeRect(x + bedW * 0.1, y + bedH * 0.05, bedW * 0.35, pillowH);
        ctx.strokeRect(x + bedW * 0.55, y + bedH * 0.05, bedW * 0.35, pillowH);

        // Blanket line
        ctx.beginPath();
        ctx.moveTo(x, y + bedH * 0.35);
        ctx.lineTo(x + bedW, y + bedH * 0.35);
        ctx.stroke();
    }

    drawSofa(ctx, room) {
        const padding = Math.min(room.width, room.height) * 0.2;
        const sofaW = room.width * 0.6;
        const sofaH = room.height * 0.3;
        const x = room.x + (room.width - sofaW) / 2;
        const y = room.y + padding;

        // Sofa back
        ctx.strokeRect(x, y, sofaW, sofaH * 0.4);
        // Sofa seat
        ctx.strokeRect(x, y + sofaH * 0.4, sofaW, sofaH * 0.6);

        // Coffee table
        const tableW = sofaW * 0.5;
        const tableH = sofaH * 0.4;
        ctx.strokeRect(x + (sofaW - tableW) / 2, y + sofaH + padding / 2, tableW, tableH);
    }

    drawKitchenCounter(ctx, room) {
        const counterDepth = Math.min(room.width, room.height) * 0.25;

        // L-shaped counter along two walls
        // Bottom counter
        ctx.fillStyle = '#e5e7eb';
        ctx.fillRect(room.x, room.y + room.height - counterDepth, room.width, counterDepth);
        ctx.strokeRect(room.x, room.y + room.height - counterDepth, room.width, counterDepth);

        // Side counter
        ctx.fillRect(room.x, room.y, counterDepth, room.height - counterDepth);
        ctx.strokeRect(room.x, room.y, counterDepth, room.height - counterDepth);

        // Sink (circle)
        const sinkX = room.x + room.width / 2;
        const sinkY = room.y + room.height - counterDepth / 2;
        const sinkR = counterDepth * 0.3;
        ctx.beginPath();
        ctx.arc(sinkX, sinkY, sinkR, 0, Math.PI * 2);
        ctx.stroke();

        // Stove (4 circles)
        const stoveX = room.x + counterDepth / 2;
        const stoveY = room.y + room.height * 0.4;
        const burnerR = counterDepth * 0.15;
        ctx.beginPath();
        ctx.arc(stoveX - burnerR, stoveY - burnerR, burnerR, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(stoveX + burnerR, stoveY - burnerR, burnerR, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(stoveX - burnerR, stoveY + burnerR, burnerR, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(stoveX + burnerR, stoveY + burnerR, burnerR, 0, Math.PI * 2);
        ctx.stroke();
    }

    drawDiningTable(ctx, room) {
        const tableW = room.width * 0.5;
        const tableH = room.height * 0.4;
        const x = room.x + (room.width - tableW) / 2;
        const y = room.y + (room.height - tableH) / 2;

        // Table
        ctx.strokeRect(x, y, tableW, tableH);

        // Chairs
        const chairSize = Math.min(tableW, tableH) * 0.2;
        // Top chairs
        ctx.strokeRect(x + tableW * 0.2, y - chairSize - 100, chairSize, chairSize);
        ctx.strokeRect(x + tableW * 0.6, y - chairSize - 100, chairSize, chairSize);
        // Bottom chairs
        ctx.strokeRect(x + tableW * 0.2, y + tableH + 100, chairSize, chairSize);
        ctx.strokeRect(x + tableW * 0.6, y + tableH + 100, chairSize, chairSize);
    }

    drawBathroom(ctx, room) {
        const padding = Math.min(room.width, room.height) * 0.1;

        // Toilet
        const toiletW = room.width * 0.3;
        const toiletH = room.height * 0.35;
        const toiletX = room.x + padding;
        const toiletY = room.y + padding;

        ctx.strokeRect(toiletX, toiletY, toiletW, toiletH * 0.6);
        ctx.beginPath();
        ctx.ellipse(toiletX + toiletW / 2, toiletY + toiletH * 0.8, toiletW / 2, toiletH * 0.25, 0, 0, Math.PI * 2);
        ctx.stroke();

        // Sink
        const sinkW = room.width * 0.25;
        const sinkH = room.height * 0.2;
        ctx.strokeRect(room.x + room.width - sinkW - padding, room.y + padding, sinkW, sinkH);
        ctx.beginPath();
        ctx.ellipse(
            room.x + room.width - sinkW / 2 - padding,
            room.y + padding + sinkH / 2,
            sinkW * 0.35,
            sinkH * 0.35,
            0, 0, Math.PI * 2
        );
        ctx.stroke();

        // Shower area (dashed lines)
        ctx.setLineDash([100, 100]);
        ctx.strokeRect(
            room.x + padding,
            room.y + room.height - room.height * 0.4,
            room.width * 0.5,
            room.height * 0.35
        );
        ctx.setLineDash([]);
    }

    drawCar(ctx, room) {
        const carW = room.width * 0.7;
        const carH = room.height * 0.85;
        const x = room.x + (room.width - carW) / 2;
        const y = room.y + (room.height - carH) / 2;

        // Car body
        ctx.strokeRect(x, y, carW, carH);

        // Hood
        ctx.beginPath();
        ctx.moveTo(x, y + carH * 0.2);
        ctx.lineTo(x + carW, y + carH * 0.2);
        ctx.stroke();

        // Trunk
        ctx.beginPath();
        ctx.moveTo(x, y + carH * 0.8);
        ctx.lineTo(x + carW, y + carH * 0.8);
        ctx.stroke();

        // Wheels
        const wheelW = carW * 0.15;
        const wheelH = carH * 0.1;
        ctx.fillStyle = '#333333';
        ctx.fillRect(x - wheelW / 2, y + carH * 0.25, wheelW, wheelH);
        ctx.fillRect(x + carW - wheelW / 2, y + carH * 0.25, wheelW, wheelH);
        ctx.fillRect(x - wheelW / 2, y + carH * 0.65, wheelW, wheelH);
        ctx.fillRect(x + carW - wheelW / 2, y + carH * 0.65, wheelW, wheelH);
    }

    drawBalconyPattern(ctx, room) {
        // Diagonal lines pattern for open area
        const step = 300;
        ctx.save();
        ctx.beginPath();
        ctx.rect(room.x, room.y, room.width, room.height);
        ctx.clip();

        ctx.strokeStyle = '#cccccc';
        for (let i = -room.height; i < room.width + room.height; i += step) {
            ctx.beginPath();
            ctx.moveTo(room.x + i, room.y);
            ctx.lineTo(room.x + i + room.height, room.y + room.height);
            ctx.stroke();
        }
        ctx.restore();
    }

    drawDesk(ctx, room) {
        const deskW = room.width * 0.6;
        const deskH = room.height * 0.3;
        const x = room.x + (room.width - deskW) / 2;
        const y = room.y + room.height * 0.2;

        // Desk
        ctx.strokeRect(x, y, deskW, deskH);

        // Chair
        const chairSize = deskH * 0.6;
        ctx.strokeRect(x + (deskW - chairSize) / 2, y + deskH + 200, chairSize, chairSize);
    }

    drawDoors(ctx) {
        const { rooms } = this.geometryData;
        const lineWidth = 2 / (this.scale * this.zoom);

        ctx.strokeStyle = '#000000';
        ctx.lineWidth = lineWidth;

        rooms.forEach(room => {
            if (!room.door) return;

            const { door } = room;
            const doorWidth = door.width;
            const swingRadius = doorWidth;

            // Clear door opening (white)
            ctx.fillStyle = '#ffffff';

            switch (door.direction) {
                case 'south':
                    // Door opening
                    ctx.fillRect(door.x, door.y - 50, doorWidth, 150);
                    // Door swing arc
                    ctx.beginPath();
                    ctx.moveTo(door.x, door.y);
                    ctx.arc(door.x, door.y, swingRadius, 0, Math.PI / 2);
                    ctx.stroke();
                    // Door line
                    ctx.beginPath();
                    ctx.moveTo(door.x, door.y);
                    ctx.lineTo(door.x + doorWidth, door.y);
                    ctx.stroke();
                    break;

                case 'north':
                    ctx.fillRect(door.x, door.y - 100, doorWidth, 150);
                    ctx.beginPath();
                    ctx.moveTo(door.x + doorWidth, door.y);
                    ctx.arc(door.x + doorWidth, door.y, swingRadius, Math.PI, Math.PI * 1.5);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(door.x, door.y);
                    ctx.lineTo(door.x + doorWidth, door.y);
                    ctx.stroke();
                    break;

                case 'west':
                    ctx.fillRect(door.x - 50, door.y, 150, doorWidth);
                    ctx.beginPath();
                    ctx.moveTo(door.x, door.y);
                    ctx.arc(door.x, door.y, swingRadius, Math.PI / 2, Math.PI);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(door.x, door.y);
                    ctx.lineTo(door.x, door.y + doorWidth);
                    ctx.stroke();
                    break;

                case 'east':
                    ctx.fillRect(door.x - 100, door.y, 150, doorWidth);
                    ctx.beginPath();
                    ctx.moveTo(door.x, door.y + doorWidth);
                    ctx.arc(door.x, door.y + doorWidth, swingRadius, -Math.PI / 2, 0);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(door.x, door.y);
                    ctx.lineTo(door.x, door.y + doorWidth);
                    ctx.stroke();
                    break;
            }
        });
    }

    drawLabels(ctx) {
        const { rooms } = this.geometryData;
        const baseSize = 400 / (this.scale * this.zoom);

        ctx.fillStyle = '#000000';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        rooms.forEach(room => {
            const cx = room.x + room.width / 2;
            const cy = room.y + room.height / 2;

            // Room name
            ctx.font = `bold ${baseSize}px Arial, sans-serif`;
            ctx.fillText(room.label.toUpperCase(), cx, cy - baseSize * 0.6);

            // Area in sq ft
            const areaSqm = (room.width * room.height) / 1_000_000;
            const areaSqft = Math.round(areaSqm / 0.0929);
            ctx.font = `${baseSize * 0.7}px Arial, sans-serif`;
            ctx.fillText(`${areaSqft} sq ft`, cx, cy + baseSize * 0.5);
        });
    }

    drawDimensions(ctx) {
        const { boundary } = this.geometryData;
        const baseSize = 350 / (this.scale * this.zoom);
        const offset = 600;

        ctx.strokeStyle = '#000000';
        ctx.fillStyle = '#000000';
        ctx.lineWidth = 1 / (this.scale * this.zoom);
        ctx.font = `${baseSize}px Arial, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Convert to feet
        const widthFt = (boundary.width / 304.8).toFixed(1);
        const heightFt = (boundary.height / 304.8).toFixed(1);

        // Top dimension line
        ctx.beginPath();
        ctx.moveTo(0, -offset);
        ctx.lineTo(boundary.width, -offset);
        ctx.stroke();

        // Arrows
        this.drawArrow(ctx, 0, -offset, 'right');
        this.drawArrow(ctx, boundary.width, -offset, 'left');

        ctx.fillText(`${widthFt}'`, boundary.width / 2, -offset - baseSize);

        // Left dimension line
        ctx.beginPath();
        ctx.moveTo(-offset, 0);
        ctx.lineTo(-offset, boundary.height);
        ctx.stroke();

        this.drawArrow(ctx, -offset, 0, 'down');
        this.drawArrow(ctx, -offset, boundary.height, 'up');

        ctx.save();
        ctx.translate(-offset - baseSize, boundary.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(`${heightFt}'`, 0, 0);
        ctx.restore();
    }

    drawArrow(ctx, x, y, direction) {
        const size = 150;
        ctx.beginPath();

        switch (direction) {
            case 'right':
                ctx.moveTo(x, y);
                ctx.lineTo(x + size, y - size / 2);
                ctx.lineTo(x + size, y + size / 2);
                break;
            case 'left':
                ctx.moveTo(x, y);
                ctx.lineTo(x - size, y - size / 2);
                ctx.lineTo(x - size, y + size / 2);
                break;
            case 'up':
                ctx.moveTo(x, y);
                ctx.lineTo(x - size / 2, y - size);
                ctx.lineTo(x + size / 2, y - size);
                break;
            case 'down':
                ctx.moveTo(x, y);
                ctx.lineTo(x - size / 2, y + size);
                ctx.lineTo(x + size / 2, y + size);
                break;
        }

        ctx.closePath();
        ctx.fill();
    }
}

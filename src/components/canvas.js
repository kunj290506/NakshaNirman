/**
 * Canvas Renderer
 * Renders floor plan on HTML5 Canvas
 */

import { WALL_THICKNESS, ROOM_TYPES, CANVAS_PADDING, CANVAS_MIN_ZOOM, CANVAS_MAX_ZOOM } from '../utils/constants.js';
import { mmToM } from '../utils/geometry.js';

export class CanvasRenderer {
    constructor(canvasElement, containerElement) {
        this.canvas = canvasElement;
        this.container = containerElement;
        this.ctx = canvasElement.getContext('2d');

        // View state
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.scale = 1; // mm to pixels

        // Geometry data
        this.geometryData = null;

        // Interaction state
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        // Bind methods
        this.resize = this.resize.bind(this);
        this.handleMouseDown = this.handleMouseDown.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);
        this.handleWheel = this.handleWheel.bind(this);

        // Set up event listeners
        this.setupEventListeners();

        // Initial resize
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

        const dx = e.clientX - this.lastMouseX;
        const dy = e.clientY - this.lastMouseY;

        this.panX += dx;
        this.panY += dy;

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

        // Zoom towards mouse position
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

        const scaleX = (rect.width - CANVAS_PADDING * 2) / boundary.width;
        const scaleY = (rect.height - CANVAS_PADDING * 2) / boundary.height;

        this.scale = Math.min(scaleX, scaleY);
        this.zoom = 1;

        // Center the floor plan
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

        // Clear canvas
        ctx.clearRect(0, 0, rect.width, rect.height);

        if (!this.geometryData) {
            return;
        }

        ctx.save();

        // Apply transformations
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.scale * this.zoom, this.scale * this.zoom);

        // Flip Y axis (CAD coordinates have Y up, canvas has Y down)
        ctx.translate(0, this.geometryData.boundary.height);
        ctx.scale(1, -1);

        // Draw grid
        this.drawGrid(ctx);

        // Draw boundary
        this.drawBoundary(ctx);

        // Draw rooms
        this.drawRooms(ctx);

        // Draw doors
        this.drawDoors(ctx);

        // Draw labels (need to flip text)
        ctx.save();
        ctx.scale(1, -1);
        this.drawLabels(ctx);
        ctx.restore();

        ctx.restore();
    }

    drawGrid(ctx) {
        const { boundary } = this.geometryData;
        const gridSize = 1000; // 1m grid

        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 1 / (this.scale * this.zoom);

        // Vertical lines
        for (let x = 0; x <= boundary.width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, boundary.height);
            ctx.stroke();
        }

        // Horizontal lines
        for (let y = 0; y <= boundary.height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(boundary.width, y);
            ctx.stroke();
        }
    }

    drawBoundary(ctx) {
        const { boundary } = this.geometryData;
        const t = WALL_THICKNESS;

        // Outer wall fill
        ctx.fillStyle = '#374151';
        ctx.fillRect(0, 0, boundary.width, boundary.height);

        // Inner area (cut out)
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(t, t, boundary.width - 2 * t, boundary.height - 2 * t);
    }

    drawRooms(ctx) {
        const { rooms } = this.geometryData;

        rooms.forEach(room => {
            // Room fill
            const color = room.color || ROOM_TYPES[room.type]?.color || '#f1f5f9';
            ctx.fillStyle = color;
            ctx.fillRect(room.x, room.y, room.width, room.height);

            // Room border
            ctx.strokeStyle = '#374151';
            ctx.lineWidth = 2 / (this.scale * this.zoom);
            ctx.strokeRect(room.x, room.y, room.width, room.height);
        });
    }

    drawDoors(ctx) {
        const { rooms } = this.geometryData;

        ctx.strokeStyle = '#0891b2';
        ctx.lineWidth = 3 / (this.scale * this.zoom);

        rooms.forEach(room => {
            if (!room.door) return;

            const { door } = room;

            // Door opening
            ctx.fillStyle = '#ffffff';
            const doorLength = door.width;

            switch (door.direction) {
                case 'north':
                case 'south':
                    ctx.fillRect(door.x, door.y - 50, doorLength, 100);
                    ctx.strokeRect(door.x, door.y - 50, doorLength, 100);
                    break;
                case 'east':
                case 'west':
                    ctx.fillRect(door.x - 50, door.y, 100, doorLength);
                    ctx.strokeRect(door.x - 50, door.y, 100, doorLength);
                    break;
            }
        });
    }

    drawLabels(ctx) {
        const { rooms, boundary } = this.geometryData;

        rooms.forEach(room => {
            const centerX = room.x + room.width / 2;
            const centerY = -(room.y + room.height / 2); // Flipped

            // Calculate font size based on room size
            const fontSize = Math.max(12, Math.min(16, Math.min(room.width, room.height) / 300));
            const scaledFontSize = fontSize / (this.scale * this.zoom);

            ctx.fillStyle = '#1f2937';
            ctx.font = `bold ${scaledFontSize * 1000}px Inter, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            // Room name
            ctx.fillText(room.label, centerX, centerY - scaledFontSize * 600);

            // Area
            const areaSqm = (room.width * room.height) / 1_000_000;
            ctx.font = `${scaledFontSize * 800}px Inter, sans-serif`;
            ctx.fillStyle = '#6b7280';
            ctx.fillText(`${areaSqm.toFixed(1)} m²`, centerX, centerY + scaledFontSize * 500);
        });

        // Boundary dimensions
        ctx.fillStyle = '#374151';
        const dimFontSize = 14 / (this.scale * this.zoom);
        ctx.font = `${dimFontSize * 1000}px Inter, sans-serif`;

        // Width
        ctx.fillText(
            `${mmToM(boundary.width).toFixed(2)} m`,
            boundary.width / 2,
            500 / (this.scale * this.zoom)
        );

        // Height
        ctx.save();
        ctx.translate(-500 / (this.scale * this.zoom), -boundary.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(`${mmToM(boundary.height).toFixed(2)} m`, 0, 0);
        ctx.restore();
    }
}

/**
 * Requirement Form Component
 * Handles structured input for floor plan requirements
 */

import { ROOM_TYPE_OPTIONS, ROOM_TYPES } from '../utils/constants.js';

export class RequirementForm {
    constructor(containerElement, onSubmit) {
        this.container = containerElement;
        this.onSubmit = onSubmit;
        this.rooms = [];
        this.roomIdCounter = 0;

        // Cache DOM elements
        this.totalAreaInput = document.getElementById('totalArea');
        this.plotWidthInput = document.getElementById('plotWidth');
        this.plotLengthInput = document.getElementById('plotLength');
        this.roomListElement = document.getElementById('roomList');
        this.addRoomBtn = document.getElementById('addRoomBtn');
        this.generateBtn = document.getElementById('generateBtn');

        // Bind methods
        this.addRoom = this.addRoom.bind(this);
        this.removeRoom = this.removeRoom.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);

        // Set up event listeners
        this.setupEventListeners();

        // Add initial rooms
        this.addDefaultRooms();
    }

    setupEventListeners() {
        this.addRoomBtn.addEventListener('click', () => this.addRoom());
        this.generateBtn.addEventListener('click', this.handleSubmit);

        // Enter key submits form
        this.container.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.target.tagName === 'INPUT') {
                e.preventDefault();
                this.handleSubmit();
            }
        });
    }

    addDefaultRooms() {
        // Add some common rooms by default
        this.addRoom('living_room', 1, 18);
        this.addRoom('bedroom', 2, 12);
        this.addRoom('kitchen', 1, 10);
        this.addRoom('bathroom', 1, 5);
    }

    addRoom(type = 'bedroom', quantity = 1, area = null) {
        const id = `room_${this.roomIdCounter++}`;
        const roomType = ROOM_TYPES[type];
        const defaultArea = area || roomType?.defaultAreaSqm || 12;

        const room = {
            id,
            type,
            quantity,
            minAreaSqm: defaultArea
        };

        this.rooms.push(room);
        this.renderRoomRow(room);
    }

    renderRoomRow(room) {
        const row = document.createElement('div');
        row.className = 'room-row';
        row.dataset.roomId = room.id;

        row.innerHTML = `
      <select class="form-input form-select room-type-select" data-field="type">
        ${ROOM_TYPE_OPTIONS.map(opt =>
            `<option value="${opt.value}" ${opt.value === room.type ? 'selected' : ''}>${opt.label}</option>`
        ).join('')}
      </select>
      <input type="number" class="form-input" data-field="quantity" 
             value="${room.quantity}" min="1" max="10" placeholder="Qty">
      <div class="input-group">
        <input type="number" class="form-input" data-field="minAreaSqm" 
               value="${room.minAreaSqm}" min="1" step="1" placeholder="Area">
        <span class="input-suffix">m²</span>
      </div>
      <button type="button" class="room-delete-btn" title="Remove room">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M18 6 6 18M6 6l12 12"/>
        </svg>
      </button>
    `;

        // Event listeners for this row
        row.querySelector('.room-delete-btn').addEventListener('click', () => {
            this.removeRoom(room.id);
        });

        row.querySelectorAll('input, select').forEach(input => {
            input.addEventListener('change', (e) => {
                const field = e.target.dataset.field;
                const value = e.target.type === 'number' ? parseFloat(e.target.value) : e.target.value;
                this.updateRoom(room.id, field, value);
            });
        });

        this.roomListElement.appendChild(row);
    }

    updateRoom(id, field, value) {
        const room = this.rooms.find(r => r.id === id);
        if (room) {
            room[field] = value;
        }
    }

    removeRoom(id) {
        const index = this.rooms.findIndex(r => r.id === id);
        if (index !== -1) {
            this.rooms.splice(index, 1);
            const row = this.roomListElement.querySelector(`[data-room-id="${id}"]`);
            if (row) {
                row.remove();
            }
        }

        // Show empty state if no rooms
        if (this.rooms.length === 0) {
            this.showEmptyState();
        }
    }

    showEmptyState() {
        if (this.roomListElement.querySelector('.room-list-empty')) return;

        const empty = document.createElement('div');
        empty.className = 'room-list-empty';
        empty.textContent = 'No rooms added. Click "Add Room" to get started.';
        this.roomListElement.appendChild(empty);
    }

    hideEmptyState() {
        const empty = this.roomListElement.querySelector('.room-list-empty');
        if (empty) empty.remove();
    }

    getFormData() {
        // Refresh room data from DOM
        this.rooms.forEach(room => {
            const row = this.roomListElement.querySelector(`[data-room-id="${room.id}"]`);
            if (row) {
                room.type = row.querySelector('[data-field="type"]').value;
                room.quantity = parseInt(row.querySelector('[data-field="quantity"]').value, 10);
                room.minAreaSqm = parseFloat(row.querySelector('[data-field="minAreaSqm"]').value);
            }
        });

        return {
            totalArea: this.totalAreaInput.value,
            plotWidth: this.plotWidthInput.value,
            plotLength: this.plotLengthInput.value,
            rooms: this.rooms.map(r => ({
                type: r.type,
                quantity: r.quantity,
                minAreaSqm: r.minAreaSqm
            }))
        };
    }

    handleSubmit() {
        const formData = this.getFormData();
        if (this.onSubmit) {
            this.onSubmit(formData);
        }
    }

    setLoading(isLoading) {
        this.generateBtn.disabled = isLoading;
        if (isLoading) {
            this.generateBtn.innerHTML = `
        <span class="spinner"></span>
        Generating...
      `;
        } else {
            this.generateBtn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        Generate Floor Plan
      `;
        }
    }

    reset() {
        this.totalAreaInput.value = '';
        this.plotWidthInput.value = '';
        this.plotLengthInput.value = '';
        this.rooms = [];
        this.roomListElement.innerHTML = '';
        this.addDefaultRooms();
    }
}

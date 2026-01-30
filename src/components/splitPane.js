/**
 * Split Pane Component
 * Handles resizable split between canvas and agent panel
 */

export class SplitPane {
    constructor(handleElement, leftPane, rightPane) {
        this.handle = handleElement;
        this.leftPane = leftPane;
        this.rightPane = rightPane;

        this.isDragging = false;
        this.startX = 0;
        this.startWidth = 0;

        // Constraints
        this.minWidth = 320;
        this.maxWidth = 600;

        // Bind methods
        this.handleMouseDown = this.handleMouseDown.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);

        // Set up event listeners
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.handle.addEventListener('mousedown', this.handleMouseDown);
        document.addEventListener('mousemove', this.handleMouseMove);
        document.addEventListener('mouseup', this.handleMouseUp);
    }

    destroy() {
        this.handle.removeEventListener('mousedown', this.handleMouseDown);
        document.removeEventListener('mousemove', this.handleMouseMove);
        document.removeEventListener('mouseup', this.handleMouseUp);
    }

    handleMouseDown(e) {
        e.preventDefault();
        this.isDragging = true;
        this.startX = e.clientX;
        this.startWidth = this.rightPane.offsetWidth;
        this.handle.classList.add('active');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    }

    handleMouseMove(e) {
        if (!this.isDragging) return;

        const dx = this.startX - e.clientX;
        const newWidth = Math.max(this.minWidth, Math.min(this.maxWidth, this.startWidth + dx));

        this.rightPane.style.width = `${newWidth}px`;

        // Trigger resize event for canvas
        window.dispatchEvent(new Event('resize'));
    }

    handleMouseUp() {
        if (!this.isDragging) return;

        this.isDragging = false;
        this.handle.classList.remove('active');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
    }
}

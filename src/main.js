/**
 * CAD Floor Planner - Main Entry Point
 * Chat-only interface with natural language processing
 */

// Import agents
import { parseNaturalLanguage, validateRequirements } from './agents/requirementAgent.js';
import { checkFeasibility } from './agents/planningAgent.js';
import { generateLayout } from './agents/geometryAgent.js';
import { generateDXF, downloadDXF } from './agents/cadAgent.js';

// Import components
import { CanvasRenderer } from './components/canvas.js';
import { ChatInterface } from './components/chatInterface.js';
import { SplitPane } from './components/splitPane.js';

// Import state
import { getState, setState, resetState } from './state.js';

/**
 * Main application class
 */
class FloorPlannerApp {
    constructor() {
        // Components
        this.canvas = null;
        this.chat = null;
        this.splitPane = null;

        // Current floor plan data
        this.geometryData = null;
        this.dxfContent = null;

        // Conversation context
        this.context = {};

        // Bind methods
        this.handleChatMessage = this.handleChatMessage.bind(this);
        this.handleReset = this.handleReset.bind(this);
        this.handleDownload = this.handleDownload.bind(this);

        // Initialize
        this.init();
    }

    init() {
        this.initCanvas();
        this.initChat();
        this.initSplitPane();
        this.initToolbar();

        console.log('CAD Floor Planner initialized (Chat Mode)');
    }

    initCanvas() {
        const canvasElement = document.getElementById('floorPlanCanvas');
        const containerElement = document.getElementById('canvasContainer');

        this.canvas = new CanvasRenderer(canvasElement, containerElement);

        // Canvas controls
        document.getElementById('zoomInBtn').addEventListener('click', () => this.canvas.zoomIn());
        document.getElementById('zoomOutBtn').addEventListener('click', () => this.canvas.zoomOut());
        document.getElementById('resetViewBtn').addEventListener('click', () => this.canvas.resetView());
    }

    initChat() {
        const chatContainer = document.getElementById('chatContainer');
        this.chat = new ChatInterface(chatContainer, this.handleChatMessage);
    }

    initSplitPane() {
        const handle = document.getElementById('resizeHandle');
        const leftPane = document.getElementById('canvasPane');
        const rightPane = document.getElementById('agentPane');

        this.splitPane = new SplitPane(handle, leftPane, rightPane);
    }

    initToolbar() {
        document.getElementById('resetBtn').addEventListener('click', this.handleReset);
        document.getElementById('downloadBtn').addEventListener('click', this.handleDownload);
    }

    async handleChatMessage(text) {
        // Handle action commands
        if (text.startsWith('__action__:')) {
            const action = text.replace('__action__:', '');
            this.handleAction(action);
            return;
        }

        this.chat.setLoading(true);

        // Simulate thinking delay for natural feel
        await this.delay(500 + Math.random() * 500);

        try {
            // Parse natural language
            const parseResult = parseNaturalLanguage(text, this.context);

            // Update context
            this.context = parseResult.data;

            // Check if we have complete requirements
            if (parseResult.complete) {
                // Show summary and generate
                const rooms = this.context.rooms.map(r => ({
                    ...r,
                    label: r.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
                }));

                this.chat.addMessage('agent', parseResult.response.text, {
                    rooms: rooms,
                    totalArea: this.context.totalAreaSqm,
                    actions: [
                        { id: 'generate', label: '✨ Generate Floor Plan', icon: '' },
                        { id: 'modify', label: '✏️ Modify', icon: '' }
                    ]
                });
            } else if (parseResult.understood) {
                // Partial understanding - ask for more
                this.chat.addMessage('agent', parseResult.response.text);
            } else {
                // Didn't understand
                this.chat.addMessage('agent', parseResult.response.text);
            }

        } catch (error) {
            console.error('Error processing message:', error);
            this.chat.addMessage('agent',
                "I encountered an issue processing that. Could you try rephrasing?",
                { error: error.message }
            );
        } finally {
            this.chat.setLoading(false);
        }
    }

    handleAction(action) {
        switch (action) {
            case 'generate':
                this.generateFloorPlan();
                break;
            case 'modify':
                this.chat.addMessage('agent',
                    "Sure! What would you like to change?\n\n• Adjust room sizes\n• Add or remove rooms\n• Change total area"
                );
                break;
            case 'download':
                this.handleDownload();
                break;
        }
    }

    async generateFloorPlan() {
        this.chat.setLoading(true);
        this.chat.addMessage('agent', '🔄 Generating your floor plan...');

        await this.delay(800);

        try {
            // Validate requirements
            const validation = validateRequirements(this.context);

            if (!validation.success) {
                this.chat.addMessage('agent',
                    "I found an issue with the requirements:",
                    { error: validation.error }
                );
                this.chat.setLoading(false);
                return;
            }

            // Check feasibility
            const planResult = checkFeasibility(validation.data);

            if (!planResult.success) {
                this.chat.addMessage('agent',
                    "The layout isn't feasible:",
                    { error: planResult.error }
                );
                this.chat.setLoading(false);
                return;
            }

            // Generate layout
            const layoutResult = generateLayout(planResult.data);

            if (!layoutResult.success) {
                this.chat.addMessage('agent',
                    "I couldn't fit all rooms in the available space:",
                    { error: layoutResult.error }
                );
                this.chat.setLoading(false);
                return;
            }

            // Success!
            this.geometryData = layoutResult.data;
            this.dxfContent = generateDXF(this.geometryData);

            // Update canvas
            this.canvas.setGeometry(this.geometryData);

            // Hide placeholder
            document.getElementById('canvasPlaceholder').classList.add('hidden');

            // Enable download button
            document.getElementById('downloadBtn').disabled = false;

            // Show success message
            this.chat.addMessage('agent',
                "✅ **Floor plan generated!**\n\nI've created your floor plan based on your requirements. You can see it in the preview on the left.\n\n• Use the zoom controls to inspect details\n• Click **Download DXF** to export for AutoCAD",
                {
                    success: 'Floor plan ready for download',
                    actions: [
                        { id: 'download', label: '📥 Download DXF', icon: '' },
                        { id: 'modify', label: '🔄 Start Over', icon: '' }
                    ]
                }
            );

        } catch (error) {
            console.error('Error generating floor plan:', error);
            this.chat.addMessage('agent',
                "Something went wrong while generating:",
                { error: error.message }
            );
        } finally {
            this.chat.setLoading(false);
        }
    }

    handleReset() {
        // Clear geometry
        this.geometryData = null;
        this.dxfContent = null;
        this.context = {};

        // Clear canvas
        this.canvas.clear();

        // Show placeholder
        document.getElementById('canvasPlaceholder').classList.remove('hidden');

        // Disable download
        document.getElementById('downloadBtn').disabled = true;

        // Reset chat
        this.chat.reset();

        // Add reset message
        this.chat.addMessage('agent',
            "🔄 Starting fresh! Tell me about your new floor plan requirements."
        );

        // Reset state
        resetState();
    }

    handleDownload() {
        if (!this.dxfContent) {
            this.chat.addMessage('agent',
                "There's no floor plan to download yet. Let me help you create one first!",
                { error: 'No floor plan generated' }
            );
            return;
        }

        const timestamp = new Date().toISOString().slice(0, 10);
        const filename = `floor_plan_${timestamp}.dxf`;

        downloadDXF(this.dxfContent, filename);

        this.chat.addMessage('agent',
            `📥 Downloaded **${filename}**\n\nYou can open this file in AutoCAD, LibreCAD, or any DXF-compatible software.`
        );
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new FloorPlannerApp();
});

/**
 * CAD Floor Planner - Main Entry Point
 * Intelligent chat interface with natural language processing
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
import { resetState } from './state.js';

/**
 * Main application class
 */
class FloorPlannerApp {
    constructor() {
        this.canvas = null;
        this.chat = null;
        this.splitPane = null;

        this.geometryData = null;
        this.dxfContent = null;
        this.context = {};

        this.handleChatMessage = this.handleChatMessage.bind(this);
        this.handleReset = this.handleReset.bind(this);
        this.handleDownload = this.handleDownload.bind(this);

        this.init();
    }

    init() {
        this.initCanvas();
        this.initChat();
        this.initSplitPane();
        this.initToolbar();

        console.log('CAD Floor Planner initialized');
    }

    initCanvas() {
        const canvasElement = document.getElementById('floorPlanCanvas');
        const containerElement = document.getElementById('canvasContainer');

        this.canvas = new CanvasRenderer(canvasElement, containerElement);

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
        // Handle action commands from buttons
        if (text.startsWith('__action__:')) {
            const action = text.replace('__action__:', '');
            this.handleAction(action);
            return;
        }

        this.chat.setLoading(true);

        // Small delay for natural feel
        await this.delay(400 + Math.random() * 300);

        try {
            // Parse the user's message
            const parseResult = parseNaturalLanguage(text, this.context);

            // Update conversation context
            this.context = parseResult.data;

            // Handle different parse outcomes
            if (parseResult.complete) {
                // Requirements are complete - show summary with generate button
                const rooms = this.context.rooms.map(r => ({
                    ...r,
                    label: r.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
                }));

                this.chat.addMessage('agent', parseResult.response, {
                    rooms: rooms,
                    totalArea: this.context.totalAreaSqm,
                    actions: [
                        { id: 'generate', label: 'Generate Floor Plan' },
                        { id: 'modify', label: 'Modify Requirements' }
                    ]
                });
            } else if (parseResult.wantsToGenerate && this.context.rooms && this.context.totalAreaSqm) {
                // User confirmed - generate directly
                this.generateFloorPlan();
            } else {
                // Need more information or provide feedback
                this.chat.addMessage('agent', parseResult.response);
            }

        } catch (error) {
            console.error('Error processing message:', error);
            this.chat.addMessage('agent',
                "There was an error processing your request. Please try rephrasing your requirements.",
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
                    "What would you like to change?\n\n- To change room sizes, say something like \"make bedroom 15 sqm\"\n- To add rooms, say \"add 1 study room of 10 sqm\"\n- To change total area, say \"total area should be 120 sqm\""
                );
                break;
            case 'download':
                this.handleDownload();
                break;
            case 'new':
                this.handleReset();
                break;
        }
    }

    async generateFloorPlan() {
        this.chat.setLoading(true);
        this.chat.addMessage('agent', 'Generating your floor plan. Please wait...');

        await this.delay(600);

        try {
            // Validate requirements
            const validation = validateRequirements(this.context);

            if (!validation.success) {
                this.chat.addMessage('agent',
                    `There is an issue with your requirements:\n\n${validation.error}\n\nPlease correct this and try again.`
                );
                this.chat.setLoading(false);
                return;
            }

            // Check architectural feasibility
            const planResult = checkFeasibility(validation.data);

            if (!planResult.success) {
                this.chat.addMessage('agent',
                    `The floor plan is not feasible:\n\n${planResult.error}\n\nPlease adjust your requirements.`
                );
                this.chat.setLoading(false);
                return;
            }

            // Generate room layout
            const layoutResult = generateLayout(planResult.data);

            if (!layoutResult.success) {
                this.chat.addMessage('agent',
                    `I could not fit all rooms in the available space:\n\n${layoutResult.error}\n\nTry increasing the total area or reducing room sizes.`
                );
                this.chat.setLoading(false);
                return;
            }

            // Success - store data and render
            this.geometryData = layoutResult.data;
            this.dxfContent = generateDXF(this.geometryData);

            // Update canvas
            this.canvas.setGeometry(this.geometryData);
            document.getElementById('canvasPlaceholder').classList.add('hidden');
            document.getElementById('downloadBtn').disabled = false;

            // Show success message
            this.chat.addMessage('agent',
                "Your floor plan has been generated successfully.\n\nYou can see the preview on the left side of the screen. Use the zoom controls to inspect the details.\n\nClick \"Download DXF\" to export the floor plan for use in AutoCAD or other CAD software.",
                {
                    success: 'Floor plan is ready',
                    actions: [
                        { id: 'download', label: 'Download DXF' },
                        { id: 'new', label: 'Start New Design' }
                    ]
                }
            );

        } catch (error) {
            console.error('Error generating floor plan:', error);
            this.chat.addMessage('agent',
                `An error occurred while generating the floor plan:\n\n${error.message}\n\nPlease try again.`
            );
        } finally {
            this.chat.setLoading(false);
        }
    }

    handleReset() {
        this.geometryData = null;
        this.dxfContent = null;
        this.context = {};

        this.canvas.clear();
        document.getElementById('canvasPlaceholder').classList.remove('hidden');
        document.getElementById('downloadBtn').disabled = true;

        this.chat.reset();
        this.chat.addMessage('agent',
            "Starting a new design. Please tell me your floor plan requirements.\n\nYou can specify the total area and the rooms you need."
        );

        resetState();
    }

    handleDownload() {
        if (!this.dxfContent) {
            this.chat.addMessage('agent',
                "There is no floor plan to download. Please generate a floor plan first."
            );
            return;
        }

        const timestamp = new Date().toISOString().slice(0, 10);
        const filename = `floor_plan_${timestamp}.dxf`;

        downloadDXF(this.dxfContent, filename);

        this.chat.addMessage('agent',
            `The file "${filename}" has been downloaded.\n\nYou can open this file in AutoCAD, LibreCAD, or any DXF-compatible software.`
        );
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new FloorPlannerApp();
});

/**
 * CAD Floor Planner - Main Entry Point
 * Intelligent conversational interface
 */

import { parseNaturalLanguage, validateRequirements, resetConversation } from './agents/requirementAgent.js';
import { checkFeasibility } from './agents/planningAgent.js';
import { generateLayout } from './agents/geometryAgent.js';
import { generateDXF, downloadDXF } from './agents/cadAgent.js';

import { CanvasRenderer } from './components/canvas.js';
import { ChatInterface } from './components/chatInterface.js';
import { SplitPane } from './components/splitPane.js';

import { resetState } from './state.js';

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
        // Handle button actions
        if (text.startsWith('__action__:')) {
            this.handleAction(text.replace('__action__:', ''));
            return;
        }

        this.chat.setLoading(true);

        // Natural typing delay
        await this.delay(300 + Math.random() * 400);

        try {
            // Process with intelligent agent
            const result = parseNaturalLanguage(text, this.context);

            // Update context
            this.context = result.data;

            // Check if user wants to generate
            if (result.wantsToGenerate) {
                await this.generateFloorPlan();
                return;
            }

            // Show agent response
            if (result.complete) {
                // Requirements complete - show with action buttons
                const rooms = this.context.rooms.map(r => ({
                    ...r,
                    label: r.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
                }));

                this.chat.addMessage('agent', result.response, {
                    rooms: rooms,
                    totalArea: this.context.totalAreaSqm,
                    actions: [
                        { id: 'generate', label: 'Generate Floor Plan' },
                        { id: 'modify', label: 'Make Changes' }
                    ]
                });
            } else {
                // Still gathering requirements
                this.chat.addMessage('agent', result.response);
            }

        } catch (error) {
            console.error('Error:', error);
            this.chat.addMessage('agent',
                "I encountered an issue. Could you please rephrase that?"
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
                    "What would you like to change? You can:\n\n- Adjust room sizes (e.g., \"make bedroom 15 sqm\")\n- Add rooms (e.g., \"add a study room\")\n- Remove rooms (e.g., \"remove the balcony\")\n- Change total area (e.g., \"total area 120 sqm\")"
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
        this.chat.addMessage('agent', 'Processing your requirements and generating the floor plan...');

        await this.delay(500);

        try {
            // Validate
            const validation = validateRequirements(this.context);

            if (!validation.success) {
                this.chat.addMessage('agent',
                    `I cannot generate the floor plan yet:\n\n${validation.error}\n\nPlease provide the missing information.`
                );
                this.chat.setLoading(false);
                return;
            }

            // Check feasibility
            const planResult = checkFeasibility(validation.data);

            if (!planResult.success) {
                this.chat.addMessage('agent',
                    `The design is not feasible:\n\n${planResult.error}\n\nPlease adjust your requirements.`
                );
                this.chat.setLoading(false);
                return;
            }

            // Generate layout
            const layoutResult = generateLayout(planResult.data);

            if (!layoutResult.success) {
                this.chat.addMessage('agent',
                    `I could not fit all rooms:\n\n${layoutResult.error}\n\nTry increasing the total area or making some rooms smaller.`
                );
                this.chat.setLoading(false);
                return;
            }

            // Success
            this.geometryData = layoutResult.data;
            this.dxfContent = generateDXF(this.geometryData);

            // Update UI
            this.canvas.setGeometry(this.geometryData);
            document.getElementById('canvasPlaceholder').classList.add('hidden');
            document.getElementById('downloadBtn').disabled = false;

            this.chat.addMessage('agent',
                "Your floor plan is ready.\n\nThe preview is shown on the left. You can use the zoom controls to examine the details.\n\nWhen you are satisfied, download the DXF file to use in CAD software like AutoCAD.",
                {
                    success: 'Floor plan generated successfully',
                    actions: [
                        { id: 'download', label: 'Download DXF File' },
                        { id: 'new', label: 'Start New Design' }
                    ]
                }
            );

        } catch (error) {
            console.error('Generation error:', error);
            this.chat.addMessage('agent',
                `An error occurred: ${error.message}\n\nPlease try again or adjust your requirements.`
            );
        } finally {
            this.chat.setLoading(false);
        }
    }

    handleReset() {
        this.geometryData = null;
        this.dxfContent = null;
        this.context = {};
        resetConversation();

        this.canvas.clear();
        document.getElementById('canvasPlaceholder').classList.remove('hidden');
        document.getElementById('downloadBtn').disabled = true;

        this.chat.reset();
        this.chat.addMessage('agent',
            "Starting fresh. Tell me about the house you want to design - the total area and what rooms you need."
        );

        resetState();
    }

    handleDownload() {
        if (!this.dxfContent) {
            this.chat.addMessage('agent',
                "No floor plan has been generated yet. Please tell me your requirements first."
            );
            return;
        }

        const timestamp = new Date().toISOString().slice(0, 10);
        const filename = `floor_plan_${timestamp}.dxf`;

        downloadDXF(this.dxfContent, filename);

        this.chat.addMessage('agent',
            `Downloaded: ${filename}\n\nThis file can be opened in AutoCAD, LibreCAD, DraftSight, or any DXF-compatible software.`
        );
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new FloorPlannerApp();
});

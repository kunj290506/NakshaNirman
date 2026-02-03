/**
 * CAD Floor Planner - Main Entry Point
 * Intelligent AI-powered floor plan generator
 */

import { parseNaturalLanguage, validateRequirements, resetConversation } from './agents/requirementAgent.js';
import { checkFeasibility } from './agents/planningAgent.js';
import { generateLayout } from './agents/geometryAgent.js';
import { generateDXF, downloadDXF } from './agents/cadAgent.js';

import { CanvasRenderer } from './components/canvas.js';
import { ChatInterface } from './components/chatInterface.js';
import { SplitPane } from './components/splitPane.js';

import { initAI, getStoredApiKey, storeApiKey, isAIAvailable } from './services/aiService.js';
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

    async init() {
        // Initialize AI if key exists
        const apiKey = getStoredApiKey();
        if (apiKey) {
            initAI(apiKey);
        }

        this.initCanvas();
        this.initChat();
        this.initSplitPane();
        this.initToolbar();
        this.initAPIKeyModal();

        // Show API key prompt if not configured
        if (!apiKey) {
            this.showAPIKeyModal();
        }

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

        // Settings button for API key
        const settingsBtn = document.getElementById('settingsBtn');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => this.showAPIKeyModal());
        }
    }

    initAPIKeyModal() {
        const modal = document.getElementById('apiKeyModal');
        const input = document.getElementById('apiKeyInput');
        const saveBtn = document.getElementById('saveApiKeyBtn');
        const skipBtn = document.getElementById('skipApiKeyBtn');
        const closeBtn = document.getElementById('closeModalBtn');

        if (saveBtn) {
            saveBtn.addEventListener('click', () => {
                const key = input.value.trim();
                if (key) {
                    storeApiKey(key);
                    this.hideAPIKeyModal();
                    this.chat.addMessage('agent', 'AI mode enabled! I can now understand complex requests. Try asking me anything about floor plans!');
                }
            });
        }

        if (skipBtn) {
            skipBtn.addEventListener('click', () => {
                this.hideAPIKeyModal();
                this.chat.addMessage('agent', 'Running in basic mode. For smarter responses, add your Gemini API key in settings.');
            });
        }

        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hideAPIKeyModal());
        }
    }

    showAPIKeyModal() {
        const modal = document.getElementById('apiKeyModal');
        if (modal) {
            modal.classList.remove('hidden');
            const existing = getStoredApiKey();
            if (existing) {
                document.getElementById('apiKeyInput').value = existing;
            }
        }
    }

    hideAPIKeyModal() {
        const modal = document.getElementById('apiKeyModal');
        if (modal) modal.classList.add('hidden');
    }

    async handleChatMessage(text) {
        // Handle image analysis results
        if (text.startsWith('__image_analyzed__:')) {
            const analysisData = JSON.parse(text.replace('__image_analyzed__:', ''));
            
            // Update context with detected plot dimensions
            if (analysisData.plotDimensions) {
                this.context.plotDimensions = {
                    width: analysisData.plotDimensions.width,
                    length: analysisData.plotDimensions.length
                };
                this.context.plotShape = analysisData.plotShape;
                this.context.totalAreaSqm = analysisData.totalAreaSqft * 0.0929; // Convert to sqm
            }
            
            return;
        }
        
        // Handle button actions
        if (text.startsWith('__action__:')) {
            this.handleAction(text.replace('__action__:', ''));
            return;
        }

        this.chat.setLoading(true);

        try {
            // Process with AI or fallback
            const result = await parseNaturalLanguage(text, this.context);

            // Update context
            this.context = result.data || this.context;

            // Check if user wants to generate
            if (result.wantsToGenerate) {
                await this.generateFloorPlan();
                return;
            }

            // Show agent response
            if (result.complete) {
                const rooms = (this.context.rooms || []).map(r => ({
                    ...r,
                    label: r.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
                }));

                this.chat.addMessage('agent', result.response, {
                    rooms: rooms,
                    totalArea: this.context.totalAreaSqm,
                    thought_process: result.thought_process, // Pass thoughts
                    actions: [
                        { id: 'generate', label: 'Generate Floor Plan' },
                        { id: 'modify', label: 'Make Changes' }
                    ]
                });
            } else {
                this.chat.addMessage('agent', result.response, {
                    thought_process: result.thought_process // Pass thoughts here too
                });
            }

        } catch (error) {
            console.error('Error:', error);
            this.chat.addMessage('agent', 'Something went wrong. Please try again.');
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
                    'What would you like to change? You can:\n- Adjust room sizes\n- Add or remove rooms\n- Change total area'
                );
                break;
            case 'use_detected_dimensions':
                // Use dimensions from image analysis
                this.chat.addMessage('agent', 'Great! Now tell me what rooms you need, or I can use a standard layout.');
                break;
            case 'manual_adjust':
                this.chat.addMessage('agent', 'Please tell me the plot dimensions you want to use (e.g., "36 feet by 36 feet" or "1220 square feet")');
                break;
            case 'configure_api':
                this.showAPIKeyModal();
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
        this.chat.addMessage('agent', 'Generating your floor plan...');

        try {
            // Validate
            const validation = validateRequirements(this.context);

            if (!validation.success) {
                this.chat.addMessage('agent', `Cannot generate: ${validation.error}`);
                this.chat.setLoading(false);
                return;
            }

            // Check feasibility
            const planResult = checkFeasibility(validation.data);

            if (!planResult.success) {
                this.chat.addMessage('agent', `Design issue: ${planResult.error}`);
                this.chat.setLoading(false);
                return;
            }

            // Generate layout (Async ML Inference)
            const layoutResult = await generateLayout(planResult.data);

            if (!layoutResult.success) {
                this.chat.addMessage('agent', `Layout issue: ${layoutResult.error}`);
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
                'Your floor plan is ready! Check the preview on the left.\n\nThe layout includes a central lobby connecting all rooms.',
                {
                    success: 'Floor plan generated',
                    actions: [
                        { id: 'download', label: 'Download DXF' },
                        { id: 'new', label: 'New Design' }
                    ]
                }
            );

        } catch (error) {
            console.error('Generation error:', error);
            this.chat.addMessage('agent', `Error: ${error.message}`);
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
        this.chat.addMessage('agent', 'Starting fresh. Tell me about your dream house!');

        resetState();
    }

    handleDownload() {
        if (!this.dxfContent) {
            this.chat.addMessage('agent', 'No floor plan to download. Generate one first!');
            return;
        }

        const filename = `floor_plan_${new Date().toISOString().slice(0, 10)}.dxf`;
        downloadDXF(this.dxfContent, filename);

        this.chat.addMessage('agent', `Downloaded: ${filename}\n\nOpen in AutoCAD or any DXF viewer.`);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new FloorPlannerApp();
});

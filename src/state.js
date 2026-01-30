/**
 * CAD Floor Planner - State Management
 * Centralized application state with event-based updates
 */

// Initial state
const initialState = {
    // User input mode
    mode: 'form', // 'form' | 'chat'

    // Validated requirements from user input
    requirements: null,

    // Planning agent output
    planResult: null,

    // Geometry agent output (room coordinates)
    geometry: null,

    // Error state
    error: null,

    // Processing state
    isProcessing: false,

    // Canvas state
    canvas: {
        zoom: 1,
        panX: 0,
        panY: 0
    },

    // Chat state
    chat: {
        messages: [],
        currentStep: 'area' // 'area' | 'rooms' | 'confirm'
    }
};

// Current state
let state = { ...initialState };

// Event listeners
const listeners = new Map();

/**
 * Get current state
 * @returns {Object} Current state
 */
export function getState() {
    return { ...state };
}

/**
 * Get a specific state property
 * @param {string} key - State key
 * @returns {*} State value
 */
export function getStateValue(key) {
    return state[key];
}

/**
 * Update state
 * @param {Object} updates - Partial state updates
 */
export function setState(updates) {
    const prevState = { ...state };
    state = { ...state, ...updates };

    // Notify listeners of changed keys
    Object.keys(updates).forEach(key => {
        if (prevState[key] !== state[key]) {
            emit(key, state[key], prevState[key]);
        }
    });

    // Always emit a general state change event
    emit('stateChange', state, prevState);
}

/**
 * Reset state to initial values
 */
export function resetState() {
    const prevState = { ...state };
    state = { ...initialState };
    emit('stateChange', state, prevState);
    emit('reset', state, prevState);
}

/**
 * Subscribe to state changes
 * @param {string} event - Event name (state key or 'stateChange')
 * @param {Function} callback - Callback function
 * @returns {Function} Unsubscribe function
 */
export function subscribe(event, callback) {
    if (!listeners.has(event)) {
        listeners.set(event, new Set());
    }
    listeners.get(event).add(callback);

    // Return unsubscribe function
    return () => {
        listeners.get(event).delete(callback);
    };
}

/**
 * Emit an event
 * @param {string} event - Event name
 * @param {*} newValue - New value
 * @param {*} oldValue - Old value
 */
function emit(event, newValue, oldValue) {
    if (listeners.has(event)) {
        listeners.get(event).forEach(callback => {
            try {
                callback(newValue, oldValue);
            } catch (error) {
                console.error(`Error in state listener for ${event}:`, error);
            }
        });
    }
}

/**
 * Set error state
 * @param {string|null} error - Error message or null to clear
 */
export function setError(error) {
    setState({ error, isProcessing: false });
}

/**
 * Clear error state
 */
export function clearError() {
    setState({ error: null });
}

/**
 * Set processing state
 * @param {boolean} isProcessing 
 */
export function setProcessing(isProcessing) {
    setState({ isProcessing });
}

/**
 * Update canvas state
 * @param {Object} canvasUpdates 
 */
export function updateCanvas(canvasUpdates) {
    setState({
        canvas: { ...state.canvas, ...canvasUpdates }
    });
}

/**
 * Add chat message
 * @param {Object} message - {role: 'user'|'agent', content: string}
 */
export function addChatMessage(message) {
    setState({
        chat: {
            ...state.chat,
            messages: [...state.chat.messages, { ...message, timestamp: Date.now() }]
        }
    });
}

/**
 * Set chat step
 * @param {string} step 
 */
export function setChatStep(step) {
    setState({
        chat: { ...state.chat, currentStep: step }
    });
}

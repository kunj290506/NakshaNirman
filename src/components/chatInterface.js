/**
 * Chat Interface Component
 * Handles conversation with the floor plan agent
 */

import { ROOM_TYPES } from '../utils/constants.js';
import { analyzeFloorPlanImage, validateImageFile, createImagePreview } from '../utils/imageAnalyzer.js';

export class ChatInterface {
  constructor(containerElement, onMessage) {
    this.container = containerElement;
    this.onMessage = onMessage;

    this.messagesContainer = document.getElementById('chatMessages');
    this.chatContainer = document.getElementById('chatContainer');
    this.chatInput = document.getElementById('chatInput');
    this.sendBtn = document.getElementById('sendBtn');
    this.uploadImageBtn = document.getElementById('uploadImageBtn');
    this.imageUploadInput = document.getElementById('imageUploadInput');

    this.sendMessage = this.sendMessage.bind(this);
    this.handleKeyDown = this.handleKeyDown.bind(this);
    this.handleInput = this.handleInput.bind(this);
    this.handleExampleClick = this.handleExampleClick.bind(this);
    this.handleImageUpload = this.handleImageUpload.bind(this);

    this.setupEventListeners();
  }

  setupEventListeners() {
    this.sendBtn.addEventListener('click', this.sendMessage);
    this.chatInput.addEventListener('keydown', this.handleKeyDown);
    this.chatInput.addEventListener('input', this.handleInput);

    // Image upload listeners
    this.uploadImageBtn.addEventListener('click', () => {
      this.imageUploadInput.click();
    });
    this.imageUploadInput.addEventListener('change', this.handleImageUpload);

    this.messagesContainer.addEventListener('click', (e) => {
      if (e.target.classList.contains('example-chip')) {
        this.handleExampleClick(e.target.textContent);
      }
      if (e.target.closest('.action-btn')) {
        const action = e.target.closest('.action-btn').dataset.action;
        if (action) {
          this.onMessage(`__action__:${action}`);
        }
      }
    });
  }

  async handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file
    const validation = validateImageFile(file);
    if (!validation.valid) {
      this.addMessage('agent', validation.error, { error: validation.error });
      return;
    }

    // Show image preview in chat
    createImagePreview(file, (previewUrl) => {
      this.addMessage('user', '', {
        imagePreview: previewUrl,
        imageName: file.name
      });
    });

    // Show analyzing message
    this.setLoading(true);
    this.addMessage('agent', 'Analyzing your floor plan image... This may take a moment.');

    // Get API key
    const apiKey = localStorage.getItem('gemini_api_key');
    if (!apiKey) {
      this.setLoading(false);
      this.addMessage('agent', 'Please configure your Gemini API key first to analyze images.', {
        error: 'API key required',
        actions: [{ id: 'configure_api', label: 'Configure API Key' }]
      });
      return;
    }

    // Analyze image
    try {
      const result = await analyzeFloorPlanImage(file, apiKey);
      this.setLoading(false);

      if (result.success) {
        // Format analysis results
        let message = `**Image Analysis Complete**\n\n`;

        if (result.plotDimensions) {
          message += `**Plot Dimensions:**\n`;
          message += `• Width: ${result.plotDimensions.widthFt?.toFixed(1) || '?'} feet (${(result.plotDimensions.width / 1000).toFixed(2)} meters)\n`;
          message += `• Length: ${result.plotDimensions.lengthFt?.toFixed(1) || '?'} feet (${(result.plotDimensions.length / 1000).toFixed(2)} meters)\n`;
          message += `• Shape: ${result.plotShape}\n\n`;
        }

        if (result.totalAreaSqft) {
          message += `**Total Area:** ${result.totalAreaSqft} sq ft (${(result.totalAreaSqft * 0.0929).toFixed(1)} sqm)\n\n`;
        }

        if (result.existingRooms && result.existingRooms.length > 0) {
          message += `**Detected Rooms:**\n`;
          result.existingRooms.forEach(room => {
            message += `• ${room.count}x ${room.type}\n`;
          });
          message += `\n`;
        }

        if (result.notes) {
          message += `**Notes:** ${result.notes}\n\n`;
        }

        message += `Would you like me to generate a floor plan based on these dimensions?`;

        this.addMessage('agent', message, {
          actions: [
            { id: 'use_detected_dimensions', label: 'Yes, use these dimensions' },
            { id: 'manual_adjust', label: 'Let me adjust manually' }
          ]
        });

        // Store analysis result for later use
        this.lastImageAnalysis = result;

        // Trigger callback with image analysis data
        if (this.onMessage) {
          this.onMessage(`__image_analyzed__:${JSON.stringify(result)}`);
        }

      } else {
        this.addMessage('agent', `Failed to analyze image: ${result.error}`, {
          error: result.error
        });
      }
    } catch (error) {
      this.setLoading(false);
      this.addMessage('agent', `Error analyzing image: ${error.message}`, {
        error: error.message
      });
    }

    // Clear input
    event.target.value = '';
  }

  handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      this.sendMessage();
    }
  }

  handleInput() {
    this.chatInput.style.height = 'auto';
    this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 150) + 'px';
  }

  handleExampleClick(text) {
    const cleanText = text.replace(/^["']|["']$/g, '');
    this.chatInput.value = cleanText;
    this.chatInput.focus();
    this.handleInput();
  }

  sendMessage() {
    const text = this.chatInput.value.trim();
    if (!text) return;

    this.addMessage('user', text);
    this.chatInput.value = '';
    this.chatInput.style.height = 'auto';

    if (this.onMessage) {
      this.onMessage(text);
    }
  }

  addMessage(role, content, options = {}) {
    const message = document.createElement('div');
    message.className = `chat-message ${role}`;

    if (role === 'agent') {
      message.innerHTML = `
        <div class="message-avatar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 8V4H8"/>
            <rect width="16" height="12" x="4" y="8" rx="2"/>
            <path d="M2 14h2M20 14h2M15 13v2M9 13v2"/>
          </svg>
        </div>
        <div class="message-content">${this.formatContent(content, options)}</div>
      `;
    } else {
      // User message with optional image preview
      let userContent = '';
      if (options.imagePreview) {
        userContent = `
          <div class="image-preview">
            <img src="${options.imagePreview}" alt="${options.imageName || 'Floor plan'}" />
            <div class="image-name">${options.imageName || 'Floor plan image'}</div>
          </div>
        `;
      } else {
        userContent = this.escapeHtml(content);
      }

      message.innerHTML = `
        <div class="message-avatar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="8" r="4"/>
            <path d="M6 21v-2a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v2"/>
          </svg>
        </div>
        <div class="message-content">${userContent}</div>
      `;
    }

    this.messagesContainer.appendChild(message);
    this.scrollToBottom();

    return message;
  }

  formatContent(content, options = {}) {
    let html = content
      .replace(/### (.*?)\n/g, '<h4 class="font-bold text-sm mt-3 mb-1">$1</h4>')
      .replace(/## (.*?)\n/g, '<h3 class="font-bold text-base mt-4 mb-2 text-gray-800">$1</h3>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>')
      .replace(/• /g, '&bull; ')
      .replace(/- /g, '&bull; ');

    // Add Architect's Thinking Process (Enhanced with icons)
    if (options.thought_process && options.thought_process.length > 0) {
      const thoughtsHtml = options.thought_process.map(t => {
        // Check if thought has emoji prefix
        const hasEmoji = /^[\u{1F300}-\u{1F9FF}]/u.test(t);
        const arrow = hasEmoji ? '' : '<span class="thought-arrow">→</span> ';
        return `<div class="thought-step">${arrow}${t}</div>`;
      }).join('');

      html = `
        <details class="thought-container" open>
          <summary class="thought-summary">
            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path></svg>
            Architect's Design Reasoning
          </summary>
          <div class="thought-content">
            ${thoughtsHtml}
          </div>
        </details>
      ` + html;
    }

    // Add room summary if provided
    if (options.rooms && options.rooms.length > 0) {
      html += this.createRoomSummary(options.rooms, options.totalArea);
    }

    // Add success message
    if (options.success) {
      html += `
        <div class="message-success">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <path d="m9 12 2 2 4-4"/>
          </svg>
          ${options.success}
        </div>
      `;
    }

    // Add error message
    if (options.error) {
      html += `
        <div class="message-error">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <path d="m15 9-6 6M9 9l6 6"/>
          </svg>
          ${options.error}
        </div>
      `;
    }

    // Add action buttons
    if (options.actions && options.actions.length > 0) {
      html += '<div class="message-actions">';
      options.actions.forEach(action => {
        html += `<button class="action-btn" data-action="${action.id}">${action.label}</button>`;
      });
      html += '</div>';
    }

    return html;
  }

  createRoomSummary(rooms, totalArea) {
    let html = '<div class="room-summary">';
    html += `<div class="room-summary-title">Floor Plan Summary (${totalArea || '?'} sqm)</div>`;

    rooms.forEach(room => {
      const color = room.color || ROOM_TYPES[room.type]?.color || '#e2e8f0';
      const label = room.label || ROOM_TYPES[room.type]?.label || room.type;
      const qty = room.quantity || 1;
      html += `
        <div class="room-summary-item">
          <span class="room-color-dot" style="background-color: ${color}"></span>
          <span>${qty}x ${label}</span>
          <span style="margin-left: auto; color: #6b7280;">${room.minAreaSqm} sqm</span>
        </div>
      `;
    });

    const totalRoomArea = rooms.reduce((sum, r) => sum + (r.minAreaSqm * (r.quantity || 1)), 0);
    html += `<div class="room-summary-total">Total room area: ${totalRoomArea} sqm</div>`;
    html += '</div>';
    return html;
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  scrollToBottom() {
    this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
  }

  showTyping() {
    if (this.messagesContainer.querySelector('.typing-indicator')) return;

    const indicator = document.createElement('div');
    indicator.className = 'chat-message agent typing-indicator';
    indicator.innerHTML = `
      <div class="message-avatar">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 8V4H8"/>
          <rect width="16" height="12" x="4" y="8" rx="2"/>
          <path d="M2 14h2M20 14h2M15 13v2M9 13v2"/>
        </svg>
      </div>
      <div class="message-content">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    `;
    this.messagesContainer.appendChild(indicator);
    this.scrollToBottom();
  }

  hideTyping() {
    const indicator = this.messagesContainer.querySelector('.typing-indicator');
    if (indicator) indicator.remove();
  }

  setLoading(isLoading) {
    this.sendBtn.disabled = isLoading;
    this.chatInput.disabled = isLoading;

    if (isLoading) {
      this.showTyping();
    } else {
      this.hideTyping();
    }
  }

  reset() {
    const messages = this.messagesContainer.querySelectorAll('.chat-message');
    messages.forEach((msg, index) => {
      if (index > 0) msg.remove();
    });
    this.chatInput.value = '';
    this.chatInput.style.height = 'auto';
  }
}

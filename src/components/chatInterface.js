/**
 * Chat Interface Component (LLM Style)
 * Handles natural language conversation for floor plan requirements
 */

import { ROOM_TYPES } from '../utils/constants.js';

export class ChatInterface {
  constructor(containerElement, onMessage) {
    this.container = containerElement;
    this.onMessage = onMessage;

    // Cache DOM elements
    this.messagesContainer = document.getElementById('chatMessages');
    this.chatContainer = document.getElementById('chatContainer');
    this.chatInput = document.getElementById('chatInput');
    this.sendBtn = document.getElementById('sendBtn');

    // Bind methods
    this.sendMessage = this.sendMessage.bind(this);
    this.handleKeyDown = this.handleKeyDown.bind(this);
    this.handleInput = this.handleInput.bind(this);
    this.handleExampleClick = this.handleExampleClick.bind(this);

    // Set up event listeners
    this.setupEventListeners();
  }

  setupEventListeners() {
    this.sendBtn.addEventListener('click', this.sendMessage);
    this.chatInput.addEventListener('keydown', this.handleKeyDown);
    this.chatInput.addEventListener('input', this.handleInput);

    // Example chip clicks
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

  handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      this.sendMessage();
    }
  }

  handleInput() {
    // Auto-resize textarea
    this.chatInput.style.height = 'auto';
    this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 150) + 'px';
  }

  handleExampleClick(text) {
    // Remove quotes from example text
    const cleanText = text.replace(/^["']|["']$/g, '');
    this.chatInput.value = cleanText;
    this.chatInput.focus();
    this.handleInput();
  }

  sendMessage() {
    const text = this.chatInput.value.trim();
    if (!text) return;

    // Add user message to UI
    this.addMessage('user', text);

    // Clear input
    this.chatInput.value = '';
    this.chatInput.style.height = 'auto';

    // Emit message
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
      message.innerHTML = `
        <div class="message-avatar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="8" r="4"/>
            <path d="M6 21v-2a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v2"/>
          </svg>
        </div>
        <div class="message-content">${this.escapeHtml(content)}</div>
      `;
    }

    this.messagesContainer.appendChild(message);
    this.scrollToBottom();

    return message;
  }

  formatContent(content, options = {}) {
    let html = content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>')
      .replace(/• /g, '&bull; ');

    // Add room summary if provided
    if (options.rooms && options.rooms.length > 0) {
      html += this.createRoomSummary(options.rooms, options.totalArea);
    }

    // Add success message if provided
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

    // Add error message if provided
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

    // Add action buttons if provided
    if (options.actions && options.actions.length > 0) {
      html += '<div class="message-actions">';
      options.actions.forEach(action => {
        html += `
          <button class="action-btn" data-action="${action.id}">
            ${action.icon || ''}
            ${action.label}
          </button>
        `;
      });
      html += '</div>';
    }

    return html;
  }

  createRoomSummary(rooms, totalArea) {
    let html = '<div class="room-summary">';
    html += `<div class="room-summary-title">📋 Floor Plan Summary (${totalArea || '?'} m²)</div>`;

    rooms.forEach(room => {
      const color = room.color || ROOM_TYPES[room.type]?.color || '#e2e8f0';
      const label = room.label || ROOM_TYPES[room.type]?.label || room.type;
      html += `
        <div class="room-summary-item">
          <span class="room-color-dot" style="background-color: ${color}"></span>
          <span>${room.quantity || 1}× ${label}</span>
          <span style="margin-left: auto; color: #6b7280;">${room.minAreaSqm} m²</span>
        </div>
      `;
    });

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
    // Clear all messages except the initial greeting
    const messages = this.messagesContainer.querySelectorAll('.chat-message');
    messages.forEach((msg, index) => {
      if (index > 0) msg.remove();
    });
    this.chatInput.value = '';
    this.chatInput.style.height = 'auto';
  }
}

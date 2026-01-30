/**
 * Status Panel Component
 * Shows processing status, errors, and success messages
 */

export class StatusPanel {
    constructor(panelElement) {
        this.panel = panelElement;
        this.contentElement = this.panel.querySelector('.status-content');
    }

    show(type, message) {
        this.panel.classList.remove('hidden', 'error', 'success', 'warning');
        this.panel.classList.add(type);
        this.contentElement.innerHTML = this.formatMessage(type, message);
    }

    formatMessage(type, message) {
        const icons = {
            error: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <path d="m15 9-6 6M9 9l6 6"/>
              </svg>`,
            success: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="12" cy="12" r="10"/>
                  <path d="m9 12 2 2 4-4"/>
                </svg>`,
            warning: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>
                  <path d="M12 9v4M12 17h.01"/>
                </svg>`
        };

        return `
      <div class="alert alert-${type}">
        ${icons[type] || ''}
        <span>${message}</span>
      </div>
    `;
    }

    showError(message) {
        this.show('error', message);
    }

    showSuccess(message) {
        this.show('success', message);
    }

    showWarning(message) {
        this.show('warning', message);
    }

    hide() {
        this.panel.classList.add('hidden');
    }

    clear() {
        this.hide();
        this.contentElement.innerHTML = '';
    }
}

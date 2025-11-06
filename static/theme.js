// Theme management
class ThemeManager {
    constructor() {
        this.currentTheme = localStorage.getItem('theme') || 'dark';
        this.init();
    }

    init() {
        this.applyTheme(this.currentTheme);
        this.setupToggleButton();
    }

    applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        document.body.classList.toggle('light-mode', theme === 'light');
        this.currentTheme = theme;
        localStorage.setItem('theme', theme);
        this.updateToggleButton();
    }

    toggleTheme() {
        const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        this.applyTheme(newTheme);
    }

    setupToggleButton() {
        const toggleBtns = document.querySelectorAll('#themeToggle');
        toggleBtns.forEach(toggleBtn => {
            if (toggleBtn) {
                toggleBtn.addEventListener('click', () => this.toggleTheme());
            }
        });
    }

    updateToggleButton() {
        const toggleBtns = document.querySelectorAll('#themeToggle');
        toggleBtns.forEach(toggleBtn => {
            const icon = toggleBtn?.querySelector('svg path');
            if (icon) {
                if (this.currentTheme === 'dark') {
                    // Sun icon for light mode
                    icon.setAttribute('d', 'M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z');
                } else {
                    // Moon icon for dark mode
                    icon.setAttribute('d', 'M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z');
                }
            }
        });
    }
}

// Initialize theme manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ThemeManager();
});
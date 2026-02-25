"""
BenchDrift App — Theme, CSS, and JS constants.

All visual styling lives here. The Gradio theme config, custom CSS for
terminal-aesthetic result cards, and the JS for badge toggling + layout.
Light/dark mode toggle via CSS custom properties.
"""

import gradio as gr

# ---------------------------------------------------------------------------
# JS — badge toggling + theme toggle
# ---------------------------------------------------------------------------

BADGE_JS = """
function toggleFeature(el) {
    const source = el.dataset.source;
    if (source === 'llm') {
        el.classList.toggle('badge-llm-on');
        el.classList.toggle('badge-llm-off');
    } else if (source === 'disc') {
        el.classList.toggle('badge-disc-on');
        el.classList.toggle('badge-disc-off');
    } else {
        el.classList.toggle('badge-on');
        el.classList.toggle('badge-off');
    }
    syncFeaturesToState();
}

function syncFeaturesToState() {
    const badges = document.querySelectorAll('.feature-badge[data-feature]');
    const state = {};
    badges.forEach(function(b) {
        const feat = b.dataset.feature;
        const isOn = b.classList.contains('badge-on') ||
                     b.classList.contains('badge-llm-on') ||
                     b.classList.contains('badge-disc-on');
        state[feat] = isOn;
    });
    const syncBox = document.querySelector('#features-sync-box textarea');
    if (syncBox) {
        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
            window.HTMLTextAreaElement.prototype, 'value'
        ).set;
        nativeInputValueSetter.call(syncBox, JSON.stringify(state));
        syncBox.dispatchEvent(new Event('input', { bubbles: true }));
    }
}

function toggleBenchDriftTheme() {
    const body = document.body;
    const isLight = body.classList.toggle('benchdrift-light');
    localStorage.setItem('benchdrift-theme', isLight ? 'light' : 'dark');
    const btn = document.querySelector('#theme-toggle-btn');
    if (btn) btn.textContent = isLight ? 'dark mode' : 'light mode';
}

// Restore saved theme on load
(function() {
    const saved = localStorage.getItem('benchdrift-theme');
    if (saved === 'light') {
        document.body.classList.add('benchdrift-light');
    }
})();
"""

# ---------------------------------------------------------------------------
# CSS — dark + light mode via custom properties
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ══════════════════════════════════════════════
   CSS CUSTOM PROPERTIES — Dark (default)
   ══════════════════════════════════════════════ */
body {
    --bd-bg: #0d1117;
    --bd-bg-card: #161b22;
    --bd-bg-card-hover: #1c2333;
    --bd-bg-input: #0d1117;
    --bd-border: #30363d;
    --bd-border-light: #2a2a2a;
    --bd-text: #c9d1d9;
    --bd-text-muted: #8b949e;
    --bd-text-dim: #555;
    --bd-text-dimmer: #444;
    --bd-text-heading: #93c5fd;
    --bd-accent-green: #4ade80;
    --bd-accent-green-bg: rgba(74, 222, 128, 0.15);
    --bd-accent-green-border: rgba(74, 222, 128, 0.3);
    --bd-accent-red: #f87171;
    --bd-accent-red-bg: rgba(248, 113, 113, 0.15);
    --bd-accent-yellow: #fbbf24;
    --bd-accent-yellow-bg: rgba(251, 191, 36, 0.15);
    --bd-accent-yellow-border: rgba(251, 191, 36, 0.3);
    --bd-accent-purple: #a855f7;
    --bd-accent-purple-bg: rgba(168, 85, 247, 0.15);
    --bd-accent-blue: #60a5fa;
    --bd-accent-blue-bg: rgba(96, 165, 250, 0.15);
    --bd-accent-blue-border: rgba(96, 165, 250, 0.3);
    --bd-accent-orange: #fb923c;
    --bd-accent-orange-bg: rgba(251, 146, 60, 0.1);
    --bd-accent-orange-border: rgba(251, 146, 60, 0.25);
    --bd-card-variant-bg: rgba(0,0,0,0.15);
    --bd-card-bg: rgba(0,0,0,0.2);
    --bd-baseline-bg: rgba(0,0,0,0.25);
    --bd-badge-off-bg: transparent;
    --bd-badge-off-color: #555;
    --bd-badge-off-border: #333;
    --bd-section-label: #555;
}

/* ══════════════════════════════════════════════
   CSS CUSTOM PROPERTIES — Light mode
   ══════════════════════════════════════════════ */
body.benchdrift-light {
    --bd-bg: #ffffff;
    --bd-bg-card: #f6f8fa;
    --bd-bg-card-hover: #eef1f5;
    --bd-bg-input: #ffffff;
    --bd-border: #d0d7de;
    --bd-border-light: #e1e4e8;
    --bd-text: #000000;
    --bd-text-muted: #333333;
    --bd-text-dim: #555555;
    --bd-text-dimmer: #777777;
    --bd-text-heading: #0550ae;
    --bd-accent-green: #116329;
    --bd-accent-green-bg: rgba(17, 99, 41, 0.12);
    --bd-accent-green-border: rgba(17, 99, 41, 0.35);
    --bd-accent-red: #b51d28;
    --bd-accent-red-bg: rgba(181, 29, 40, 0.1);
    --bd-accent-yellow: #7a5200;
    --bd-accent-yellow-bg: rgba(122, 82, 0, 0.1);
    --bd-accent-yellow-border: rgba(122, 82, 0, 0.35);
    --bd-accent-purple: #6e3fc6;
    --bd-accent-purple-bg: rgba(110, 63, 198, 0.1);
    --bd-accent-blue: #0550ae;
    --bd-accent-blue-bg: rgba(5, 80, 174, 0.1);
    --bd-accent-blue-border: rgba(5, 80, 174, 0.35);
    --bd-accent-orange: #953800;
    --bd-accent-orange-bg: rgba(149, 56, 0, 0.08);
    --bd-accent-orange-border: rgba(149, 56, 0, 0.25);
    --bd-card-variant-bg: #f0f2f5;
    --bd-card-bg: #f6f8fa;
    --bd-baseline-bg: #f0f4f8;
    --bd-badge-off-bg: #f6f8fa;
    --bd-badge-off-color: #999999;
    --bd-badge-off-border: #d0d7de;
    --bd-section-label: #333333;
}

/* ══════════════════════════════════════════════
   Light mode — Override Gradio CSS custom properties
   This is the CORRECT way to theme Gradio 6.x.
   ══════════════════════════════════════════════ */
body.benchdrift-light {
    /* Page background (outside the app container) */
    background: #f0f2f5 !important;
    background-color: #f0f2f5 !important;
    color: #1f2328 !important;

    /* Gradio 6 CSS custom properties */
    --body-background-fill: #f0f2f5 !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #d0d7de !important;
    --block-label-text-color: #656d76 !important;
    --block-title-text-color: #1f2328 !important;
    --body-text-color: #1f2328 !important;
    --body-text-color-subdued: #656d76 !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #d0d7de !important;
    --input-text-color: #1f2328 !important;
    --button-primary-background-fill: #1a7f37 !important;
    --button-primary-background-fill-hover: #2ea043 !important;
    --button-primary-text-color: #ffffff !important;
    --button-secondary-background-fill: #f6f8fa !important;
    --button-secondary-border-color: #d0d7de !important;
    --button-secondary-text-color: #1f2328 !important;
    --checkbox-background-color: #ffffff !important;
    --checkbox-background-color-selected: #0550ae !important;
    --checkbox-border-color: #d0d7de !important;
    --checkbox-border-color-selected: #0550ae !important;
    --checkbox-label-background-fill: #f6f8fa !important;
    --checkbox-label-background-fill-hover: #eef1f5 !important;
    --checkbox-label-background-fill-selected: #ddf4ff !important;
    --checkbox-label-border-color: #d0d7de !important;
    --checkbox-label-border-color-selected: #0550ae !important;
    --checkbox-label-text-color: #1f2328 !important;
    --checkbox-label-text-color-selected: #0550ae !important;
    --radio-circle: #ffffff !important;
    --radio-circle-selected: #0550ae !important;
    --table-even-background-fill: #f6f8fa !important;
    --table-odd-background-fill: #ffffff !important;
    --table-row-focus: #eef1f5 !important;
    --panel-background-fill: #ffffff !important;
    --section-header-text-color: #1f2328 !important;
    --border-color-primary: #d0d7de !important;
    --neutral-50: #f6f8fa !important;
    --neutral-100: #eef1f5 !important;
    --neutral-200: #d0d7de !important;
    --neutral-300: #afb8c1 !important;
    --neutral-400: #8b949e !important;
    --neutral-500: #6e7781 !important;
    --neutral-600: #57606a !important;
    --neutral-700: #424a53 !important;
    --neutral-800: #32383f !important;
    --neutral-900: #1f2328 !important;
    --neutral-950: #0d1117 !important;
    --color-accent: #0550ae !important;
    --color-accent-soft: #ddf4ff !important;
    --link-text-color: #0550ae !important;
    --shadow-drop: 0 1px 3px rgba(31, 35, 40, 0.12) !important;
}

/* Gradio container and main content */
body.benchdrift-light .gradio-container {
    background: #ffffff !important;
}
body.benchdrift-light footer,
body.benchdrift-light .footer {
    background: #f0f2f5 !important;
}

/* Text colors — catch-all for elements that don't use CSS vars */
body.benchdrift-light label,
body.benchdrift-light .label-wrap,
body.benchdrift-light .label-wrap span {
    color: #424a53 !important;
}
body.benchdrift-light p, body.benchdrift-light span, body.benchdrift-light div,
body.benchdrift-light b, body.benchdrift-light strong,
body.benchdrift-light h1, body.benchdrift-light h2, body.benchdrift-light h3,
body.benchdrift-light .prose {
    color: #1f2328;
}
body.benchdrift-light details, body.benchdrift-light summary {
    color: #1f2328 !important;
}

/* Sidebar */
body.benchdrift-light .sidebar,
body.benchdrift-light [class*="sidebar"] {
    background: #f6f8fa !important;
}

/* Dropdown listbox */
body.benchdrift-light ul[role="listbox"] {
    background: #ffffff !important;
    border-color: #d0d7de !important;
}
body.benchdrift-light ul[role="listbox"] li {
    color: #1f2328 !important;
}
body.benchdrift-light ul[role="listbox"] li:hover {
    background: #f6f8fa !important;
}

/* Slider track */
body.benchdrift-light input[type="range"] {
    accent-color: #0550ae;
}

/* Table / Examples */
body.benchdrift-light #benchdrift-examples table { background: #ffffff !important; }
body.benchdrift-light #benchdrift-examples td { background: #f6f8fa !important; color: #1f2328 !important; }
body.benchdrift-light #benchdrift-examples th { background: #ffffff !important; color: #656d76 !important; }
body.benchdrift-light #benchdrift-examples tr:hover td { background: #eef1f5 !important; }

/* Scrollbar */
body.benchdrift-light ::-webkit-scrollbar-track { background: #f6f8fa; }
body.benchdrift-light ::-webkit-scrollbar-thumb { background: #d0d7de; }

/* ══════════════════════════════════════════════
   Light mode — Direct element overrides
   Gradio's make_theme() injects inline <style> with _dark suffix values
   that beat CSS custom properties. These !important rules override those.
   ══════════════════════════════════════════════ */

/* Text inputs, textareas, selects */
body.benchdrift-light input:not([type="checkbox"]):not([type="radio"]):not([type="range"]),
body.benchdrift-light textarea,
body.benchdrift-light select {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-color: #d0d7de !important;
    color: #1f2328 !important;
}
/* Placeholder text */
body.benchdrift-light input::placeholder,
body.benchdrift-light textarea::placeholder {
    color: #8b949e !important;
}

/* Dropdown wrapper — the outer container that looks like an input */
body.benchdrift-light div[data-testid="dropdown"],
body.benchdrift-light div[data-testid="dropdown"] .wrap,
body.benchdrift-light div[data-testid="dropdown"] .secondary-wrap,
body.benchdrift-light .secondary-wrap,
body.benchdrift-light .wrap-inner {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-color: #d0d7de !important;
    color: #1f2328 !important;
}
body.benchdrift-light div[data-testid="dropdown"] input,
body.benchdrift-light div[data-testid="dropdown"] span,
body.benchdrift-light .secondary-wrap input,
body.benchdrift-light .secondary-wrap span {
    color: #1f2328 !important;
}
/* Dropdown caret/arrow SVG */
body.benchdrift-light div[data-testid="dropdown"] svg,
body.benchdrift-light .secondary-wrap svg {
    color: #57606a !important;
    fill: #57606a !important;
    opacity: 1 !important;
}

/* Block/panel backgrounds (accordion, form containers) */
body.benchdrift-light .block,
body.benchdrift-light div[class*="block"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-color: #d0d7de !important;
}

/* Info/helper text below inputs */
body.benchdrift-light span[data-testid="block-info"],
body.benchdrift-light .gr-prose,
body.benchdrift-light .info-text {
    color: #656d76 !important;
}

/* Buttons — secondary (stop, force re-run) */
body.benchdrift-light button.secondary,
body.benchdrift-light .gr-button-secondary {
    background: #f6f8fa !important;
    border-color: #d0d7de !important;
    color: #1f2328 !important;
}

/* Sidebar contents — inputs, dropdowns, sliders inside sidebar */
body.benchdrift-light .sidebar input:not([type="checkbox"]):not([type="radio"]):not([type="range"]),
body.benchdrift-light .sidebar textarea,
body.benchdrift-light .sidebar select,
body.benchdrift-light [class*="sidebar"] input:not([type="checkbox"]):not([type="radio"]):not([type="range"]),
body.benchdrift-light [class*="sidebar"] textarea {
    background: #ffffff !important;
    color: #1f2328 !important;
    border-color: #d0d7de !important;
}
body.benchdrift-light .sidebar div[data-testid="dropdown"],
body.benchdrift-light .sidebar div[data-testid="dropdown"] .wrap,
body.benchdrift-light .sidebar div[data-testid="dropdown"] .secondary-wrap,
body.benchdrift-light .sidebar .secondary-wrap,
body.benchdrift-light [class*="sidebar"] div[data-testid="dropdown"],
body.benchdrift-light [class*="sidebar"] .secondary-wrap {
    background: #ffffff !important;
    border-color: #d0d7de !important;
    color: #1f2328 !important;
}
body.benchdrift-light .sidebar div[data-testid="dropdown"] input,
body.benchdrift-light .sidebar div[data-testid="dropdown"] span,
body.benchdrift-light .sidebar .secondary-wrap input,
body.benchdrift-light .sidebar .secondary-wrap span,
body.benchdrift-light [class*="sidebar"] div[data-testid="dropdown"] input,
body.benchdrift-light [class*="sidebar"] div[data-testid="dropdown"] span {
    color: #1f2328 !important;
}
body.benchdrift-light .sidebar svg,
body.benchdrift-light [class*="sidebar"] svg {
    color: #57606a !important;
    fill: #57606a !important;
}
/* Sidebar block backgrounds */
body.benchdrift-light .sidebar .block,
body.benchdrift-light .sidebar div[class*="block"],
body.benchdrift-light [class*="sidebar"] .block,
body.benchdrift-light [class*="sidebar"] div[class*="block"] {
    background: #f6f8fa !important;
    border-color: #d0d7de !important;
}

/* File upload area */
body.benchdrift-light .file-upload,
body.benchdrift-light [class*="upload"] {
    background: #f6f8fa !important;
    border-color: #d0d7de !important;
    color: #1f2328 !important;
}

/* ══════════════════════════════════════════════
   DARK MODE — dropdown / input text fix
   (scoped to :not(.benchdrift-light) so light mode isn't clobbered)
   ══════════════════════════════════════════════ */
body:not(.benchdrift-light) input[data-testid],
body:not(.benchdrift-light) textarea[data-testid],
body:not(.benchdrift-light) .gr-text-input,
body:not(.benchdrift-light) .gr-input,
body:not(.benchdrift-light) ul[role="listbox"] li,
body:not(.benchdrift-light) div[data-testid="dropdown"] input,
body:not(.benchdrift-light) div[data-testid="dropdown"] span,
body:not(.benchdrift-light) .secondary-wrap input,
body:not(.benchdrift-light) .secondary-wrap span {
    color: #c9d1d9 !important;
}
body:not(.benchdrift-light) ul[role="listbox"] {
    background: #161b22 !important;
    border-color: #30363d !important;
}
body:not(.benchdrift-light) ul[role="listbox"] li {
    color: #c9d1d9 !important;
}
body:not(.benchdrift-light) ul[role="listbox"] li:hover,
body:not(.benchdrift-light) ul[role="listbox"] li[aria-selected="true"] {
    background: #1c2333 !important;
}
body:not(.benchdrift-light) .gr-prose,
body:not(.benchdrift-light) .info-text,
body:not(.benchdrift-light) span.info {
    color: #8b949e !important;
}

/* ── Dark mode: Sidebar / Settings panel — force dark bg ── */
body:not(.benchdrift-light) .sidebar,
body:not(.benchdrift-light) [class*="sidebar"],
body:not(.benchdrift-light) aside,
body:not(.benchdrift-light) aside > div,
body:not(.benchdrift-light) aside [class*="block"],
body:not(.benchdrift-light) aside [class*="panel"],
body:not(.benchdrift-light) aside [class*="form"],
body:not(.benchdrift-light) aside .wrap,
body:not(.benchdrift-light) aside .contain {
    background: #0d1117 !important;
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
}
body:not(.benchdrift-light) aside label,
body:not(.benchdrift-light) aside span,
body:not(.benchdrift-light) aside p,
body:not(.benchdrift-light) aside div,
body:not(.benchdrift-light) aside h1,
body:not(.benchdrift-light) aside h2,
body:not(.benchdrift-light) aside h3 {
    color: #c9d1d9 !important;
}
body:not(.benchdrift-light) aside input,
body:not(.benchdrift-light) aside textarea,
body:not(.benchdrift-light) aside select {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #c9d1d9 !important;
}
body:not(.benchdrift-light) aside div[data-testid="dropdown"],
body:not(.benchdrift-light) aside div[data-testid="dropdown"] .wrap,
body:not(.benchdrift-light) aside div[data-testid="dropdown"] .secondary-wrap,
body:not(.benchdrift-light) aside .secondary-wrap {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
}
body:not(.benchdrift-light) aside div[data-testid="dropdown"] input,
body:not(.benchdrift-light) aside div[data-testid="dropdown"] span,
body:not(.benchdrift-light) aside .secondary-wrap input,
body:not(.benchdrift-light) aside .secondary-wrap span {
    color: #c9d1d9 !important;
}
body:not(.benchdrift-light) aside div[data-testid="dropdown"] svg,
body:not(.benchdrift-light) aside .secondary-wrap svg,
body:not(.benchdrift-light) aside svg {
    color: #8b949e !important;
    fill: #8b949e !important;
    opacity: 1 !important;
}
/* Dark mode sidebar: radio + checkbox */
body:not(.benchdrift-light) aside input[type="radio"],
body:not(.benchdrift-light) aside input[type="checkbox"] {
    accent-color: #60a5fa !important;
    border-color: #30363d !important;
}
/* Dark mode sidebar: slider track */
body:not(.benchdrift-light) aside input[type="range"] {
    accent-color: #60a5fa !important;
}
/* Dark mode sidebar: markdown headers */
body:not(.benchdrift-light) aside .prose h3,
body:not(.benchdrift-light) aside .markdown h3 {
    color: #93c5fd !important;
}
/* Dark mode sidebar: info text */
body:not(.benchdrift-light) aside .gr-prose,
body:not(.benchdrift-light) aside span[data-testid="block-info"],
body:not(.benchdrift-light) aside .info-text {
    color: #8b949e !important;
}

/* ── Dark mode: visible outlines on all inputs ── */
body:not(.benchdrift-light) input,
body:not(.benchdrift-light) textarea,
body:not(.benchdrift-light) select {
    border-color: #30363d !important;
}
body:not(.benchdrift-light) div[data-testid="dropdown"],
body:not(.benchdrift-light) div[data-testid="dropdown"] .wrap,
body:not(.benchdrift-light) div[data-testid="dropdown"] .secondary-wrap,
body:not(.benchdrift-light) .secondary-wrap {
    border: 1px solid #30363d !important;
    border-radius: 4px !important;
}
body:not(.benchdrift-light) div[data-testid="dropdown"] svg,
body:not(.benchdrift-light) .secondary-wrap svg {
    color: #8b949e !important;
    fill: #8b949e !important;
    opacity: 1 !important;
}
/* Dark mode: checkbox / radio outlines */
body:not(.benchdrift-light) input[type="checkbox"],
body:not(.benchdrift-light) input[type="radio"] {
    accent-color: #60a5fa !important;
    border-color: #30363d !important;
}

/* ══════════════════════════════════════════════
   LAYOUT — using CSS custom properties
   ══════════════════════════════════════════════ */

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', 'Cascadia Code', monospace !important;
}

.app-header {
    text-align: center; padding: 24px 0 12px 0;
    border-bottom: 1px solid var(--bd-border); margin-bottom: 20px;
}
.app-header h1 { font-size: 1.6em; margin: 0; letter-spacing: 2px; text-transform: uppercase; font-weight: 700; color: var(--bd-text); }
.app-header .subtitle { font-size: 0.85em; opacity: 0.5; margin-top: 4px; color: var(--bd-text-muted); }
.app-header .header-actions { margin-top: 8px; }
.app-header .header-actions button {
    background: transparent; border: 1px solid var(--bd-border);
    color: var(--bd-text-muted); padding: 2px 12px; border-radius: 3px;
    font-size: 0.72em; font-family: monospace; cursor: pointer;
}
.app-header .header-actions button:hover { color: var(--bd-text); border-color: var(--bd-text-dim); }

/* Section labels */
.section-label {
    font-size: 0.82em; color: var(--bd-text-muted); text-transform: uppercase;
    letter-spacing: 2px; font-weight: 700; margin: 20px 0 8px 0; font-family: monospace;
    padding-bottom: 6px; border-bottom: 2px solid var(--bd-border);
}

/* Feature badges — clickable */
.feature-badges { display: flex; flex-wrap: wrap; gap: 5px; margin: 6px 0; }
.feature-badge {
    display: inline-block; padding: 2px 8px; border-radius: 3px;
    font-size: 0.75em; font-family: monospace;
    cursor: pointer; user-select: none; transition: all 0.15s ease;
}
.feature-badge:hover { opacity: 0.8; transform: scale(1.05); }
.feature-badge:active { transform: scale(0.97); }
.badge-on { background: var(--bd-accent-green-bg); color: var(--bd-accent-green); border: 1px solid var(--bd-accent-green-border); }
.badge-off { background: var(--bd-badge-off-bg); color: var(--bd-badge-off-color); border: 1px solid var(--bd-badge-off-border); }
.badge-llm-on { background: var(--bd-accent-blue-bg); color: var(--bd-accent-blue); border: 1px solid var(--bd-accent-blue-border); }
.badge-llm-off { background: var(--bd-badge-off-bg); color: var(--bd-text-dimmer); border: 1px solid var(--bd-border-light); }
.badge-disc-on { background: var(--bd-accent-yellow-bg); color: var(--bd-accent-yellow); border: 1px solid var(--bd-accent-yellow-border); }
.badge-disc-off { background: var(--bd-badge-off-bg); color: var(--bd-text-dimmer); border: 1px solid var(--bd-border-light); }
.badge-section-label {
    font-size: 0.65em; color: var(--bd-section-label); text-transform: uppercase;
    letter-spacing: 1px; margin: 8px 0 3px 0; font-family: monospace;
}
#features-sync-box { display: none !important; }

/* Axis-grouped analysis cards */
.axis-group { border: 1px solid var(--bd-border-light); border-radius: 4px; margin: 6px 0; padding: 10px 12px; }
.axis-group-header { display: flex; align-items: center; gap: 8px; }
.axis-group-rank { font-size: 0.7em; color: var(--bd-text-dim); font-weight: 600; min-width: 20px; }
.axis-group-name { font-weight: 700; color: var(--bd-text-heading); font-size: 0.9em; }
.axis-group-slots { font-size: 0.65em; color: var(--bd-text-dim); margin-left: auto; font-family: monospace; }
.axis-group-desc { font-size: 0.7em; color: var(--bd-text-dim); margin-top: 2px; margin-left: 28px; }
.axis-group-features { margin-top: 6px; margin-left: 28px; }
.axis-group-transforms { font-size: 0.72em; color: var(--bd-text-dimmer); margin-top: 4px; margin-left: 28px; line-height: 1.6; }
.unassigned-section { border: 1px dashed var(--bd-border-light); border-radius: 4px; margin: 8px 0; padding: 8px 12px; }

/* Referential entity display */
.ref-entities { margin-top: 6px; margin-left: 28px; font-size: 0.78em; font-family: monospace; }
.ref-fragment {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; margin: 2px 3px; border-radius: 3px;
    background: var(--bd-accent-orange-bg); border: 1px solid var(--bd-accent-orange-border); color: var(--bd-accent-orange);
}
.ref-fragment-type { font-size: 0.8em; color: #777; }
.ref-entity {
    display: inline-block; padding: 1px 6px; margin: 1px 2px; border-radius: 2px;
    background: var(--bd-accent-orange-bg); border: 1px solid rgba(251, 146, 60, 0.15);
    color: #d97706; font-size: 0.85em;
}
.ref-label { color: var(--bd-text-dim); font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.5px; margin-right: 4px; }

/* Result cards — 2-column grid */
.results-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
}
.result-card {
    border: 1px solid var(--bd-border); border-radius: 4px; padding: 10px 12px;
    font-family: monospace; font-size: 0.85em; background: var(--bd-card-bg);
}
.card-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 6px; padding-bottom: 6px; border-bottom: 1px solid var(--bd-border-light);
}
.card-name { font-weight: 600; color: var(--bd-text-heading); }
.card-variant {
    padding: 8px 10px; margin: 6px 0; border-radius: 3px;
    border-left: 2px solid var(--bd-text-dimmer); line-height: 1.5;
    background: var(--bd-card-variant-bg); color: var(--bd-text-muted); white-space: pre-wrap; word-wrap: break-word;
}

/* Status badges */
.st { display: inline-block; padding: 1px 8px; border-radius: 2px; font-size: 0.82em; font-weight: 600; }
.st-ok { background: var(--bd-accent-green-bg); color: var(--bd-accent-green); }
.st-drift { background: var(--bd-accent-red-bg); color: var(--bd-accent-red); }
.st-wait { background: var(--bd-accent-yellow-bg); color: var(--bd-accent-yellow); }
.st-err { background: var(--bd-accent-purple-bg); color: var(--bd-accent-purple); }
.st-bwrong { background: rgba(148, 163, 184, 0.15); color: #94a3b8; }
.st-neutral { background: rgba(148, 163, 184, 0.1); color: #6b7280; }
.st-pos-drift { background: var(--bd-accent-green-bg); color: var(--bd-accent-green); }

.card-answer { margin-top: 4px; color: var(--bd-text-muted); font-size: 0.85em; }
.card-summary-line {
    display: flex; align-items: center; gap: 8px;
    margin-top: 4px; color: var(--bd-text-muted); font-size: 0.85em;
}
.card-divider { color: var(--bd-border); }
.card-expand { margin-top: 6px; }
.card-expand > summary {
    color: var(--bd-text-dim); font-size: 0.78em; cursor: pointer;
    user-select: none; padding: 2px 0;
}
.card-expand > summary:hover { color: var(--bd-text-heading); }
.card-details-body { margin-top: 4px; }
.card-detail-section { margin-bottom: 6px; }
.card-detail-label { font-size: 0.72em; color: var(--bd-text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }

.summary-box {
    border: 1px solid var(--bd-border); border-radius: 4px; padding: 14px;
    margin-top: 12px; font-family: monospace; font-size: 0.88em; background: var(--bd-card-bg);
}
.summary-box .sum-title { font-weight: 700; color: var(--bd-text-heading); margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid var(--bd-border-light); }
.summary-box p { margin: 3px 0; color: var(--bd-text-muted); }
.summary-box b { color: var(--bd-text); }

.baseline-card {
    border: 1px solid var(--bd-border); border-radius: 4px; padding: 10px 14px;
    margin-bottom: 12px; font-family: monospace; font-size: 0.88em;
    background: var(--bd-baseline-bg); border-left: 3px solid var(--bd-text-heading);
}

/* Drift bar chart */
.drift-chart { margin-top: 12px; }

/* Axes checkbox group */
.axes-checkboxgroup .wrap { gap: 6px !important; }
.axes-checkboxgroup label { font-size: 0.82em !important; }

/* Example table */
#benchdrift-examples { background: transparent !important; }
#benchdrift-examples table { background: var(--bd-bg) !important; border: 1px solid var(--bd-border) !important; }
#benchdrift-examples td { background: var(--bd-bg-card) !important; color: var(--bd-text) !important; font-size: 0.85em; border-color: var(--bd-border) !important; }
#benchdrift-examples th { background: var(--bd-bg) !important; color: var(--bd-text-muted) !important; border-color: var(--bd-border) !important; }
#benchdrift-examples tr:hover td { background: var(--bd-bg-card-hover) !important; }
#benchdrift-examples .gallery-item { background: var(--bd-bg-card) !important; border: 1px solid var(--bd-border) !important; color: var(--bd-text) !important; }
#benchdrift-examples .gallery-item:hover { background: var(--bd-bg-card-hover) !important; }

/* ══════════════════════════════════════════════
   RESPONSIVE — Mobile & Tablet
   ══════════════════════════════════════════════ */

@media (max-width: 768px) {
    .gradio-container { max-width: 100% !important; padding: 0 8px !important; }
    .input-row { flex-direction: column !important; }
    .input-row > .gr-column { min-width: 100% !important; }
    .app-header h1 { font-size: 1.2em; letter-spacing: 1px; }
    .app-header .subtitle { font-size: 0.78em; }
    .action-buttons { flex-direction: column !important; gap: 6px !important; }
    .action-buttons > button { width: 100% !important; }
    .hf-config-row { flex-wrap: wrap !important; }
    .hf-config-row > * { min-width: 45% !important; flex: 1 1 45% !important; }
    .axis-group-desc, .axis-group-features, .axis-group-transforms { margin-left: 8px !important; }
    .drift-bar-name { width: 25%; min-width: 70px; font-size: 0.7em; }
    .feature-badge { padding: 4px 10px; font-size: 0.82em; }
    .results-grid { grid-template-columns: 1fr !important; }
    .result-card { padding: 8px 10px; }
    .card-variant { padding: 6px 8px; }
    .axis-group-desc { font-size: 0.76em; }
    .axis-group-transforms { font-size: 0.78em; }
    .badge-section-label { font-size: 0.72em; }
}

@media (max-width: 480px) {
    .gradio-container { padding: 0 4px !important; }
    .app-header { padding: 16px 0 8px 0; }
    .app-header h1 { font-size: 1.0em; }
    .hf-config-row > * { min-width: 100% !important; flex: 1 1 100% !important; }
}
"""


def make_theme():
    """Build and return the Gradio theme with dark developer styling."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="#0d1117",
        body_background_fill_dark="#0d1117",
        block_background_fill="#161b22",
        block_background_fill_dark="#161b22",
        block_border_color="#30363d",
        block_border_color_dark="#30363d",
        block_label_text_color="#8b949e",
        block_label_text_color_dark="#8b949e",
        block_title_text_color="#c9d1d9",
        block_title_text_color_dark="#c9d1d9",
        body_text_color="#c9d1d9",
        body_text_color_dark="#c9d1d9",
        body_text_color_subdued="#8b949e",
        body_text_color_subdued_dark="#8b949e",
        input_background_fill="#0d1117",
        input_background_fill_dark="#0d1117",
        input_border_color="#30363d",
        input_border_color_dark="#30363d",
        button_primary_background_fill="#238636",
        button_primary_background_fill_dark="#238636",
        button_primary_background_fill_hover="#2ea043",
        button_primary_background_fill_hover_dark="#2ea043",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#21262d",
        button_secondary_background_fill_dark="#21262d",
        button_secondary_border_color="#30363d",
        button_secondary_border_color_dark="#30363d",
        button_secondary_text_color="#c9d1d9",
        button_secondary_text_color_dark="#c9d1d9",
    )

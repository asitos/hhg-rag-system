import re

with open("frontend/index.html", "r") as f:
    content = f.read()

dropdown_html = """
            <div id="demo-scenario-container" class="hidden" style="margin-top: 16px; background: rgba(0,0,0,0.1); padding: 12px; border-radius: 4px; border: 1px dashed var(--ink);">
                <label style="font: 500 11px 'DM Mono', monospace; text-transform: uppercase; margin-bottom: 4px; display: block;">Demo Scenario (Backend Simulation)</label>
                <select id="demo-scenario-select" style="width: 100%; padding: 8px; background: var(--input-bg); color: var(--input-text); border: 1px solid var(--ink);">
                    <option value="english">English RAG (Default)</option>
                    <option value="hindi">Hindi RAG</option>
                    <option value="off_topic">Off-Topic Guardrail</option>
                    <option value="no_context">Grounding / No Context</option>
                    <option value="grounding_failure">Grounding Failure (Bad Citation)</option>
                </select>
            </div>
"""

# Insert right after the tabs in the input deck
content = content.replace('<div class="tabs">', dropdown_html + '\n                <div class="tabs">')

with open("frontend/index.html", "w") as f:
    f.write(content)

with open("frontend/js/app.js", "r") as f:
    js_content = f.read()
    
# Add logic to fetch and set scenario
js_scenario = """
    const demoScenarioContainer = document.getElementById('demo-scenario-container');
    const demoScenarioSelect = document.getElementById('demo-scenario-select');
    
    demoScenarioSelect.addEventListener('change', async (e) => {
        try {
            await fetch(`${API_BASE}/api/v1/scenario`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scenario: e.target.value })
            });
        } catch (err) {
            console.error("Failed to set scenario", err);
        }
    });
"""

js_content = js_content.replace("// Theme Toggle", js_scenario + "\n    // Theme Toggle")

# Make sure the container shows up when mode is 'demo'
health_patch_old = """                if (data.mode === "mock") {
                    demoBanner.classList.remove("hidden");
                }"""
health_patch_new = """                if (data.mode === "mock" || data.mode === "demo") {
                    demoBanner.classList.remove("hidden");
                    demoBanner.textContent = `MODE: ${data.mode.toUpperCase()} (Using local backend RAG with API mocks)`;
                    demoScenarioContainer.classList.remove("hidden");
                }"""

js_content = js_content.replace(health_patch_old, health_patch_new)

with open("frontend/js/app.js", "w") as f:
    f.write(js_content)

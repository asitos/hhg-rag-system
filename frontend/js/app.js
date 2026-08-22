document.addEventListener('DOMContentLoaded', () => {
    // Determine if we are running locally with a backend, or strictly as a static frontend
    const API_BASE = window.APP_CONFIG?.API_BASE_URL !== undefined ? window.APP_CONFIG.API_BASE_URL : "";
    let isDemoMode = false;
    
    // UI Elements
    const themeBtn = document.getElementById('theme-btn');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    const recordBtn = document.getElementById('record-btn');
    const recordingStatus = document.getElementById('recording-status');
    const textInput = document.getElementById('text-input');
    const submitTextBtn = document.getElementById('submit-text-btn');
    
    const resultContainer = document.getElementById('result-container');
    const loadingState = document.getElementById('loading-state');
    const errorState = document.getElementById('error-state');
    
    const outTranscript = document.getElementById('transcript-out');
    const outAnswer = document.getElementById('answer-out');
    const outSources = document.getElementById('sources-out');
    
    const outHealth = document.getElementById('health-out');
    const outGuardrail = document.getElementById('guardrail-out');
    const outSttLat = document.getElementById('stt-lat-out');
    const outTotalLat = document.getElementById('total-lat-out');
    const demoBanner = document.getElementById('demo-banner');
    
    // Pipeline UI
    const pipelineContainer = document.createElement('div');
    pipelineContainer.className = 'pipeline-viz hidden';
    pipelineContainer.id = 'pipeline-viz';
    pipelineContainer.innerHTML = `
        <div class="pipeline-stage" id="stage-stt">● STT</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-stage" id="stage-retrieval">● RETRIEVAL</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-stage" id="stage-rerank">● RERANK</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-stage" id="stage-generation">● GENERATION</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-stage" id="stage-guardrails">● GUARDRAILS</div>
    `;
    document.querySelector('.output-deck').insertBefore(pipelineContainer, loadingState);

    
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

    // Theme Toggle
    themeBtn.addEventListener('click', () => {
        const isDark = document.body.classList.toggle('dark');
        themeBtn.textContent = isDark ? '☀️' : '🌙';
    });

    // Tabs
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
        });
    });

    // Health Check & Demo Mode Fallback
    async function checkHealth() {
        if (!API_BASE && window.location.protocol.startsWith('http') && window.location.port !== "8000") {
            // Force demo mode for GitHub Pages or static host
            enableDemoMode();
            return;
        }
        try {
            const res = await fetch(`${API_BASE}/health`);
            if (res.ok) {
                const data = await res.json();
                outHealth.textContent = "Online";
                outHealth.style.color = "var(--success)";
                if (data.mode === "mock" || data.mode === "demo") {
                    demoBanner.classList.remove("hidden");
                    demoBanner.textContent = `MODE: ${data.mode.toUpperCase()} (Using local backend RAG with API mocks)`;
                    demoScenarioContainer.classList.remove("hidden");
                }
            } else {
                throw new Error("Bad status");
            }
        } catch (e) {
            enableDemoMode();
        }
    }
    
    function enableDemoMode() {
        isDemoMode = true;
        outHealth.textContent = "Static Demo";
        outHealth.style.color = "var(--gold)";
        demoBanner.classList.remove("hidden");
        demoBanner.textContent = "STATIC DEMO MODE: Backend APIs are disabled. Running purely in browser.";
    }
    
    checkHealth();

    // UI States
    function showLoading() {
        resultContainer.classList.add('hidden');
        errorState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        pipelineContainer.classList.remove('hidden');
        
        document.querySelectorAll('.pipeline-stage').forEach(el => {
            el.className = 'pipeline-stage';
        });
        
        outSttLat.textContent = "0.00s";
        outTotalLat.textContent = "0.00s";
        outGuardrail.textContent = "-";
    }

    function setStage(stageId, status = 'active') {
        const el = document.getElementById(`stage-${stageId}`);
        if (el) el.className = `pipeline-stage ${status}`;
    }

    function showError(msg) {
        loadingState.classList.add('hidden');
        resultContainer.classList.add('hidden');
        errorState.classList.remove('hidden');
        document.getElementById('error-text').textContent = msg;
    }

    function showResult(data) {
        loadingState.classList.add('hidden');
        errorState.classList.add('hidden');
        resultContainer.classList.remove('hidden');
        
        outTranscript.textContent = data.transcript || "No transcript available.";
        outAnswer.textContent = data.answer || "No answer generated.";
        
        if (data.sources && data.sources.length > 0) {
            outSources.innerHTML = data.sources.map(s => `[${s.chunk_id}] ${s.text} <br><small>Score: ${parseFloat(s.score).toFixed(2)}</small>`).join('<hr style="border-top: 1px solid rgba(128,128,128,0.2); margin: 8px 0;">');
        } else {
            outSources.textContent = "No sources retrieved.";
        }
        
        if (data.latency) {
            const prefix = isDemoMode ? "(UI sim) " : "";
            outSttLat.textContent = prefix + (data.latency.stt_ms / 1000).toFixed(2) + "s";
            outTotalLat.textContent = prefix + (data.latency.total_ms / 1000).toFixed(2) + "s";
        }
        
        outGuardrail.textContent = data.guardrail || "PASS";
        if (data.guardrail !== "PASS") {
            outGuardrail.style.color = "var(--error)";
            outAnswer.textContent += `\n\nGuardrail reason: ${data.guardrail_reason || "Unknown"}`;
            setStage('guardrails', 'error');
        } else {
            outGuardrail.style.color = "var(--success)";
            setStage('guardrails', 'success');
        }
    }

    // --- STATIC DEMO PIPELINE ---
    async function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
    
    async function runStaticDemoPipeline(query, isVoice = false) {
        let sttMs = 0;
        
        if (isVoice) {
            setStage('stt', 'active');
            await sleep(300);
            sttMs = 300;
            setStage('stt', 'success');
            // Mock STT logic if no query provided
            if (!query) query = "What is a corporation?";
        }
        
        let lowerQuery = query.toLowerCase();
        
        // Pre-guardrail
        if (lowerQuery.includes("weather")) {
            setStage('guardrails', 'error');
            return {
                transcript: query,
                answer: "OFF-TOPIC: I can only answer questions supported by the indexed dataset.",
                sources: [],
                guardrail: "FAIL_OFFTOPIC",
                latency: { stt_ms: sttMs, total_ms: sttMs + 50 }
            };
        }
        
        setStage('retrieval', 'active');
        await sleep(100);
        let sources = [];
        try {
            const res = await fetch('assets/mock-data.json');
            const data = await res.json();
            
            // Simple mock semantic search
            if (lowerQuery.includes("corporation")) {
                sources.push(data.find(d => d.id === "demo_001"));
            } else if (lowerQuery.includes("india") || lowerQuery.includes("capital")) {
                sources.push(data.find(d => d.id === "demo_002"));
            } else if (lowerQuery.includes("भारत") || lowerQuery.includes("राजधानी")) {
                sources.push(data.find(d => d.id === "demo_003"));
            } else if (lowerQuery.includes("mutual fund")) {
                sources.push(data.find(d => d.id === "demo_004"));
            }
        } catch (e) {
            console.error("Failed to load mock data:", e);
        }
        setStage('retrieval', 'success');
        
        setStage('rerank', 'active');
        await sleep(50);
        sources = sources.filter(s => s).map(s => ({ chunk_id: s.id, text: s.text, score: 0.94 }));
        setStage('rerank', 'success');
        
        setStage('generation', 'active');
        await sleep(500);
        setStage('generation', 'success');
        
        setStage('guardrails', 'active');
        await sleep(50);
        
        if (sources.length === 0) {
            return {
                transcript: query,
                answer: "I cannot answer this from the available context.",
                sources: [],
                guardrail: "FAIL_GROUNDING",
                guardrail_reason: "No context",
                latency: { stt_ms: sttMs, total_ms: sttMs + 100 + 50 + 500 + 50 }
            };
        }
        
        return {
            transcript: query,
            answer: `Based on the retrieved context:\n\n[${sources[0].chunk_id}] ${sources[0].text}\n\nThis answer is generated entirely in the browser for the GitHub Pages demo.`,
            sources: sources,
            guardrail: "PASS",
            latency: { stt_ms: sttMs, total_ms: sttMs + 100 + 50 + 500 + 50 }
        };
    }

    // Text Submission
    submitTextBtn.addEventListener('click', async () => {
        const query = textInput.value.trim();
        if (!query) return;
        
        showLoading();
        if (isDemoMode) {
            const data = await runStaticDemoPipeline(query, false);
            showResult(data);
            return;
        }
        
        try {
            const formData = new FormData();
            formData.append('query', query);
            
            const res = await fetch(`${API_BASE}/api/v1/text`, {
                method: 'POST',
                body: formData
            });
            
            if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
            const data = await res.json();
            showResult(data);
        } catch (e) {
            showError(`Failed to process text query: ${e.message}`);
        }
    });

    // Voice Recording
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;

    recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            // Start recording
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.addEventListener('dataavailable', event => {
                    audioChunks.push(event.data);
                });
                
                mediaRecorder.addEventListener('stop', async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    stream.getTracks().forEach(track => track.stop());
                    
                    showLoading();
                    
                    if (isDemoMode) {
                        // Pass an empty string so the demo pipeline picks a default mock query
                        const data = await runStaticDemoPipeline("", true);
                        showResult(data);
                        return;
                    }
                    
                    try {
                        const formData = new FormData();
                        formData.append('audio', audioBlob, 'recording.webm');
                        
                        const res = await fetch(`${API_BASE}/api/v1/voice`, {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                        const data = await res.json();
                        showResult(data);
                    } catch (e) {
                        showError(`Failed to process voice query: ${e.message}`);
                    }
                });
                
                mediaRecorder.start();
                isRecording = true;
                recordBtn.innerHTML = '<span class="icon">🛑</span> Stop Recording';
                recordBtn.style.background = 'var(--error)';
                recordBtn.style.color = 'white';
                recordingStatus.classList.remove('hidden');
                
            } catch (err) {
                showError("Microphone access denied or not supported in this browser.");
            }
        } else {
            // Stop recording
            mediaRecorder.stop();
            isRecording = false;
            recordBtn.innerHTML = '<span class="icon">🎤</span> Start Recording';
            recordBtn.style.background = 'var(--btn-primary)';
            recordBtn.style.color = 'var(--btn-primary-text)';
            recordingStatus.classList.add('hidden');
        }
    });
});

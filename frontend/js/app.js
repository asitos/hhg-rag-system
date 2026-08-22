document.addEventListener('DOMContentLoaded', () => {
    const API_BASE = window.APP_CONFIG?.API_BASE_URL || 'http://localhost:8000';
    
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

    // Health Check
    async function checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/health`);
            if (res.ok) {
                outHealth.textContent = "Online";
                outHealth.style.color = "var(--success)";
            } else {
                throw new Error("Bad status");
            }
        } catch (e) {
            outHealth.textContent = "Unavailable";
            outHealth.style.color = "var(--error)";
        }
    }
    checkHealth();

    // UI States
    function showLoading() {
        resultContainer.classList.add('hidden');
        errorState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        
        outSttLat.textContent = "0.00s";
        outTotalLat.textContent = "0.00s";
        outGuardrail.textContent = "-";
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
            outSttLat.textContent = (data.latency.stt_ms / 1000).toFixed(2) + "s";
            outTotalLat.textContent = (data.latency.total_ms / 1000).toFixed(2) + "s";
        }
        
        outGuardrail.textContent = data.guardrail || "PASS";
        if (data.guardrail !== "PASS") {
            outGuardrail.style.color = "var(--error)";
            outAnswer.textContent += `\n\nGuardrail reason: ${data.guardrail_reason || "Unknown"}`;
        } else {
            outGuardrail.style.color = "var(--success)";
        }
    }

    // Text Submission
    submitTextBtn.addEventListener('click', async () => {
        const query = textInput.value.trim();
        if (!query) return;
        
        showLoading();
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
                    // Stop tracks to release microphone
                    stream.getTracks().forEach(track => track.stop());
                    
                    showLoading();
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

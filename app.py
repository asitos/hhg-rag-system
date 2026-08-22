import gradio as gr
import os

from bench.bench import build_system
from src.stt.sarvam import SarvamSTT
from src.models import PipelineInput

# Initialize system globally
try:
    pipeline, _ = build_system()
except Exception as e:
    pipeline = None
    print(f"Warning: Could not initialize pipeline. Ensure index exists. Error: {e}")

stt_client = SarvamSTT()

def process_audio(audio_path):
    if not pipeline:
        return "Pipeline not initialized.", "", 0.0, 0.0
    if not audio_path:
        return "No audio provided.", "", 0.0, 0.0
        
    try:
        transcript, stt_latency = stt_client.transcribe(audio_path)
        if not transcript:
            return "Could not transcribe audio. Please try again.", "", 0.0, 0.0
            
        p_input = PipelineInput(query=transcript)
        response = pipeline.execute(p_input)
        
        sources = "\n".join([f"[{s.chunk_id}] {s.text}" for s in response.sources])
        total_lat = (response.latency.get("post_stt_ms", 0) + stt_latency) / 1000
        return response.answer, sources, total_lat, stt_latency / 1000
    except Exception as e:
        import logging
        logging.error(f"Audio processing error: {str(e)}", exc_info=True)
        return f"Processing error: {str(e)[:100]}", "", 0.0, 0.0

def process_text(text):
    if not pipeline:
        return "Pipeline not initialized.", "", 0.0, 0.0
    if not text:
        return "No text provided.", "", 0.0, 0.0
        
    try:
        p_input = PipelineInput(query=text)
        response = pipeline.execute(p_input)
        
        sources = "\n".join([f"[{s.chunk_id}] {s.text}" for s in response.sources])
        total_lat = response.latency.get("post_stt_ms", 0) / 1000
        return response.answer, sources, total_lat, 0.0
    except Exception as e:
        return f"Error: {str(e)}", "", 0.0, 0.0

ui_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --ink: #16130f;
    --panel: #e8c875;
    --panel-light: #f2d994;
    --line: #3c2b16;
    --gold: #d6a84f;
    --champagne: #f3d58a;
    --paper: #17130d;
}

body, .gradio-container {
    background: var(--gold) !important;
    color: var(--ink) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 24px 32px 44px !important;
    background-image: linear-gradient(rgba(22, 19, 15, .07) 1px, transparent 1px), linear-gradient(90deg, rgba(22, 19, 15, .07) 1px, transparent 1px) !important;
    background-size: 34px 34px !important;
}

.console {
    perspective: 1200px;
    transform-style: preserve-3d;
}

.masthead {
    position: relative;
    min-height: 172px;
    padding: 32px 38px;
    overflow: hidden;
    border: 2px solid var(--ink);
    border-left: 8px solid var(--ink);
    background: var(--panel-light);
    box-shadow: 0 8px 18px rgba(22, 19, 15, .16);
}

.masthead:after {
    content: 'LIVE / RAG-02';
    position: absolute;
    right: 28px;
    top: 24px;
    color: var(--ink);
    font: 500 11px 'DM Mono', monospace;
    letter-spacing: 2px;
}

.eyebrow {
    color: var(--ink);
    font: 500 11px 'DM Mono', monospace;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.masthead h1 {
    max-width: 690px;
    margin: 14px 0 8px;
    color: var(--ink);
    font-size: clamp(30px, 5vw, 62px);
    line-height: .98;
    letter-spacing: 0;
}

.masthead p { max-width: 580px; margin: 0; color: #493618; font-size: 15px; }
.workbench { gap: 24px !important; margin-top: 36px; align-items: stretch !important; }
.input-deck, .output-deck { transform-style: preserve-3d; }
.panel {
    position: relative;
    min-height: 330px;
    padding: 24px !important;
    border: 2px solid var(--ink) !important;
    background: var(--panel-light) !important;
    box-shadow: 0 8px 18px rgba(22, 19, 15, .14);
}
.panel:before {
    content: '';
    position: absolute;
    inset: 8px;
    border: 1px solid rgba(22, 19, 15, .16);
    pointer-events: none;
}
.section-label { color: var(--ink); font: 500 11px 'DM Mono', monospace; letter-spacing: 2px; text-transform: uppercase; }
.panel h2 { margin: 7px 0 14px; color: var(--ink); font-size: 22px; letter-spacing: 0; }
.tabs { border-bottom: 1px solid var(--line) !important; }
.tabs button { color: #6b5128 !important; font: 500 12px 'DM Mono', monospace !important; }
.tabs button.selected { color: var(--ink) !important; border-color: var(--ink) !important; }
textarea, input, .audio-container { border-color: var(--ink) !important; background: #f8e9b8 !important; color: var(--paper) !important; border-radius: 2px !important; }
label span { color: #493618 !important; font: 500 11px 'DM Mono', monospace !important; text-transform: uppercase; }
button.primary { border-radius: 2px !important; background: var(--ink) !important; color: var(--champagne) !important; font-weight: 700 !important; box-shadow: 5px 5px 0 #6b5128; }
button.primary:hover { box-shadow: 7px 7px 0 #6b5128; }
.telemetry { margin-top: 24px; gap: 24px !important; }
.metric { border-top: 2px solid var(--ink); padding-top: 12px !important; background: transparent !important; }
.metric input { border: 0 !important; padding-left: 0 !important; font: 500 22px 'DM Mono', monospace !important; }
@media (max-width: 800px) {
    .gradio-container { padding: 12px !important; }
    .masthead { padding: 24px; min-height: 190px; box-shadow: 0 6px 14px rgba(22, 19, 15, .14); }
    .masthead:after { right: 16px; top: 16px; }
    .workbench { gap: 16px !important; margin-top: 24px; }
    .panel { padding: 18px !important; }
}
"""

with gr.Blocks(
    title="HH Goa 2026 Voice RAG",
    css=ui_css,
    theme=gr.themes.Base(),
) as demo:
    with gr.Column(elem_classes=["console"]):
        gr.HTML("""
        <header class="masthead">
            <div class="eyebrow">Hacker House Goa / Voice Intelligence</div>
            <h1>Ask the knowledge base.</h1>
            <p>Multilingual retrieval, grounded answers, and source traces in one instrument panel.</p>
        </header>
        """)

        with gr.Row(elem_classes=["workbench"]):
            with gr.Column(elem_classes=["panel", "input-deck"]):
                gr.HTML('<div class="section-label">01 / Query input</div><h2>Send a signal</h2>')
                with gr.Tabs(elem_classes=["tabs"]):
                    with gr.Tab("VOICE"):
                        audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Record query")
                        btn_voice = gr.Button("Transmit voice", variant="primary")
                    with gr.Tab("TEXT"):
                        text_in = gr.Textbox(label="Type query", lines=5, placeholder="Ask about the corpus...")
                        btn_text = gr.Button("Run retrieval", variant="primary")

            with gr.Column(elem_classes=["panel", "output-panel"]):
                gr.HTML('<div class="section-label">02 / Grounded response</div><h2>Signal decoded</h2>')
                answer_out = gr.Textbox(label="Answer", lines=6)
                sources_out = gr.Textbox(label="Retrieved sources", lines=5)

        with gr.Row(elem_classes=["telemetry"]):
            with gr.Column(elem_classes=["metric"]):
                lat_out = gr.Number(label="Total latency (s)")
            with gr.Column(elem_classes=["metric"]):
                stt_out = gr.Number(label="STT latency (s)")

    btn_voice.click(process_audio, inputs=[audio_in], outputs=[answer_out, sources_out, lat_out, stt_out])
    btn_text.click(process_text, inputs=[text_in], outputs=[answer_out, sources_out, lat_out, stt_out])

if __name__ == "__main__":
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=server_port)

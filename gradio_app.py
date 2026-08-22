import gradio as gr
import asyncio
import os
from src.config import settings

# Force demo mode
settings.app_mode = "demo"

from src.pipeline.harness import RAGPipeline
from src.retrieval.vector_store import VectorStore
from src.retrieval.embedder import Embedder
from src.models import PipelineInput

# 1. Initialize DB and Models
print("Loading embeddings...")
embedder = Embedder(settings.embedding_model_id)

# Initialize VectorStore - it will load from disk
print("Connecting to Qdrant...")
store = VectorStore(persist_dir="data/qdrant_db")
count = store.client.count(store.collection_name).count
if count == 0:
    print("WARNING: Qdrant DB is empty. Building demo index now...")
    # Add fixture dataset
    import json
    with open("tests/fixtures/passages.json", "r") as f:
        passages = json.load(f)
    
    vecs = []
    payloads = []
    for p in passages:
        vecs.append(embedder.embed_query(p["text"]))
        payloads.append({
            "chunk_id": p["id"],
            "text": p["text"],
            "language": p["language"],
            "strategy": "demo"
        })
    store.add_vectors(vecs, payloads)
    print(f"Added {len(vecs)} vectors to Qdrant demo index.")

# Create pipeline
pipeline = RAGPipeline(store)
pipeline.embedder = embedder # Pass the loaded embedder

# --- Gradio UI ---
css = """
body { font-family: 'Space Grotesk', sans-serif; }
.sources-card { border: 1px solid #444; padding: 10px; margin-top: 10px; border-radius: 4px; background: #222; }
.status-badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; font-family: monospace; }
.success-badge { background: #1b5e20; color: #a5d6a7; }
.error-badge { background: #b71c1c; color: #ffcdd2; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(text_size="lg"), css=css) as demo:
    gr.Markdown(f"# 🎙️ HH Goa Voice RAG (Local Mode)\n**APP_MODE={settings.app_mode.upper()}** - External APIs Disabled. Using Real Local RAG.")
    
    with gr.Row():
        scenario_dropdown = gr.Dropdown(
            choices=["english", "hindi", "no_context", "off_topic", "grounding_failure"],
            value="english",
            label="Video Demo Scenario",
            info="Select deterministic behavior for Mock STT and Generator"
        )
        
    def set_scenario(val):
        settings.demo_scenario = val
        return val
        
    scenario_dropdown.change(set_scenario, scenario_dropdown, None)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Voice Input")
            text_input = gr.Textbox(label="Text Input", placeholder="Or type a query here...")
            submit_btn = gr.Button("Run Pipeline", variant="primary")
            
        with gr.Column(scale=2):
            transcript_out = gr.Textbox(label="Transcript (Simulated via MockSTT)", interactive=False)
            answer_out = gr.Textbox(label="Answer", interactive=False, lines=4)
            guardrail_out = gr.HTML(label="Guardrail Status")
            
            with gr.Accordion("Pipeline Latency & Stages", open=True):
                latency_out = gr.Markdown("Waiting...")
                
            sources_out = gr.HTML(label="Retrieved Sources")
            
    async def process_query(audio_path, text_query):
        if audio_path is not None:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            res = await pipeline.run(audio_bytes)
        elif text_query:
            res = await pipeline.execute(PipelineInput(query=text_query))
        else:
            return "No input", "No input", "<div></div>", "0ms", "<div></div>"
            
        # Format sources
        sources_html = ""
        for s in res.sources:
            sources_html += f'''
            <div class="sources-card">
                <b>[{s.chunk_id}]</b> ({s.language})<br>
                <i>{s.text}</i><br>
                <small>Score: {s.score:.3f}</small>
            </div>
            '''
            
        # Format Latency
        lat = res.latency
        stt = lat.get('stt_ms', 0)
        emb = lat.get('embedding_ms', 0)
        ret = lat.get('retrieval_ms', 0)
        rer = lat.get('reranking_ms', 0)
        gen = lat.get('generation_ms', 0)
        tot = lat.get('total_ms', 0)
        
        lat_md = f"""
        | Stage | Latency | Note |
        |---|---|---|
        | **STT** | {stt:.1f} ms | *(Simulated)* |
        | **Embedding** | {emb:.1f} ms | |
        | **Retrieval** | {ret:.1f} ms | |
        | **Reranking** | {rer:.1f} ms | |
        | **Generation** | {gen:.1f} ms | *(Simulated)* |
        | **Total** | **{tot:.1f} ms** | |
        """
        
        # Guardrail Badge
        if "FAIL" in res.guardrail.name:
            g_html = f'<span class="status-badge error-badge">{res.guardrail.name}: {res.guardrail_reason}</span>'
        else:
            g_html = f'<span class="status-badge success-badge">GROUNDED ✓</span>'
            
        return res.transcript, res.answer, g_html, lat_md, sources_html

    submit_btn.click(process_query, inputs=[audio_input, text_input], outputs=[transcript_out, answer_out, guardrail_out, latency_out, sources_out])

if __name__ == "__main__":
    print("""
HH Goa Voice RAG
────────────────────────────
Mode: DEMO
STT: Mock
Embedding: Local
Vector DB: Qdrant Local
Reranker: Local
Generator: Mock
Guardrails: Enabled
────────────────────────────
Ready. Launching UI...
    """)
    demo.launch(server_name="0.0.0.0", server_port=7860)

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
            return "Could not transcribe audio.", "", 0.0, 0.0
            
        p_input = PipelineInput(query=transcript)
        response = pipeline.execute(p_input)
        
        sources = "\n".join([f"[{s.chunk_id}] {s.text}" for s in response.sources])
        total_lat = (response.latency.get("post_stt_ms", 0) + stt_latency) / 1000
        return response.answer, sources, total_lat, stt_latency / 1000
    except Exception as e:
        return f"Error: {str(e)}", "", 0.0, 0.0

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

with gr.Blocks(title="HH Goa 2026 Voice RAG") as demo:
    gr.Markdown("# 🎤 Hacker House Goa 2026: Voice-Enabled RAG Command Center")
    
    with gr.Tab("Voice"):
        audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Record Query")
        btn_voice = gr.Button("Submit Voice")
        
    with gr.Tab("Text"):
        text_in = gr.Textbox(label="Type Query")
        btn_text = gr.Button("Submit Text")
        
    with gr.Row():
        answer_out = gr.Textbox(label="Answer", lines=4)
        sources_out = gr.Textbox(label="Retrieved Sources", lines=4)
        
    with gr.Row():
        lat_out = gr.Number(label="Total Latency (s)")
        stt_out = gr.Number(label="STT Latency (s)")
        
    btn_voice.click(process_audio, inputs=[audio_in], outputs=[answer_out, sources_out, lat_out, stt_out])
    btn_text.click(process_text, inputs=[text_in], outputs=[answer_out, sources_out, lat_out, stt_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

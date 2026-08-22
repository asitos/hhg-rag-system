import argparse
from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

def export_model(model_id: str, output_dir: str):
    print(f"Exporting {model_id} to ONNX format at {output_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument("--output", type=str, default="models/embedder.onnx")
    args = parser.parse_args()
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    export_model(args.model, args.output)

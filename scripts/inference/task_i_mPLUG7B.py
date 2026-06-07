import os
import json
import torch
import gc
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# =========================
# Configurations
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

JSON_INPUT_PATH = os.path.join(DATA_DIR, "generation_checkpoint.json")

TOOL_IDENTIFICATION_PROMPT = "List all tools in this image. Please provide only the names of the tools, separated by commas. Do not include any explanations or extra text."

MODEL_NAME = "mPLUG/mPLUG-Owl3-7B-241101"

# Force standard float16 to prevent bfloat16 kernel crashes on this model
DTYPE = torch.float16 

# =========================
# Model Loading 
# =========================
def load_mplug_owl3(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading mPLUG-Owl3 model: {model_name}")
    print(f"Device: {device}")
    print(f"DType: {DTYPE}")

    # No SDPA optimization to prevent core dumps
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = model.init_processor(tokenizer)

    model.eval()
    print("✅ Model loaded successfully!")

    return model, processor, tokenizer

# =========================
# Inference (Stable 1-by-1 version)
# =========================
def vqa_image_mplug(model, processor, tokenizer, image_fp, prompt, max_new_tokens=128):
    device = next(model.parameters()).device

    image = Image.open(image_fp).convert("RGB")
    
    # Resize to prevent RoPE sequence length overflow (Modifies in-place)
    image.thumbnail((768, 768)) 
    
    if image.width < 30 or image.height < 30:
        return "ERROR_TOO_SMALL"

    # mPLUG strictly requires the empty assistant role to trigger generation
    messages = [
        {"role": "user", "content": f"<|image|>\n{prompt}"},
        {"role": "assistant", "content": ""}
    ]

    inputs = processor(messages, images=[image], videos=None)
    
    # Safely cast ALL floating point tensors to float16 and move to device
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v):
                inputs[k] = v.to(DTYPE).to(device)
            else:
                inputs[k] = v.to(device)

    # Bypass broken KV-Cache mechanism and set custom decode flags
    inputs.update({
        'tokenizer': tokenizer,
        'max_new_tokens': max_new_tokens,
        'decode_text': True, 
        'use_cache': False, 
    })

    with torch.no_grad():
        generated_text = model.generate(**inputs)

    if isinstance(generated_text, list):
        generated_text = generated_text[0]

    # Clean up whitespace
    cleaned = " ".join(generated_text.strip().split())
    
    return cleaned


# =========================
# Batch Processing 
# =========================
def process_batch(json_file, model_name):
    output_file = os.path.join(RESULTS_DIR, f"all_tools_identified_{model_name.replace('/', '_')}.json")
    
    model, processor, tokenizer = load_mplug_owl3(model_name)
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    completed_results = {}

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                completed_results = {item["id"]: item for item in checkpoint_data}
            print(f"Resuming {model_name}: Found {len(completed_results)} already processed items.")
        except (json.JSONDecodeError, KeyError):
            print("Warning: Existing output file is corrupted or formatted differently. Starting fresh.")
    
    print(f"Starting inference on {len(data)} items for {model_name}...")
    
    for index, item in enumerate(data):
        item_id = item.get("id", f"unknown_id_{index}")
        img_path = item.get("image_path", "")

        if item_id in completed_results:
            continue
            
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}. Skipping.")
            continue
            
        print(f"[{index + 1}/{len(data)}] Processing ID: {item_id}...")
        
        try:
            response_text = vqa_image_mplug(model, processor, tokenizer, img_path, TOOL_IDENTIFICATION_PROMPT)
            
            # --- THE FIX: Parse string into a list to match Qwen script ---
            identified_tools_list = [tool.strip() for tool in response_text.split(',') if tool.strip()]
            
            print(f"  Identified tools: {identified_tools_list}")
            
            completed_results[item_id] = {
                "id": item_id,
                "original_id": item.get("original_id"),
                "image_path": img_path,
                "model_used": model_name,
                "identified_tools": identified_tools_list 
            }

            # Save periodically as a list
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(list(completed_results.values()), f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"  [!] Error processing {item_id}: {e}")

    print(f"\nBatch processing complete! Results saved to {output_file}")
    
    # Memory cleanup
    del model
    del processor
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"\n{'='*50}\nStarting pipeline for model: {MODEL_NAME}\n{'='*50}")
    process_batch(JSON_INPUT_PATH, MODEL_NAME)
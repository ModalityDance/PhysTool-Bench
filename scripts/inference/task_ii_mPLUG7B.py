import os
import json
import gc
import torch
import contextlib
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

TOOL_IDENTIFICATION_PROMPT = "Given the following TASK, which tool(s) in the image are most appropriate to complete the task? Please list the name(s) of the selected tools in the order they should be used and separate them by commas. No explanation needed.\nTASK: {task_instruct}\nSELECTED TOOL(S) (in order of use): (Start your answer tool list with \'TOOL(S):\' and end it with \'END\')"
MODEL_NAME = "mPLUG/mPLUG-Owl3-7B-241101"

# 1. FIXED: Force standard float16. bfloat16 causes kernel crashes on this model.
DTYPE = torch.float16 

# =========================
# Model Loading 
# =========================
def load_mplug_owl3(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading mPLUG-Owl3 model: {model_name}")
    print(f"Device: {device}")
    print(f"DType: {DTYPE}")

    # Removed SDPA optimization to prevent core dumps
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 2. FIXED: Use the model's native init_processor instead of AutoProcessor
    processor = model.init_processor(tokenizer)

    model.eval()
    print("✅ Model loaded successfully!")

    return model, processor, tokenizer

# =========================
# Batched Inference
# =========================
def vqa_image_mplug_batch(
    model,
    processor,
    tokenizer,
    image_fps,
    prompts,
    max_new_tokens=128,
):
    device = next(model.parameters()).device
    results = []

    for fp, p in zip(image_fps, prompts):
        try:
            print(f"    -> Testing image: {fp}", flush=True) 
            
            image = Image.open(fp).convert("RGB")
            
            # 1. NEW: Resize to prevent RoPE sequence length overflow (Modifies in-place)
            # This keeps the aspect ratio but scales massive images down to a safe limit.
            image.thumbnail((768, 768)) 
            
            if image.width < 30 or image.height < 30:
                print(f"    [!] Skipping {fp} - Image too small ({image.size})", flush=True)
                results.append("ERROR_TOO_SMALL")
                continue

            messages = [
                {"role": "user", "content": f"<|image|>\n{p}"},
                {"role": "assistant", "content": ""}
            ]

            inputs = processor(messages, images=[image], videos=None)
            
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if torch.is_floating_point(v):
                        inputs[k] = v.to(DTYPE).to(device)
                    else:
                        inputs[k] = v.to(device)

            inputs.update({
                'tokenizer': tokenizer,
                'max_new_tokens': max_new_tokens,
                'decode_text': True, 
                'use_cache': False, # 2. NEW: Bypass the broken KV-Cache mechanism
            })

            with torch.no_grad():
                generated_text = model.generate(**inputs)

            if isinstance(generated_text, list):
                generated_text = generated_text[0]

            cleaned = " ".join(generated_text.strip().split())
            
            if "TOOL(S):" in cleaned:
                cleaned = cleaned.split("TOOL(S):")[-1].strip()

            print(f"    <- Output: {cleaned}", flush=True)
            results.append(cleaned)

        except Exception as e:
            print(f"  [!] Error processing image {fp}: {e}", flush=True)
            results.append("ERROR")

    return results

# =========================
# Batched Processing Loop 
# =========================
def process_batch(json_file, model_name, batch_size=2):
    output_file = os.path.join(RESULTS_DIR, f"task_ii_results_mPLUG.json")

    model, processor, tokenizer = load_mplug_owl3(model_name)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_items = len(data)
    results = []
    processed_ids = set()

    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed_ids = {item.get("id") for item in results if "id" in item}
        except:
            results = []
            processed_ids = set()

    pending_items_with_idx = [
        (idx, item) for idx, item in enumerate(data) 
        if item.get("id", "unknown") not in processed_ids
    ]
    
    print(f"Starting batched inference on {len(pending_items_with_idx)} items (Batch Size: {batch_size})...")

    for i in range(0, len(pending_items_with_idx), batch_size):
        batch = pending_items_with_idx[i : i + batch_size]
        
        valid_items = []
        image_fps = []
        prompts = []

        for idx, item in batch:
            img_path = item.get("image_path", "")
            if os.path.exists(img_path):
                valid_items.append((idx, item))
                image_fps.append(img_path)
                task_instruct = item.get("task_instruct", "")
                prompts.append(TOOL_IDENTIFICATION_PROMPT.format(task_instruct=task_instruct))

        if not valid_items:
            continue

        try:
            response_texts = vqa_image_mplug_batch(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image_fps=image_fps,
                prompts=prompts,
            )

            for (orig_idx, item), response_text in zip(valid_items, response_texts):
                item_id = item.get("id")
                slot = item.get("slot", 0)
                base_id = item.get("original_id", item_id.replace(f"_slot_{slot}", ""))

                print(f"\n[{orig_idx + 1}/{total_items}] ID: {base_id} | Slot: {slot}")
                print(f"  ✓ Identified tools: {response_text}")
                print("  💾 Progress saved.")

                result_item = {
                    "id": item_id,
                    "original_id": base_id,
                    "slot": slot,
                    "task_instruct": item.get("task_instruct", ""),
                    "image_path": item.get("image_path", ""),
                    "model_used": model_name,
                    "identified_tools": response_text,
                }
                
                results.append(result_item)
                processed_ids.add(item_id)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"\n[!] Error processing batch starting at index {i}: {e}")

    print(f"\nBatch processing complete for {model_name}! Results saved to {output_file}")

    del model
    del processor
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print(f"Starting pipeline for model: {MODEL_NAME}")
    print(f"{'=' * 60}")
    process_batch(JSON_INPUT_PATH, MODEL_NAME, batch_size=2)
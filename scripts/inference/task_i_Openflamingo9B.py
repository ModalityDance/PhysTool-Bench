import os
import json
import gc
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

JSON_INPUT_PATH = os.path.join(DATA_DIR, "generation_checkpoint.json")

# Generic tool listing prompt (same as mPLUG version)
TOOL_IDENTIFICATION_PROMPT = "List all tools in this image. Please provide only the names of the tools, separated by commas. Do not include any explanations or extra text."

# OpenFlamingo 9B checkpoint
MODEL_NAME = "openflamingo/OpenFlamingo-9B-vitl-mpt7b"

# Model-specific configuration (based on official OpenFlamingo examples)
OPENFLAMINGO_CONFIGS = {
    "openflamingo/OpenFlamingo-9B-vitl-mpt7b": {
        "clip_vision_encoder_path": "ViT-L-14",
        "clip_vision_encoder_pretrained": "openai",
        "lang_encoder_path": "anas-awadalla/mpt-7b",
        "tokenizer_path": "anas-awadalla/mpt-7b",
        "cross_attn_every_n_layers": 4,
    },
}

# MPT architecture overflows in float16 → use bfloat16 if supported, else float32
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32

def resolve_image_path(img_path: str) -> str:
    if img_path.startswith("./"):
        relative_part = img_path[2:]
        return os.path.join(DATA_DIR, relative_part)
    return img_path

def load_openflamingo(model_name: str):
    """Load OpenFlamingo model, processor, and tokenizer."""
    if model_name not in OPENFLAMINGO_CONFIGS:
        raise ValueError(
            f"Unsupported MODEL_NAME: {model_name}\n"
            f"Supported: {list(OPENFLAMINGO_CONFIGS.keys())}"
        )

    cfg = OPENFLAMINGO_CONFIGS[model_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading OpenFlamingo model: {model_name}")
    print(f"Device: {device}")
    print(f"DType: {DTYPE}")

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=cfg["clip_vision_encoder_path"],
        clip_vision_encoder_pretrained=cfg["clip_vision_encoder_pretrained"],
        lang_encoder_path=cfg["lang_encoder_path"],
        tokenizer_path=cfg["tokenizer_path"],
        cross_attn_every_n_layers=cfg["cross_attn_every_n_layers"],
    )

    print("Downloading/loading checkpoint...")
    checkpoint_path = hf_hub_download(model_name, "checkpoint.pt")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    if device == "cuda":
        model = model.to(device=device, dtype=DTYPE)
    else:
        model = model.to(device=device)

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, image_processor, tokenizer, device


# =========================
# Single‑Image Inference
# =========================

def vqa_image_openflamingo(model, image_processor, tokenizer, device, image_fp, prompt, max_new_tokens=64):
    image = Image.open(image_fp).convert("RGB")
    vision_x = image_processor(image)
    # OpenFlamingo 期望的维度是 (batch_size, num_media, num_frames, channels, height, width)
    vision_x = vision_x.unsqueeze(0).unsqueeze(1).unsqueeze(1).to(device=device, dtype=DTYPE)

    # 【关键修复】使用 OpenFlamingo 标准的 Zero-shot Prompt 格式
    # 必须包含 <image> 标签，并使用 Question/Answer 结构引导输出
    full_prompt = f"<image>Question: {prompt} Answer:"

    tokenizer.padding_side = "left"
    lang_x = tokenizer([full_prompt], return_tensors="pt", padding=True)
    input_ids = lang_x["input_ids"].to(device)
    attention_mask = lang_x["attention_mask"].to(device)

    # 获取结束标志，OpenFlamingo 通常在生成完一个 chunk 后输出 <|endofchunk|>
    endofchunk_token_id = tokenizer.encode("<|endofchunk|>")[-1]

    with torch.no_grad():
        generated_ids = model.generate(
            vision_x=vision_x,
            lang_x=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,            # 贪婪解码
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=endofchunk_token_id, # 【可选优化】遇到 endofchunk 时停止生成
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

    prompt_length = input_ids.shape[1]
    new_tokens = generated_ids[:, prompt_length:]
    response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    response = response.replace("<|endofchunk|>", "").strip()
    print(f"[DEBUG] Raw response: '{response}'")
    
    response = " ".join(response.strip().split())
    return response

# =========================
# Batch Processing (per item, with checkpoint resume)
# =========================

def process_batch(json_file, model_name):
    """Process all images one by one, saving results incrementally."""
    output_file = os.path.join(RESULTS_DIR, f"all_tools_identified_openflamingo.json")

    model, image_processor, tokenizer, device = load_openflamingo(model_name)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load existing results (resume support)
    completed_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
                completed_results = {item["id"]: item for item in checkpoint_data}
            print(f"Resuming {model_name}: Found {len(completed_results)} already processed items.")
        except (json.JSONDecodeError, KeyError):
            print("Warning: Existing output file is corrupted. Starting fresh.")

    print(f"Starting inference on {len(data)} items for {model_name}...")

    for index, item in enumerate(data):
        item_id = item.get("id", f"unknown_id_{index}")
        img_path = item.get("image_path", "")
        img_path = resolve_image_path(img_path)

        if item_id in completed_results:
            continue

        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}. Skipping.")
            continue

        print(f"[{index + 1}/{len(data)}] Processing ID: {item_id}...")

        try:
            response_text = vqa_image_openflamingo(
                model, image_processor, tokenizer, device,
                img_path, TOOL_IDENTIFICATION_PROMPT, max_new_tokens=64
            )

            # Parse comma‑separated list into a Python list (same as mPLUG version)
            identified_tools_list = [tool.strip() for tool in response_text.split(',') if tool.strip()]

            print(f"  Identified tools: {identified_tools_list}")

            completed_results[item_id] = {
                "id": item_id,
                "original_id": item.get("original_id"),
                "image_path": img_path,
                "model_used": model_name,
                "identified_tools": identified_tools_list
            }

            # Save after each item (safe and allows resuming)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(list(completed_results.values()), f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"  [!] Error processing {item_id}: {e}")

    print(f"\nBatch processing complete! Results saved to {output_file}")

    # Cleanup
    del model
    del image_processor
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================
# Main
# =========================

if __name__ == "__main__":
    print(f"\n{'='*50}\nStarting pipeline for model: {MODEL_NAME}\n{'='*50}")
    process_batch(JSON_INPUT_PATH, MODEL_NAME)
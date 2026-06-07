import os
import json
import gc
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms

# =========================
# Configurations
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

JSON_INPUT_PATH = os.path.join(DATA_DIR, "generation_checkpoint.json")

# Prompt that includes the task and asks for tool selection
TOOL_IDENTIFICATION_PROMPT = (
    "Given the following TASK, which tool(s) in the image are most appropriate to complete the task? "
    "Please list the name(s) of the selected tools in the order they should be used and separate them by commas. "
    "No explanation needed.\n"
    "TASK: {task_instruct}\n"
    "SELECTED TOOL(S) (in order of use):"
)

MODEL_NAME = "openflamingo/OpenFlamingo-9B-vitl-mpt7b"

OPENFLAMINGO_CONFIGS = {
    "openflamingo/OpenFlamingo-9B-vitl-mpt7b": {
        "clip_vision_encoder_path": "ViT-L-14",
        "clip_vision_encoder_pretrained": "openai",
        "lang_encoder_path": "anas-awadalla/mpt-7b",
        "tokenizer_path": "anas-awadalla/mpt-7b",
        "cross_attn_every_n_layers": 4,
    },
}

# Use same dtype logic as in task_i (bfloat16 if available)
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
    """Load OpenFlamingo model, image processor, and tokenizer."""
    if model_name not in OPENFLAMINGO_CONFIGS:
        raise ValueError(f"Unsupported MODEL_NAME: {model_name}\n"
                         f"Supported: {list(OPENFLAMINGO_CONFIGS.keys())}")

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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, image_processor, tokenizer, device

# =========================
# Single‑image inference (identical to task_i)
# =========================
def vqa_image_openflamingo(model, image_processor, tokenizer, device, image_fp, prompt, max_new_tokens=64):
    image = Image.open(image_fp).convert("RGB")
    vision_x = image_processor(image)
    vision_x = vision_x.unsqueeze(0).unsqueeze(1).unsqueeze(1).to(device=device, dtype=DTYPE)

    # Use the same zero‑shot format that worked in task_i
    full_prompt = f"<image>Question: {prompt} Answer:"

    tokenizer.padding_side = "left"
    lang_x = tokenizer([full_prompt], return_tensors="pt", padding=True)
    input_ids = lang_x["input_ids"].to(device)
    attention_mask = lang_x["attention_mask"].to(device)

    endofchunk_token_id = tokenizer.encode("<|endofchunk|>")[-1]

    with torch.no_grad():
        generated_ids = model.generate(
            vision_x=vision_x,
            lang_x=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=endofchunk_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

    prompt_length = input_ids.shape[1]
    new_tokens = generated_ids[:, prompt_length:]
    response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
    response = response.replace("<|endofchunk|>", "").strip()
    # Keep whitespace minimal but preserve the answer
    response = " ".join(response.split())
    return response

# =========================
# Sequential processing with checkpoint resume
# =========================
def process_batch(json_file, model_name):
    output_file = os.path.join(RESULTS_DIR, "task_ii_results_Openflamingo9B.json")

    model, image_processor, tokenizer, device = load_openflamingo(model_name)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load already processed results
    completed = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
                completed = {item["id"]: item for item in existing}
            print(f"Resuming: found {len(completed)} already processed items.")
        except (json.JSONDecodeError, KeyError):
            print("Warning: existing output file is corrupted. Starting fresh.")

    total = len(data)
    print(f"Starting inference on {total} items for {model_name}...")

    for idx, item in enumerate(data):
        item_id = item.get("id", f"unknown_{idx}")
        if item_id in completed:
            continue

        img_path = resolve_image_path(item.get("image_path", ""))
        if not os.path.exists(img_path):
            print(f"Warning: image not found – {img_path}. Skipping.")
            continue

        task_instruct = item.get("task_instruct", "")
        prompt = TOOL_IDENTIFICATION_PROMPT.format(task_instruct=task_instruct)

        print(f"\n[{idx+1}/{total}] ID: {item.get('original_id', item_id)} | Slot: {item.get('slot', 0)}")
        print(f"  Querying {model_name}...")

        try:
            response = vqa_image_openflamingo(
                model, image_processor, tokenizer, device,
                img_path, prompt, max_new_tokens=64
            )
            # The answer may already be a comma‑separated list;
            # we store it exactly as returned.
            print(f"  ✓ Identified tools: {response}")

            completed[item_id] = {
                "id": item_id,
                "original_id": item.get("original_id"),
                "slot": item.get("slot", 0),
                "task_instruct": task_instruct,
                "image_path": img_path,
                "model_used": model_name,
                "identified_tools": response,
            }

            # Save after each item
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(list(completed.values()), f, indent=4, ensure_ascii=False)
            print("  💾 Progress saved.")

        except Exception as e:
            print(f"  [!] Error processing {item_id}: {e}")

    print(f"\nBatch processing complete! Results saved to {output_file}")

    # Cleanup
    del model, image_processor, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Starting pipeline for model: {MODEL_NAME}")
    print(f"{'='*60}")
    process_batch(JSON_INPUT_PATH, MODEL_NAME)
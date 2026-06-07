import os
import json
import gc
import re
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms
import re

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
# Pick one released OpenFlamingo checkpoint.
# 3B is the easiest to start with.
MODEL_NAME = "openflamingo/OpenFlamingo-9B-vitl-mpt7b"

# Model-specific config for released checkpoints
# Based on the official initialization examples / model cards.
OPENFLAMINGO_CONFIGS = {
    "openflamingo/OpenFlamingo-3B-vitl-mpt1b": {
        "clip_vision_encoder_path": "ViT-L-14",
        "clip_vision_encoder_pretrained": "openai",
        "lang_encoder_path": "anas-awadalla/mpt-1b-redpajama-200b",
        "tokenizer_path": "anas-awadalla/mpt-1b-redpajama-200b",
        "cross_attn_every_n_layers": 1,
    },
    "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct": {
        "clip_vision_encoder_path": "ViT-L-14",
        "clip_vision_encoder_pretrained": "openai",
        "lang_encoder_path": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        "tokenizer_path": "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        "cross_attn_every_n_layers": 2,
    },
    "openflamingo/OpenFlamingo-9B-vitl-mpt7b": {
        "clip_vision_encoder_path": "ViT-L-14",
        "clip_vision_encoder_pretrained": "openai",
        "lang_encoder_path": "anas-awadalla/mpt-7b",
        "tokenizer_path": "anas-awadalla/mpt-7b",
        "cross_attn_every_n_layers": 4,
    },
}

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    DTYPE = torch.bfloat16
else:
    # MPT 架构在 float16 下必然溢出导致 FPE 和乱码。必须回退到 float32。
    DTYPE = torch.float32


# =========================
# Utilities
# =========================
def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _clean_assistant_output(text: str) -> str:
    """
    Tries to keep only the assistant answer after the prompt.
    """
    if "Assistant:" in text:
        text = text.split("Assistant:", 1)[-1]

    text = text.strip()

    # Remove trailing role markers if generation spills over
    text = re.split(r"\b(User:|<\|endofchunk\|>)\b", text)[0].strip()

    # Flatten newlines to make saved outputs cleaner
    text = " ".join(text.split())
    return text


# =========================
# Model Loading
# =========================
def load_openflamingo(model_name: str):
    if model_name not in OPENFLAMINGO_CONFIGS:
        raise ValueError(
            f"Unsupported MODEL_NAME: {model_name}\n"
            f"Supported: {list(OPENFLAMINGO_CONFIGS.keys())}"
        )

    cfg = OPENFLAMINGO_CONFIGS[model_name]
    device = _get_device()

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

    # map_location keeps CPU load safe before moving model
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.eval()

    if device == "cuda":
        model = model.to(device=device, dtype=DTYPE)
    else:
        model = model.to(device=device)

    # Make sure tokenizer has a pad token for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, image_processor, tokenizer

# =========================
# Batched Inference
# =========================
def vqa_image_openflamingo_batch(
    model,
    image_processor,
    tokenizer,
    image_fps,
    prompts,
    max_new_tokens=64,
    num_beams=3,
):
    device = next(model.parameters()).device

    # 1. Process all images into a single batch tensor
    vision_x_list = []
    for fp in image_fps:
        image = Image.open(fp).convert("RGB")
        processed_image = image_processor(image)
        # Add the num_media and num_frames dimensions: [1, 1, C, H, W]
        vision_x_list.append(processed_image.unsqueeze(0).unsqueeze(0))
    
    # Stack list and cast to match the model's datatype (bfloat16)
    vision_x = torch.stack(vision_x_list).to(device=device, dtype=next(model.parameters()).dtype)

    # ADD THIS SAFETY CHECK:
    if torch.isnan(vision_x).any():
        print("WARNING: NaN detected in vision tensor! An image in this batch is corrupted.")
    # 2. Format and tokenize all prompts EXACTLY as requested (Image + Prompt only)
    text_prompts = [f"<image>{p}" for p in prompts]

    # Causal LMs require left-padding for batched generation
    tokenizer.padding_side = "left" 

    lang_x = tokenizer(
        text_prompts,
        return_tensors="pt",
        padding=True,
    )

    input_ids = lang_x["input_ids"].to(device)
    attention_mask = lang_x["attention_mask"].to(device)
    
    # Grab the <|endofchunk|> token so the model knows when to stop
    eos_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]

    # 3. Generate answers for the whole batch
    with torch.no_grad():
        generated_ids = model.generate(
            vision_x=vision_x,
            lang_x=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    # 4. Decode ONLY the newly generated tokens (ignore the prompt)
    prompt_length = input_ids.shape[1]
    new_generated_ids = generated_ids[:, prompt_length:]
    
    decoded_texts = tokenizer.batch_decode(new_generated_ids, skip_special_tokens=True)
    
    # Clean up any trailing whitespace or random newlines
    cleaned_texts = [" ".join(text.strip().split()) for text in decoded_texts]
    
    return cleaned_texts

# =========================
# Batched Processing Loop
# =========================
def process_batch(json_file, model_name, batch_size=1):
    output_file = os.path.join(RESULTS_DIR, "task_ii_results_Openflamingo9B.json")

    model, image_processor, tokenizer = load_openflamingo(model_name)

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
        except json.JSONDecodeError:
            results = []
            processed_ids = set()

    # Pair items with their original index for the [X/2510] printout
    pending_items_with_idx = [
        (idx, item) for idx, item in enumerate(data) 
        if item.get("id", "unknown") not in processed_ids
    ]
    
    # Process in chunks
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
            # Run the batched inference
            response_texts = vqa_image_openflamingo_batch(
                model=model,
                image_processor=image_processor,
                tokenizer=tokenizer,
                image_fps=image_fps,
                prompts=prompts,
                max_new_tokens=64,
                num_beams=1,
            )

            # Format the output and save results
            for (orig_idx, item), response_text in zip(valid_items, response_texts):
                item_id = item.get("id")
                slot = item.get("slot", 0)
                
                # Try to get the base ID without the slot suffix for the display
                base_id = item.get("original_id", item_id.replace(f"_slot_{slot}", ""))
                final_tools = response_text.strip()

                # Print exactly matching the requested format
                print(f"\n[{orig_idx + 1}/{total_items}] ID: {base_id} | Slot: {slot}")
                print(f"  Querying {model_name}...")
                print(f"  ✓ Identified tools: {final_tools}")

                result_item = {
                    "id": item_id,
                    "original_id": base_id,
                    "slot": slot,
                    "task_instruct": item.get("task_instruct", ""),
                    "image_path": item.get("image_path", ""),
                    "model_used": model_name,
                    "identified_tools": final_tools,
                }
                
                results.append(result_item)
                processed_ids.add(item_id)

                # Print save confirmation per item to match screenshot
                print("  💾 Progress saved.")

            # Save to disk once per batch to avoid thrashing your drive
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"\n[!] Error processing batch starting at index {i}: {e}")

    print(f"\nBatch processing complete for {model_name}! Results saved to {output_file}")

    del model
    del image_processor
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print(f"Starting pipeline for model: {MODEL_NAME}")
    print(f"{'=' * 60}")
    process_batch(JSON_INPUT_PATH, MODEL_NAME)
import os
import json
import torch
from PIL import Image
import gc
import re
from transformers import AutoModel, AutoTokenizer

# --- Configurations ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

JSON_INPUT_PATH = os.path.join(DATA_DIR, "generation_checkpoint.json")

TOOL_IDENTIFICATION_PROMPT = "Given the following TASK, which tool(s) in the image are most appropriate to complete the task? Please list the name(s) of the selected tools in the order they should be used and separate them by commas. No explanation needed.\nTASK: {task_instruct}\nSELECTED TOOL(S) (in order of use): (Start your answer tool list with \'TOOL(S):\' and end it with \'END\')"
MODEL_NAME = 'openbmb/MiniCPM-V-4_5'  # or 'openbmb/MiniCPM-o-2_6'
ENABLE_THINKING = False  # Thinking mode
STREAM_OUTPUT = True      # Stream answer progressively
DTYPE = torch.float16

# --- Load MiniCPM model ---
def load_minicpm(model_name):
    print(f"Loading MiniCPM model {model_name}...")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation='sdpa',
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

# --- Inference function for a single image + question ---
def vqa_image_minicpm(model, tokenizer, image_fp, prompt, prev_msgs=None):
    """
    image_fp: path to image
    prompt: text question
    prev_msgs: list of previous messages for multi-turn context
    """
    image = Image.open(image_fp).convert('RGB')
    msgs = prev_msgs.copy() if prev_msgs else []
    msgs.append({'role': 'user', 'content': [image, prompt]})

    answer_gen = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        enable_thinking=ENABLE_THINKING,
        stream=STREAM_OUTPUT
    )

    generated_text = ""
    for new_text in answer_gen:
        generated_text += new_text
        print(new_text, flush=True, end='')

    # Append assistant reply to history
    return generated_text

# --- Batch Processing ---
def process_batch(json_file, model_name):
    output_file = os.path.join(RESULTS_DIR, f"task_ii_results_MiniCPM.json")

    model, tokenizer = load_minicpm(model_name)

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    processed_ids = set()

    # Resume from existing output
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            processed_ids = {item.get("id") for item in results if "id" in item}
            print(f"Resuming {model_name}: Found {len(processed_ids)} already processed items.")
        except json.JSONDecodeError:
            print("Warning: Existing output file is corrupted. Starting fresh.")

    print(f"Starting inference on {len(data)} items for {model_name}...")

    for index, item in enumerate(data):
        item_id = item.get("id", f"unknown_id_{index}")
        task_instruct = item.get("task_instruct", "")
        img_path = item.get("image_path", "")

        if item_id in processed_ids:
            continue
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}. Skipping.")
            continue

        prompt = TOOL_IDENTIFICATION_PROMPT.format(task_instruct=task_instruct)
        print(f"\n[{index + 1}/{len(data)}] Processing ID: {item_id}...")

        try:
            generated_text = vqa_image_minicpm(model, tokenizer, img_path, prompt)

            match = re.search(r'TOOL\(S\):(.*?)(?:END|$)', generated_text, re.DOTALL | re.IGNORECASE)
            final_tools = match.group(1).strip() if match else generated_text

            result_item = {
                "id": item_id,
                "original_id": item.get("original_id", ""),
                "slot": item.get("slot", 0),
                "task_instruct": task_instruct,
                "image_path": img_path,
                "model_used": model_name,
                "identified_tools": final_tools
            }

            results.append(result_item)
            processed_ids.add(item_id)

            # Save periodically
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"Error processing {item_id}: {e}")

    print(f"\nBatch processing complete for {model_name}! Results saved to {output_file}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# --- Main ---
if __name__ == "__main__":
    print(f"\n{'='*50}\nStarting MiniCPM pipeline\n{'='*50}")
    process_batch(JSON_INPUT_PATH, MODEL_NAME)
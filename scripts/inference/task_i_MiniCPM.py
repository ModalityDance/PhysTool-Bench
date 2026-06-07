import os
import json
import torch
import gc
import re
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
# We use the specific framing to guide MiniCPM's output for easier parsing
TOOL_IDENTIFICATION_PROMPT = "List all tools in this image. Please provide only the names of the tools, separated by commas. Start your answer tool list with 'TOOL(S):' and end it with 'END'. Do not include any explanations."

MODEL_NAME = 'openbmb/MiniCPM-V-4_5'
DTYPE = torch.bfloat16 # MiniCPM performs best with bfloat16

# =========================
# Model Loading
# =========================
def load_minicpm(model_name):
    print(f"\nLoading MiniCPM model: {model_name}")
    # MiniCPM-V 4.5 prefers SDPA and bfloat16 for efficiency
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=DTYPE
    ).eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✅ Model loaded successfully!")
    return model, tokenizer

# =========================
# Inference & Parsing
# =========================
def parse_tool_list(raw_text):
    """
    Extracts the content between 'TOOL(S):' and 'END', 
    then converts the comma-separated string into a clean list.
    """
    # Use regex to find content between markers (case insensitive)
    pattern = re.compile(r"TOOL\(S\):\s*(.*?)\s*END", re.IGNORECASE | re.DOTALL)
    match = pattern.search(raw_text)
    
    content = match.group(1) if match else raw_text
    
    # Split by commas, strip whitespace, and filter out empty strings
    tools = [t.strip() for t in content.split(',') if t.strip()]
    return tools

def vqa_image_minicpm(model, tokenizer, image_fp, prompt):
    image = Image.open(image_fp).convert('RGB')
    
    # Message format required by MiniCPM-V
    msgs = [{'role': 'user', 'content': [image, prompt]}]

    # We use non-streaming here for simpler batch processing logic
    answer = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=False, # Use greedy decoding for more consistent lists
        max_new_tokens=256
    )
    
    return answer

# =========================
# Batch Processing
# =========================
def process_batch(json_file, model_name):
    # Output name matches the mPLUG naming convention
    output_file = os.path.join(RESULTS_DIR, f"all_tools_identified_{model_name.replace('/', '_')}.json")
    
    model, tokenizer = load_minicpm(model_name)
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    completed_results = {}

    # Resume capability
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                completed_results = {item["id"]: item for item in checkpoint_data}
            print(f"Resuming: Found {len(completed_results)} already processed items.")
        except (json.JSONDecodeError, KeyError):
            print("Warning: Existing output file corrupted. Starting fresh.")
    
    print(f"Starting inference on {len(data)} items...")
    
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
            raw_response = vqa_image_minicpm(model, tokenizer, img_path, TOOL_IDENTIFICATION_PROMPT)
            
            # Extract list using the "TOOL(S): ... END" logic
            identified_tools_list = parse_tool_list(raw_response)
            
            print(f"  Identified tools: {identified_tools_list}")
            
            # Format matches mPLUG script exactly
            completed_results[item_id] = {
                "id": item_id,
                "original_id": item.get("original_id"),
                "image_path": img_path,
                "model_used": model_name,
                "identified_tools": identified_tools_list 
            }

            # Save periodically
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(list(completed_results.values()), f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"  [!] Error processing {item_id}: {e}")

    print(f"\nBatch processing complete! Results saved to {output_file}")
    
    # Memory cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print(f"\n{'='*50}\nStarting MiniCPM Find-All Pipeline\n{'='*50}")
    process_batch(JSON_INPUT_PATH, MODEL_NAME)
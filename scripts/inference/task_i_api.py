import json
import requests
import os
import time
import argparse
import base64
from typing import Optional, List

API_TOKEN = None
CHAT_API_URL = "https://api.gpt.ge/v1/chat/completions"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_OUTPUT_DIR = RESULTS_DIR
DEFAULT_INPUT_FILE = os.path.join(DATA_DIR, "generation_checkpoint.json")

# Updated prompt to ask for all tools, formatted as a comma-separated list
TOOL_IDENTIFICATION_PROMPT = "List all tools in this image. Please provide only the names of the tools, separated by commas. Do not include any explanations or extra text."

def get_image_mime_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    return {'png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.webp': 'image/webp'}.get(ext, 'image/png')

def identify_all_tools(image_path: str, model: str, max_retries: int = 3, retry_delay: int = 5) -> Optional[List[str]]:
    """Identifies all tools in an image and returns them as a list of strings."""
    try:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
        mime_type = get_image_mime_type(image_path)
    except Exception as e:
        print(f"  Error reading image: {e}")
        return None
    
    headers = {'Authorization': f'Bearer {API_TOKEN}', 'Content-Type': 'application/json'}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TOOL_IDENTIFICATION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 500 # Slightly increased in case there are many tools
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(CHAT_API_URL, headers=headers, json=payload, timeout=240)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "").strip()
                # Parse the comma-separated string into a clean list of individual tools
                tools_list = [tool.strip() for tool in content.split(',') if tool.strip()]
                return tools_list
            return None
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1: time.sleep(retry_delay)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429: time.sleep(retry_delay * 2)
            elif e.response.status_code >= 500 and attempt < max_retries - 1: time.sleep(retry_delay)
            else: 
                print(e.response)
                return None
        except Exception: 
            return None
    return None

def load_checkpoint(checkpoint_path: str) -> dict:
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return {item["id"]: item for item in json.load(f)}
        except json.JSONDecodeError: pass
    return {}

def save_checkpoint(results_dict: dict, checkpoint_path: str):
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(list(results_dict.values()), f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Identify all tools in images using an LLM.")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT_FILE, help=f"Input JSON map (default: {DEFAULT_INPUT_FILE})")                          
    parser.add_argument("--api-token", "-t", required=True, help="API token for authentication")
    parser.add_argument("--model", "-m", default="gpt-4o", help="Model to use for API request (default: gpt-4o)")
    parser.add_argument("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR, help=f"Dir for results (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--delay", type=int, default=2, help="Delay between API requests")
    
    args = parser.parse_args()
    
    # Set global API_TOKEN from command line argument
    global API_TOKEN
    API_TOKEN = args.api_token
    
    if API_TOKEN == "sk-" or API_TOKEN == "YOUR_API_KEY_HERE":
        print("ERROR: Please update API_TOKEN.")
        return 1

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            generated_data = json.load(f)
    except Exception as e:
        print(f"Error loading {args.input}: {e}\nDid you run the image generation script first?")
        return 1
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    checkpoint_file = f"all_tools_identified_{args.model.replace('/', '_')}.json"
    checkpoint_path = os.path.join(args.output_dir, checkpoint_file)
    completed_results = load_checkpoint(checkpoint_path)

    for i, item_data in enumerate(generated_data, 1):
        comp_id = item_data["id"]
        
        if comp_id in completed_results:
            #print(f"  Already processed by {args.model}. Skipping.")
            continue

        print(f"\n[{i}/{len(generated_data)}] ID: {item_data.get('original_id', comp_id)}")

        image_path = item_data["image_path"]
        if not os.path.exists(image_path):
            print(f"  Image missing: {image_path}. Skipping.")
            continue

        print(f"  Querying {args.model}...")
        
        # No longer passing task_instruct
        identified_tools_list = identify_all_tools(image_path, args.model)
        
        if identified_tools_list:
            print(f"  Identified tools: {identified_tools_list}")
            completed_results[comp_id] = {
                "id": comp_id,
                "original_id": item_data.get("original_id"),
                "image_path": image_path,
                "model_used": args.model,
                "identified_tools": identified_tools_list # Now saves as a JSON array
            }
            save_checkpoint(completed_results, checkpoint_path)
            print("  Progress saved.")
        else:
            print("  Failed to identify tools.")
            
        if i < len(generated_data): time.sleep(args.delay)

    return 0

if __name__ == "__main__": exit(main())
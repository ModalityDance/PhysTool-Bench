import json
import argparse
import os
import difflib
from typing import List

# =========================
# Path resolution (relative to this script)
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def normalize_tool_name(name: str) -> str:
    """Helper to standardise tool names for string comparison."""
    return name.strip().lower()

def tools_match(tool1: str, tool2: str, negative_tools: List[str] = None) -> bool:
    """
    Check if two tool names match (case-insensitive, handles variations).
    Utilizes set intersection logic to determine base-word matches and filters
    out ambiguities against negative tools.
    """
    if negative_tools is None:
        negative_tools = []
    
    t1 = normalize_tool_name(tool1)
    t2 = normalize_tool_name(tool2)
    
    if t1 == t2:
        return True
    
    # 1. Exact match and Compound/Plural bypass
    # Strip spaces and trailing 's' to catch variations like "Jack stand" vs "Jackstands"
    clean_t1 = t1.replace(" ", "").rstrip("s")
    clean_t2 = t2.replace(" ", "").rstrip("s")
    if clean_t1 == clean_t2 and len(clean_t1) > 0:
        return True

    # 2. Set Intersection Logic for partial/out-of-order matches
    set1 = set(t1.split())
    set2 = set(t2.split())
    
    # Calculate intersection of words
    intersection = set1 & set2
    
    # If there is a valid intersection and one tool is a subset of the other
    if intersection and (intersection == set1 or intersection == set2):
        
        is_ambiguous = False
        
        # Check if this intersection is also found entirely in a negative tool
        for neg_tool in negative_tools:
            neg_set = set(normalize_tool_name(neg_tool).split())
            if intersection.issubset(neg_set):
                is_ambiguous = True
                break
                
        # If it's unique to the target (not found in negatives), it's a match!
        if not is_ambiguous:
            return True
    
    return False

def get_best_match_fuzzy(target, candidates, threshold=0.90):
    """Finds the best fuzzy match for a target string from a list of candidates."""
    best_match = None
    best_score = 0.0
    
    for candidate in candidates:
        score = difflib.SequenceMatcher(None, target, candidate).ratio()
        if target in candidate or candidate in target:
            score = max(score, 0.85)

        if score > best_score:
            best_score = score
            best_match = candidate
            
    if best_score >= threshold:
        return best_match, best_score
    return None, 0.0

def calculate_metrics(ground_truth_tools, predicted_tools, match_method="fuzzy", threshold=0.75):
    """Calculates precision and recall using the selected matching method."""
    gt_pool = [tool.strip().lower() for tool in ground_truth_tools]
    pred_pool = [tool.strip().lower() for tool in predicted_tools]
    
    true_positives = 0
    matched_pairs = []
    unmatched_preds = []
    
    for pred in pred_pool:
        best_match = None
        score = 0.0
        
        if match_method == "fuzzy":
            best_match, score = get_best_match_fuzzy(pred, gt_pool, threshold)
            
        elif match_method == "string":
            # Test the prediction against every remaining ground truth tool
            for gt_candidate in gt_pool:
                # The "negative tools" are the OTHER ground truth tools in the image.
                # This prevents "wrench" from matching "pipe wrench" if "socket wrench" is also in the image.
                other_gts = [t for t in gt_pool if t != gt_candidate]
                
                if tools_match(pred, gt_candidate, negative_tools=other_gts):
                    best_match = gt_candidate
                    score = 1.0 # Boolean match
                    break
        
        if best_match:
            true_positives += 1
            matched_pairs.append({
                "predicted": pred, 
                "ground_truth_match": best_match, 
                "similarity_score": round(score, 3) if match_method == "fuzzy" else "Exact/String Match"
            })
            # Remove from pool to prevent double-matching
            gt_pool.remove(best_match) 
        else:
            unmatched_preds.append(pred)
            
    false_positives = len(unmatched_preds)
    false_negatives = len(gt_pool) # Remaining unmatched ground truth tools
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    return precision, recall, matched_pairs, unmatched_preds, gt_pool

def main():
    parser = argparse.ArgumentParser(description="Evaluate tool identification precision and recall.")
    parser.add_argument("--model", "-m", required=True, help="Model name used for predictions (e.g., gpt-4o, MiniCPM)")
    parser.add_argument("--ground-truth", "-g", default=None, help="Path to ground truth JSON (default: data/corrected_tools.json)")
    parser.add_argument("--predictions", "-p", default=None, help="Path to predictions JSON (default: results/all_tools_identified_{model}.json)")
    parser.add_argument("--output-json", "-o", default=None, help="Path to save output JSON (default: results/eval_tool_finding_{model}.json)")
    parser.add_argument("--match-method", choices=["fuzzy", "string"], default="fuzzy", help="Method to match tools (fuzzy or string)")
    parser.add_argument("--threshold", "-t", type=float, default=0.75, help="Fuzzy match threshold (0.0 to 1.0), ignored for 'string' method")
    
    args = parser.parse_args()

    # Build default paths using the provided model name
    if args.ground_truth is None:
        args.ground_truth = os.path.join(DATA_DIR, "corrected_tools.json")
    if args.predictions is None:
        args.predictions = os.path.join(RESULTS_DIR, f"all_tools_identified_{args.model}.json")
    if args.output_json is None:
        args.output_json = os.path.join(RESULTS_DIR, f"eval_tool_finding_{args.model}.json")

    # 1. Load the data
    try:
        with open(args.ground_truth, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        with open(args.predictions, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return 1

    # 2. Build the flexible mapping of Ground Truth tools
    gt_map = {}
    for item in gt_data:
        gt_id = str(item.get("id"))
        gt_slot = item.get("slot")
        # Use shuffled_available_tools (all tools present in the image)
        tools = item.get("shuffled_available_tools", [])
        
        gt_map[gt_id] = tools
        if gt_slot is not None:
            gt_map[f"{gt_id}_{gt_slot}"] = tools

    total_precision = 0.0
    total_recall = 0.0
    evaluated_count = 0
    
    final_results = {
        "summary": {},
        "per_image_results": []
    }

    print(f"Using match method: {args.match_method.upper()}")
    print(f"{'ID':<25} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 50)

    # 3. Compare predictions to ground truth
    for item in pred_data:
        comp_id = str(item.get("id", ""))
        orig_id = str(item.get("original_id", ""))
        
        if comp_id in gt_map:
            ground_truth_tools = gt_map[comp_id]
        elif orig_id in gt_map:
            ground_truth_tools = gt_map[orig_id]
        else:
            continue

        predicted_tools = item.get("identified_tools", [])
        
        precision, recall, matched, unmatched_preds, missed_gts = calculate_metrics(
            ground_truth_tools, 
            predicted_tools, 
            match_method=args.match_method,
            threshold=args.threshold
        )
        
        # Print per-picture stats
        print(f"{comp_id:<25} | {precision:.2f}       | {recall:.2f}")
        
        # Append detailed breakdown to JSON structure
        final_results["per_image_results"].append({
            "id": comp_id,
            "original_id": orig_id,
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4)
            },
            "matching_details": {
                "matched_pairs": matched,
                "unmatched_predictions": unmatched_preds,
                "missed_ground_truth": missed_gts
            }
        })
        
        total_precision += precision
        total_recall += recall
        evaluated_count += 1

    # 4. Calculate averages and save JSON
    print("-" * 50)
    if evaluated_count > 0:
        avg_precision = total_precision / evaluated_count
        avg_recall = total_recall / evaluated_count
        
        print(f"{'AVERAGE':<25} | {avg_precision:.2f}       | {avg_recall:.2f}")
        print(f"\nTotal images evaluated: {evaluated_count}")
        
        final_results["summary"] = {
            "match_method_used": args.match_method,
            "average_precision": round(avg_precision, 4),
            "average_recall": round(avg_recall, 4),
            "total_images_evaluated": evaluated_count
        }
        if args.match_method == "fuzzy":
            final_results["summary"]["fuzzy_threshold"] = args.threshold
        
        # Write out to JSON file
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
            
        print(f"Detailed JSON results successfully saved to: {args.output_json}")
    else:
        print("No matching records found to evaluate.")

    return 0

if __name__ == "__main__":
    exit(main())
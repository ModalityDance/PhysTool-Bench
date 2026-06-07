import json
import argparse
import os
import sys
import requests
from typing import List, Dict, Tuple, Optional

# --- API Configuration ---
API_URL = "https://api.gpt.ge/v1/chat/completions"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

API_KEY = None
MODEL_NAME = "gemini-3.1-pro-preview"

def enforce_one_to_one(mapping: Dict[str, str], log_prefix: str = "") -> Dict[str, str]:
    """
    Ensure the mapping is one‑to‑one: each identified tool maps to a unique target,
    and each target is used by at most one identified tool.
    If conflicts are found, keep the first occurrence and drop the rest.
    """
    seen_targets = set()
    clean = {}
    conflicts = []
    for ident, targ in mapping.items():
        if targ in seen_targets:
            conflicts.append(f"  '{ident}' -> '{targ}' (target already assigned to another tool)")
        else:
            seen_targets.add(targ)
            clean[ident] = targ
    if conflicts:
        print(f"{log_prefix}One-to-one enforcement removed {len(conflicts)} conflicting mapping(s):")
        for c in conflicts:
            print(c)
    return clean

def get_gemini_matches(task_instruct: str, identified: List[str], target: List[str], negative: List[str]) -> Optional[Dict[str, str]]:
    """
    Uses Gemini to match identified tools to target tools while avoiding negative tools.
    Returns a one-to-one mapping, or None if an error occurs (caller should skip the item).
    """
    if not identified or not target:
        return {} 

    prompt = f"""
    You are an expert evaluator. I have a list of 'Identified Tools' predicted by a model. 
    Your task is to map each 'Identified Tool' to the correct 'Target Tool' name (if applicable) for the provided task, while ensuring it does not refer to any 'Negative Tools' (distractors).

    Rules:
    1. Only match if the Identified Tool is clearly the same tool as a Target Tool.
    2. If the Identified Tool is ambiguous and could potentially refer to a Negative Tool, DO NOT match it.
    3. Use the exact string from the Target Tools list for the value in your mapping.
    4. Return ONLY a valid JSON object where keys are the Identified Tool strings and values are the corresponding Target Tool strings.
    5. DO NOT map multiple Identified Tools to the same Target Tool – each target can appear at most once.

    Target Tools: {json.dumps(target)}
    Negative Tools: {json.dumps(negative)}
    Identified Tools: {json.dumps(identified)}
    Task Instruction: {json.dumps(task_instruct)}

    Result (JSON Map):
    """

    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.0,
        "response_format": { "type": "json_object" }
    })
    
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(API_URL, headers=headers, data=payload, timeout=120)
        response.raise_for_status()
        res_json = response.json()
        content = res_json['choices'][0]['message']['content']
        raw_mapping = json.loads(content)

        # 统一转为小写，防止大小写差异导致漏匹配
        mapping = {str(k).lower(): str(v).lower() for k, v in raw_mapping.items()}

        # 过滤掉不存在于目标和预测列表中的幻觉键值
        valid_mapping = {k: v for k, v in mapping.items() if k in identified and v in target}

        # 强制执行一对一匹配规则
        one_to_one = enforce_one_to_one(valid_mapping, log_prefix=f"[{task_instruct[:50]}...] ")
        return one_to_one
    except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
        print(f"Error in Gemini matching: {e}")
        return None

def parse_identified_tools(identified_str: str) -> List[str]:
    if not identified_str:
        return []
    identified_str = identified_str.rstrip('.')
    return [t.strip() for t in identified_str.split(',') if t.strip()]

def calculate_metrics_with_gemini(identified: List[str], target: List[str], target_steps: List[int], matched_details: Dict[str, str], ordered: bool, k_values: List[int]):
    """Calculates metrics using the Gemini-provided matches based on unique identified tools and step constraints."""
    
    seen = set()
    unique_identified = []
    for tool in identified:
        if tool not in seen:
            unique_identified.append(tool)
            seen.add(tool)
            
    matched_targets = set()
    identified_matched_info = []

    for identified_tool in unique_identified:
        matched_target = matched_details.get(identified_tool)
        if matched_target and matched_target in target:
            matched_targets.add(matched_target)
            target_idx = target.index(matched_target)
            identified_matched_info.append({
                "target_tool": matched_target,
                "step": target_steps[target_idx]
            })

    correct_count = len(matched_targets)
    num_unique_identified = len(unique_identified)
    num_target = len(target)

    precision_value = correct_count / num_unique_identified if num_unique_identified > 0 else 0.0
    recall_value = correct_count / num_target if num_target > 0 else 0.0

    success_rate = False
    # SR: 必须找出数量对齐且完全正确的工具
    if num_unique_identified == num_target and correct_count == num_target:
        if ordered:
            # 步骤必须是非严格递增（同一步骤工具的顺序无所谓）
            matched_steps = [info["step"] for info in identified_matched_info]
            success_rate = all(matched_steps[i] <= matched_steps[i+1] for i in range(len(matched_steps)-1))
        else:
            success_rate = True

    sr_at_k = {}
    if ordered:
        for k in k_values:
            if num_target >= k:
                if num_unique_identified >= k:
                    prefix_tools = unique_identified[:k]
                    prefix_targets = [matched_details.get(t) for t in prefix_tools]
                    
                    if all(t in target for t in prefix_targets):
                        prefix_steps = [target_steps[target.index(t)] for t in prefix_targets]
                        
                        # 条件1: 保证前 k 个工具在 step 上是非严格递增的
                        is_non_decreasing = all(prefix_steps[i] <= prefix_steps[i+1] for i in range(k-1))
                        
                        # 条件2: 无跳步漏洞。若选了较大 step 的工具，必须确保所有更小 step 的工具都已经包含在内
                        max_step_in_prefix = max(prefix_steps) if prefix_steps else 0
                        required_prereqs = [t for i, t in enumerate(target) if target_steps[i] < max_step_in_prefix]
                        has_all_prereqs = all(req in prefix_targets for req in required_prereqs)
                        
                        sr_at_k[f"k={k}"] = (is_non_decreasing and has_all_prereqs)
                    else:
                        sr_at_k[f"k={k}"] = False
                else:
                    sr_at_k[f"k={k}"] = False
            else:
                sr_at_k[f"k={k}"] = None

    poa = None
    # 计算 target 中存在跨 step 关系的组合数（同 step 视为无约束，不纳入 POA 计算）
    total_pairs_1 = sum(1 for i in range(num_target) for j in range(i+1, num_target) if target_steps[i] < target_steps[j])
    
    if ordered and total_pairs_1 > 0:
        correct_pairs = 0
        total_pairs_2 = 0
        n = len(identified_matched_info)
        
        for i in range(n):
            for j in range(i + 1, n):
                step_i = identified_matched_info[i]["step"]
                step_j = identified_matched_info[j]["step"]
                
                # 仅对存在跨 step 的被识别工具对进行正误判断
                if step_i != step_j:
                    total_pairs_2 += 1
                    if step_i < step_j:
                        correct_pairs += 1
        
        poa = {
            "correct_pairs": correct_pairs,
            "total_pairs_1": total_pairs_1,
            "total_pairs_2": total_pairs_2,
            "value_1": correct_pairs / total_pairs_1 if total_pairs_1 > 0 else 0.0,
            "value_2": correct_pairs / total_pairs_2 if total_pairs_2 > 0 else 0.0
        }

    return {
        "success_rate": success_rate,
        "success_rate_at_k": sr_at_k if ordered else None,
        "precision": {"correct_tools": correct_count, "total_predicted": num_unique_identified, "value": precision_value},
        "recall": {"correct_tools": correct_count, "total_target": num_target, "value": recall_value},
        "pairwise_order_accuracy": poa,
        "num_unique_identified": num_unique_identified
    }

def save_report(evaluations, stats, output_path, k_list):
    """Build the final report and write it to output_path."""
    report = {
        "summary": {
            "total_evaluated": stats["valid"],
            "ordered_tasks": stats["ordered"],
            "unordered_tasks": stats["valid"] - stats["ordered"],
            "success_rate": {
                "count": stats["success"],
                "percentage": stats["success"] / stats["valid"] if stats["valid"] else 0
            },
            "success_rate_at_k": {
                k: {
                    "success_count": c,
                    "valid_tasks": stats["sr_k_valid"].get(k, 0),
                    "percentage": c / stats["sr_k_valid"][k] if stats["sr_k_valid"].get(k, 0) > 0 else 0,
                    "note": f"Only calculated for ordered tasks where target tools >= {k.split('=')[1]}"
                }
                for k, c in stats["sr_k"].items()
            } if stats["ordered"] else None,
            "average_precision": stats["prec"] / stats["valid"] if stats["valid"] else 0,
            "average_recall": stats["rec"] / stats["valid"] if stats["valid"] else 0,
            "overall_average_pairwise_order_accuracy": stats["poa1"] / stats["poa_cnt"] if stats["poa_cnt"] else None,
            "average_correct_pairwise_order_accuracy": stats["poa2"] / stats["poa_cnt"] if stats["poa_cnt"] else None,
            "pairwise_order_accuracy_note": (
                "Only calculated for ordered tasks with cross-step dependencies in target tools" if stats["poa_cnt"]
                else "No valid tasks for POA calculation"
            )
        },
        "evaluations": evaluations
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Gemini-based report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Gemini-based tool evaluation with new target_steps logic')
    parser.add_argument('--api-token', '-t', required=True, help='API token for authentication')
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--results', '-r', type=str, default=None)
    parser.add_argument('--input', '-i', type=str, default=os.path.join(DATA_DIR, "corrected_tools.json"))
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--k-values', '-k', type=str, default="1,2,3")
    args = parser.parse_args()

    global API_KEY
    API_KEY = args.api_token

    if args.results is None:
        results_path = os.path.join(RESULTS_DIR, f"task_ii_results_{args.model}.json")
    else:
        results_path = args.results

    if args.output is None:
        output_path = os.path.join(RESULTS_DIR, f"evaluation_of_{args.model}_with_gemini.json")
    else:
        output_path = args.output

    k_list = [int(k.strip()) for k in args.k_values.split(',')]

    existing_evaluations = []
    already_evaluated_keys = set()
    
    stats = {
        "success": 0, "prec": 0.0, "rec": 0.0, "poa1": 0.0, "poa2": 0.0,
        "sr_k": {f"k={k}": 0 for k in k_list}, 
        "sr_k_valid": {f"k={k}": 0 for k in k_list}, 
        "valid": 0, "ordered": 0, "poa_cnt": 0
    }

    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                prev_report = json.load(f)
            existing_evaluations = prev_report.get("evaluations", [])
            print(f"Found existing output with {len(existing_evaluations)} evaluations. Resuming...")

            for eval_item in existing_evaluations:
                key = (eval_item.get("original_id"), eval_item.get("slot", 0))
                already_evaluated_keys.add(key)

                stats["valid"] += 1
                if eval_item.get("success_rate"):
                    stats["success"] += 1
                stats["prec"] += eval_item["precision"]["value"]
                stats["rec"] += eval_item["recall"]["value"]
                if eval_item.get("ordered"):
                    stats["ordered"] += 1
                    sr_at_k = eval_item.get("success_rate_at_k")
                    if sr_at_k:
                        for k_key, k_val in sr_at_k.items():
                            if k_val is not None:
                                stats["sr_k_valid"][k_key] += 1
                                if k_val is True:
                                    stats["sr_k"][k_key] += 1
                                    
                    poa = eval_item.get("pairwise_order_accuracy")
                    if poa:
                        stats["poa1"] += poa["value_1"]
                        stats["poa2"] += poa["value_2"]
                        stats["poa_cnt"] += 1
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: could not parse existing output file ({e}). Starting fresh.")
            existing_evaluations = []
            already_evaluated_keys = set()
            stats = {
                "success": 0, "prec": 0.0, "rec": 0.0, "poa1": 0.0, "poa2": 0.0,
                "sr_k": {f"k={k}": 0 for k in k_list}, 
                "sr_k_valid": {f"k={k}": 0 for k in k_list},
                "valid": 0, "ordered": 0, "poa_cnt": 0
            }

    with open(results_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    with open(args.input, 'r', encoding='utf-8') as f:
        ground_truth = {(item['id'], item.get('slot', 0)): item for item in json.load(f)}

    evaluations = existing_evaluations.copy()

    try:
        for idx, pred in enumerate(predictions):
            original_id = pred.get("original_id")
            slot = pred.get("slot", 0)
            key = (original_id, slot)
            
            if key in already_evaluated_keys:
                print(f"[{idx+1}/{len(predictions)}] Skipping ID: {key[0]} (already evaluated)")
                continue
                
            if key not in ground_truth:
                print(f"⚠️ [{idx+1}/{len(predictions)}] Warning: {original_id}_slot_{slot} 找不到对应的 Ground Truth，跳过。")
                continue
                
            gt_item = ground_truth[key]
            identified = [t.lower() for t in parse_identified_tools(pred.get("identified_tools", ""))]
            
            # --- 解析新的 ground_truth 格式 (corrected_tools.json) ---
            refined_tax = gt_item.get("refined_taxonomy", {})
            target = [t.lower() for t in refined_tax.get("target_tools", [])]
            target_steps = refined_tax.get("target_steps", [])
            negative_tools = [t.lower() for t in refined_tax.get("negative_tools", [])]
            ordered = gt_item.get("ordered", True)
            task_instruct = gt_item.get("task_instruct", "")
            
            print(f"[{idx+1}/{len(predictions)}] Evaluating ID: {original_id} via API...")

            # 仅在有必要时调用 API
            matched_details = get_gemini_matches(task_instruct, identified, target, negative_tools)
            if matched_details is None:
                print(f"[{idx+1}/{len(predictions)}] ERROR: Skipping ID {original_id} due to Gemini API failure.")
                continue
            
            # 使用新版本的计算指标逻辑
            metrics = calculate_metrics_with_gemini(identified, target, target_steps, matched_details, ordered, k_list)
            
            eval_item = {
                "identified_tools": identified,
                "target_tools": target,
                "target_steps": target_steps,
                "negative_tools": negative_tools,
                "matched_details": matched_details,
                "num_identified": metrics["num_unique_identified"],
                "num_target": len(target),
                "ordered": ordered,
                "success_rate": metrics["success_rate"],
                "success_rate_at_k": metrics["success_rate_at_k"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "pairwise_order_accuracy": metrics["pairwise_order_accuracy"],
                "id": pred.get("id"),
                "original_id": original_id,
                "slot": slot,
                "task_instruct": task_instruct
            }
            evaluations.append(eval_item)

            stats["valid"] += 1
            if metrics["success_rate"]:
                stats["success"] += 1
            stats["prec"] += metrics["precision"]["value"]
            stats["rec"] += metrics["recall"]["value"]
            if ordered:
                stats["ordered"] += 1
                for k_key in stats["sr_k"]:
                    if metrics["success_rate_at_k"]:
                        k_val = metrics["success_rate_at_k"].get(k_key)
                        if k_val is not None:
                            stats["sr_k_valid"][k_key] += 1
                            if k_val is True:
                                stats["sr_k"][k_key] += 1
                                
                if metrics["pairwise_order_accuracy"]:
                    stats["poa1"] += metrics["pairwise_order_accuracy"]["value_1"]
                    stats["poa2"] += metrics["pairwise_order_accuracy"]["value_2"]
                    stats["poa_cnt"] += 1

        save_report(evaluations, stats, output_path, k_list)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress before exit...")
        save_report(evaluations, stats, output_path, k_list)
        sys.exit(0)

if __name__ == "__main__":
    main()
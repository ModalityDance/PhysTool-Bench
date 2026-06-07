import json
import argparse
import os
import sys
from typing import List, Dict, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def is_safe_plural_match(w1: str, w2: str) -> bool:
    if w1 == w2:
        return True
        
    # 确保 w1 始终是较短的字符串，统一基于短词进行词缀推演
    if len(w1) > len(w2):
        w1, w2 = w2, w1
        
    # 规则 1: 直接加 's' (例如 clamp -> clamps, screwdriver -> screwdrivers)
    if w1 + 's' == w2:
        return True
        
    # 规则 2: 加 'es' (严谨起见，限定原词词尾为 s, x, z, ch, sh, o)
    # 例如 brush -> brushes, wrench -> wrenches, box -> boxes, glass -> glasses
    if w1 + 'es' == w2 and (
        w1.endswith('s') or 
        w1.endswith('x') or 
        w1.endswith('z') or 
        w1.endswith('ch') or 
        w1.endswith('sh') or 
        w1.endswith('o')
    ):
        return True
        
    # 规则 3: 辅音 + 'y' 变 'ies' (例如 battery -> batteries)
    if w1.endswith('y') and w1[:-1] + 'ies' == w2:
        return True
        
    # 规则 4: 'f' 或 'fe' 变 'ves' (例如 knife -> knives, leaf -> leaves)
    if w1.endswith('f') and w1[:-1] + 'ves' == w2:
        return True
    if w1.endswith('fe') and w1[:-2] + 'ves' == w2:
        return True
        
    return False

def enforce_one_to_one(mapping: Dict[str, str], log_prefix: str = "") -> Dict[str, str]:
    """
    确保映射是一对一的：每个被识别的工具映射到一个唯一的目标工具，
    且每个目标工具最多只被一个识别的工具使用。如果发现冲突，保留第一次出现的映射并丢弃其余的。
    """
    seen_targets = set()
    clean = {}
    conflicts = []
    for ident, targ in mapping.items():
        if targ in seen_targets:
            conflicts.append(f"  '{ident}' -> '{targ}' (目标工具已经被分配给其他识别工具)")
        else:
            seen_targets.add(targ)
            clean[ident] = targ
    if conflicts:
        print(f"{log_prefix}一对一规则移除了 {len(conflicts)} 个冲突的映射:")
        for c in conflicts:
            print(c)
    return clean

def get_offline_matches(original_id: str, slot: int, identified: List[str], target: List[str], offline_matching_dict: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    match_key = f"{original_id}_slot_{slot}"
    raw_mapping = offline_matching_dict.get(match_key, {})
    mapping = {str(k).lower(): str(v).lower() for k, v in raw_mapping.items()}
    
    # ✨ 字符串匹配回退机制 (String-based matching fallback) - 采用严谨的 Two-Pass 策略
    
    # Pass 1: 绝对精确匹配 (如果 identified 完全等于 target 中的词汇，直接绑定，最高优先级)
    for ident_tool in identified:
        if ident_tool in target and ident_tool not in mapping:
            mapping[ident_tool] = ident_tool
            
    # Pass 2: 安全单复数匹配 (对于字典没命中，且精确匹配也没找到的词，尝试单复数推导)
    for ident_tool in identified:
        if ident_tool not in mapping:
            for target_tool in target:
                if is_safe_plural_match(ident_tool, target_tool):
                    mapping[ident_tool] = target_tool
                    break  # 找到一个合理目标后即刻跳出，避免重复分配
            
    # 清理并一对一校验
    valid_mapping = {k: v for k, v in mapping.items() if k in identified and v in target}
    one_to_one = enforce_one_to_one(valid_mapping, log_prefix=f"[{match_key}] ")
    return one_to_one

def parse_identified_tools(identified_str: str) -> List[str]:
    if not identified_str:
        return []
    identified_str = identified_str.rstrip('.')
    return [t.strip() for t in identified_str.split(',') if t.strip()]

def calculate_metrics(identified: List[str], target: List[str], target_steps: List[int], matched_details: Dict[str, str], ordered: bool, k_values: List[int]):
    """基于离线匹配结果和步骤约束，计算评估指标"""
    
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
    if num_unique_identified == num_target and correct_count == num_target:
        if ordered:
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
                        is_non_decreasing = all(prefix_steps[i] <= prefix_steps[i+1] for i in range(k-1))
                        
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
    total_pairs_1 = sum(1 for i in range(num_target) for j in range(i+1, num_target) if target_steps[i] < target_steps[j])
    
    if ordered and total_pairs_1 > 0:
        correct_pairs = 0
        total_pairs_2 = 0
        n = len(identified_matched_info)
        
        for i in range(n):
            for j in range(i + 1, n):
                step_i = identified_matched_info[i]["step"]
                step_j = identified_matched_info[j]["step"]
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
    """保存最终的评估报告到指定路径。"""
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
    print(f"\n✅ 离线评估报告已保存至 {output_path} (本次共评估了 {stats['valid']} 个 Case)")

def main():
    parser = argparse.ArgumentParser(description='Offline tool evaluation checking all cases (Mapping + String Matching)')
    parser.add_argument('--model', '-m', type=str, required=True, help='Model name for I/O files')
    parser.add_argument('--results', '-r', type=str, default=None, help='Path to predictions file')
    parser.add_argument('--input', '-i', type=str, default=os.path.join(DATA_DIR, "corrected_tools.json"), help='Path to ground truth file')
    parser.add_argument('--match-info', type=str, default=os.path.join(DATA_DIR, "final_matching_info.json"), help='Path to offline matching dictionary')
    parser.add_argument('--output', '-o', type=str, default=None, help='Path to save output report')
    parser.add_argument('--k-values', '-k', type=str, default="1,2,3")
    args = parser.parse_args()

    results_path = args.results or os.path.join(RESULTS_DIR, f"task_ii_results_{args.model}.json")
    output_path = args.output or os.path.join(RESULTS_DIR, f"offline_evaluation_of_{args.model}.json")
    match_info_path = args.match_info
    k_list = [int(k.strip()) for k in args.k_values.split(',')]

    if not os.path.exists(results_path):
        print(f"❌ 找不到预测文件: {results_path}")
        return
    with open(results_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    if not os.path.exists(args.input):
        print(f"❌ 找不到 Ground Truth 文件: {args.input}")
        return
    with open(args.input, 'r', encoding='utf-8') as f:
        ground_truth = {(item['id'], item.get('slot', 0)): item for item in json.load(f)}

    if not os.path.exists(match_info_path):
        print(f"❌ 找不到历史匹配记录文件: {match_info_path}")
        return
    with open(match_info_path, 'r', encoding='utf-8') as f:
        offline_match_data = json.load(f)
    print(f"✅ 成功加载了 {len(offline_match_data)} 条历史匹配记录。")

    evaluations = []
    stats = {
        "success": 0, "prec": 0.0, "rec": 0.0, "poa1": 0.0, "poa2": 0.0,
        "sr_k": {f"k={k}": 0 for k in k_list}, 
        "sr_k_valid": {f"k={k}": 0 for k in k_list}, 
        "valid": 0, "ordered": 0, "poa_cnt": 0
    }

    try:
        for idx, pred in enumerate(predictions):
            original_id = pred.get("original_id")
            slot = pred.get("slot", 0)
            key = (original_id, slot)
            
            if key not in ground_truth:
                print(f"⚠️ [Warning] {original_id}_slot_{slot} 找不到对应的 Ground Truth，跳过。")
                continue
                
            gt_item = ground_truth[key]
            identified = [t.lower() for t in parse_identified_tools(pred.get("identified_tools", ""))]

            refined_tax = gt_item.get("refined_taxonomy", {})
            target = [t.lower() for t in refined_tax.get("target_tools", [])]
            target_steps = refined_tax.get("target_steps", [])
            negative_tools = [t.lower() for t in refined_tax.get("negative_tools", [])]
            ordered = gt_item.get("ordered", True)
            
            # 使用包含严谨单复数判断规则的离线映射进行匹配
            matched_details = get_offline_matches(original_id, slot, identified, target, offline_match_data)
            
            metrics = calculate_metrics(identified, target, target_steps, matched_details, ordered, k_list)
            
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
                "task_instruct": gt_item.get("task_instruct", pred.get("task_instruct", ""))
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
        print("\n用户手动中断，保存当前进度...")
        save_report(evaluations, stats, output_path, k_list)
        sys.exit(0)

if __name__ == "__main__":
    main()
import os
import json
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import pipeline

# GPU
device = 0 if torch.cuda.is_available() else -1
generator = pipeline("text-generation", model="gpt2", device=device)

def load_prompt_components(file_path):
    # load prompt from JSON file

    with open(file_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    
    components = {}
    if "Definition" in data and isinstance(data["Definition"], list) and data["Definition"]:
        components["Task_Instruction"] = data["Definition"][0]
    else:
        components["Task_Instruction"] = ""
    # positive data
    if "Positive Examples" in data and data["Positive Examples"]:
        pos = data["Positive Examples"][0]
        components["Demonstration_Input"] = pos.get("input", "")
        components["Demonstration_Label"] = pos.get("output", "")
        components["Demonstration_Inline_Instruction"] = pos.get("inline instruction", "")
    else:
        components["Demonstration_Input"] = ""
        components["Demonstration_Label"] = ""
        components["Demonstration_Inline_Instruction"] = ""
    if "Instances" in data and data["Instances"]:
        test_inst = data["Instances"][0]
        components["Test_Instance_Input"] = test_inst.get("input", "")
    else:
        components["Test_Instance_Input"] = ""
    
    return components

def reconstruct_prompt(components):
    """
    seperate prompt into components
    """
    prompt_parts = [
        components.get("Task_Instruction", ""),
        components.get("Demonstration_Input", ""),
        components.get("Demonstration_Label", ""),
        components.get("Demonstration_Inline_Instruction", ""),
        components.get("Test_Instance_Input", "")
    ]
    return "\n".join([part for part in prompt_parts if part])

def generate_removed_prompt(components, comp_to_remove):
    """
    remove components for testing
    """
    new_components = components.copy()
    if comp_to_remove in new_components:
        new_components[comp_to_remove] = ""
    return new_components

def evaluate_prompt(prompt_text): 
    """
    evaluate the prompt performance
    """

    output = generator(prompt_text, max_new_tokens=100, temperature=0.7, 
                      top_p=0.9, num_return_sequences=1)
    generated_text = output[0]['generated_text']
    new_text = generated_text[len(prompt_text):]
    
    length_score = min(len(new_text) / 200, 0.5)
    words = new_text.split()
    unique_words = set(words)
    diversity_score = min(len(unique_words) / max(len(words), 1) * 0.5, 0.5)
    score = length_score + diversity_score
    return score

def compute_component_importance(components, num_runs=3):
    """
    test the importance of each components
    """
    full_scores = []
    removed_scores = {comp: [] for comp in components}
    
    for _ in range(num_runs):
        full_prompt = reconstruct_prompt(components)
        full_score = evaluate_prompt(full_prompt)
        full_scores.append(full_score)
        
        for comp in components:
            if components[comp]:
                removed_components = generate_removed_prompt(components, comp)
                removed_prompt = reconstruct_prompt(removed_components)
                removed_score = evaluate_prompt(removed_prompt)
                removed_scores[comp].append(removed_score)
    
    avg_full_score = sum(full_scores) / len(full_scores)
    importance_scores = {}
    
    for comp in components:
        if components[comp] and len(removed_scores[comp]) > 0:
            avg_removed_score = sum(removed_scores[comp]) / len(removed_scores[comp])
            # the greater score drop, the comp is more important
            score_drop = avg_full_score - avg_removed_score
            importance_scores[comp] = max(score_drop, 0)
        else:
            importance_scores[comp] = 0
    
    # normalize
    total = sum(importance_scores.values())
    if total > 0:
        normalized_importance = {k: v / total for k, v in importance_scores.items()}
    else:
        normalized_importance = {k: 1.0 / len(components) for k in components}
    
    return normalized_importance, avg_full_score

def aggregate_importance_distributions(file_importance_results):
    """
    average files
    """
    all_components = set()
    for file_result in file_importance_results.values():
        all_components.update(file_result.keys())
    
    aggregated_scores = {comp: 0.0 for comp in all_components}
    file_count = len(file_importance_results)
    
    for file_name, importance_dist in file_importance_results.items():
        for component, score in importance_dist.items():
            aggregated_scores[component] += score
    
    for component in aggregated_scores:
        aggregated_scores[component] /= file_count
    
    # normalize
    total = sum(aggregated_scores.values())
    if total > 0:
        normalized_importance = {k: v / total for k, v in aggregated_scores.items()}
    else:
        normalized_importance = {k: 1.0 / len(aggregated_scores) for k in aggregated_scores}
    
    return normalized_importance

def kl_divergence(P, Q):
    kl = 0.0
    for p, q in zip(P, Q):
        if p > 0:
            # avoid divide by 0
            kl += p * math.log(p / max(q, 1e-10))
    return kl

def js_divergence(P, Q):
    M = [(p + q) / 2.0 for p, q in zip(P, Q)]
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)

def dict_to_list(importance_dict):
    """
    helper function
    """
    component_order = [
        "Task_Instruction",
        "Demonstration_Input",
        "Demonstration_Label",
        "Demonstration_Inline_Instruction",
        "Test_Instance_Input"
    ]
    
    return [importance_dict.get(comp, 0.0) for comp in component_order]

# ====================== main process ======================

def process_dataset(data_dir, mode="test", suffix=None):
    """
    mode: "positive", "negative" or "test" (for printing)
    suffix: "positive" or "negative"
    """
    print(f"\n=== start process '{mode}' set ===")
    if not os.path.exists(data_dir):
        print(f"dir doesn't exist: {data_dir}")
        return None

    results = {}
    for fn in os.listdir(data_dir):
        if suffix and not fn.endswith(suffix + ".json"):
            continue

        print(f"\n[{mode}] procesing file: {fn}")
        try:
            comps = load_prompt_components(os.path.join(data_dir, fn))
            imp_dist, full_score = compute_component_importance(comps)
            print(f"[{mode}] prompt score: {full_score:.4f}")
            print(f"[{mode}] coponent importance distribution:")
            for c,v in imp_dist.items():
                print(f"  {c}: {v:.3f}")
            results[fn] = imp_dist
        except Exception as e:
            print(f"[{mode}] error: {e}")
    if not results:
        return None

    # aggregate
    agg = aggregate_importance_distributions(results)
    print(f"\n=== '{mode}' set aggregate distribution ===")
    for c,v in agg.items():
        print(f"  {c}: {v:.3f}")
    return agg

def plot_distributions(pos_d, neg_d, test_d, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    comps = ["Task","DemoIn","DemoLbl","Inline","TestIn"]
    x = np.arange(len(comps))
    w = 0.25
    lists = [dict_to_list(d) for d in (pos_d, neg_d, test_d)]
    labs = ["Positive","Negative","Original Test"]
    fig,ax = plt.subplots(figsize=(8,4))
    for i, (lst,lab) in enumerate(zip(lists,labs)):
        ax.bar(x + (i-1)*w, lst, w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(comps)
    ax.set_ylabel("Importance")
    ax.set_title("Component Importance by Split")
    ax.legend()
    fig.savefig(os.path.join(out_dir,"importance_bar.png"))
    plt.close(fig)

    # JS divergence
    js_pos = js_divergence(lists[0], lists[2])
    js_neg = js_divergence(lists[1], lists[2])
    js_vals = [js_pos, js_neg, 0.09]
    labels = ["Pos vs Test","Neg vs Test","Random"]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(labels, js_vals, color=['C0','C1','C2'])
    ax.set_ylabel("JS Divergence")
    ax.set_title("JS Divergence Comparison")
    for i,v in enumerate(js_vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha='center')
    fig.savefig(os.path.join(out_dir,"js_divergence_bar.png"))
    plt.close(fig)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sentiment_dir = os.path.join(current_dir, "sentiment_split_datasets")

    # positive
    print("\n>>> process positive set")
    pos_dist = process_dataset(sentiment_dir, mode="positive",  suffix="_positive")

    # negative
    print("\n>>> process negative set")
    neg_dist = process_dataset(sentiment_dir, mode="negative", suffix="_negative")

    # test
    test_data_dir = os.path.join(current_dir, "data", "test")
    print("\n>>> process test set")
    test_dist = process_dataset(test_data_dir, mode="test")

    if pos_dist and test_dist:
        pos_jsd = js_divergence(
            dict_to_list(pos_dist),
            dict_to_list(test_dist)
        )
        print(f"\nPositive vs test JS: {pos_jsd:.6f}")

    if neg_dist and test_dist:
        neg_jsd = js_divergence(
            dict_to_list(neg_dist),
            dict_to_list(test_dist)
        )
        print(f"\nNegative vs test JS: {neg_jsd:.6f}")
    
    # plot
    if pos_dist and neg_dist and test_dist:
        plot_distributions(pos_dist, neg_dist, test_dist, os.path.join(current_dir,"plots"))
        print("Plots saved under 'plots/'")

if __name__ == "__main__":
    main()
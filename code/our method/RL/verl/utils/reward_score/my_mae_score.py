import json
import re
from os.path import dirname, join

format_penalty = -1.0
outcome_weight = 0.7
p_weight = 0.3
max_distance = 4


def extract_solution(solution_str):
    match = re.fullmatch(
        r'.*?<answer>\s*(\[.*\])\s*###\s*([1-5])\s*</answer>\s*',
        solution_str,
        flags=re.S,
    )
    if match is None:
        return None

    try:
        data = json.loads(match.group(1))
        score_after_hash = int(match.group(2))

        if not isinstance(data, list) or len(data) < 2:
            return None


        summary = data[-1]
        if not isinstance(summary, dict):
            return None

        final_score = summary.get("satisfaction_score")

        if type(final_score) is not int or not 1 <= final_score <= 5:
            return None


        if final_score != score_after_hash:
            return None


        intermediate_labels = []
        for item in data[:-1]:
            if not isinstance(item, dict):
                return None

            polarity = item.get("polarity")
            if polarity not in ("+", "-", "0"):
                return None

            intermediate_labels.append(polarity)

        return {
            "intermediate_labels": intermediate_labels,
            "final_score": final_score,
        }

    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def compute_score(solution_str, ground_truth, extra_info=None):
    with open(join(dirname(__file__), "reference_signal_labels.json"), encoding="utf-8") as file:
        labels_by_query_id = json.load(file)
    query_id = str(extra_info["query_id"])
    intermediate_labels = [str(value) for value in labels_by_query_id[query_id]]

    parsed = extract_solution(solution_str)
    if parsed is None:
        return format_penalty

    try:
        y_pred = parsed["final_score"]
        y_true = int(ground_truth)
    except (TypeError, ValueError):
        return format_penalty
    if y_true < 1 or y_true > 5:
        return format_penalty

    predicted_labels = [str(value) for value in parsed["intermediate_labels"]]
    if len(predicted_labels) != len(intermediate_labels):
        return format_penalty

    distance = abs(y_pred - y_true)
    outcome_reward = max(0.0, 1.0 - distance / max_distance)

    selected = [i for i, polarity in enumerate(intermediate_labels) if polarity != "0"]
    process_reward = (
        sum(predicted_labels[i] == intermediate_labels[i] for i in selected) / len(selected)
        if selected else 0.0
    )
    return outcome_weight * outcome_reward + p_weight * process_reward

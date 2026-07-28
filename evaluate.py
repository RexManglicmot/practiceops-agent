import json

from agents import action_agent, diagnostic_agent


def precision_recall_f1(predicted_sets: list, expected_sets: list) -> tuple:
    tp = fp = fn = 0
    for predicted, expected in zip(predicted_sets, expected_sets):
        predicted, expected = set(predicted), set(expected)
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _labels(result: dict) -> list:
    return result.get("issues", result.get("actions", []))


def evaluate():
    with open("test_cases.json") as f:
        test_cases = json.load(f)

    agent1_predicted, agent1_expected = [], []
    agent2_predicted, agent2_expected = [], []

    total_responses = 0
    valid_json_count = 0
    total_predicted_labels = 0
    unsupported_count = 0

    for case in test_cases:
        result1 = diagnostic_agent(case["metrics"])
        agent1_predicted.append(result1["issues"])
        agent1_expected.append(case["expected_issues"])

        # Agent 2 is evaluated on ground-truth issues so its score isn't
        # polluted by Agent 1's classification errors.
        result2 = action_agent(case["expected_issues"])
        agent2_predicted.append(result2["actions"])
        agent2_expected.append(case["expected_actions"])

        for result in (result1, result2):
            total_responses += 1
            valid_json_count += int(result["valid_json"])
            unsupported_count += result["unsupported_count"]
            total_predicted_labels += len(_labels(result)) + result["unsupported_count"]

    p1, r1, f1_1 = precision_recall_f1(agent1_predicted, agent1_expected)
    p2, r2, f1_2 = precision_recall_f1(agent2_predicted, agent2_expected)

    json_validity_rate = valid_json_count / total_responses if total_responses else 0.0
    unsupported_label_rate = (
        unsupported_count / total_predicted_labels if total_predicted_labels else 0.0
    )

    print(f"{'Agent':<20}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    print("-" * 56)
    print(f"{'Diagnostic Agent':<20}{p1:<12.2f}{r1:<12.2f}{f1_1:<12.2f}")
    print(f"{'Action Agent':<20}{p2:<12.2f}{r2:<12.2f}{f1_2:<12.2f}")
    print()
    print(f"JSON validity rate: {json_validity_rate:.0%}")
    print(f"Unsupported-label rate: {unsupported_label_rate:.0%}")


if __name__ == "__main__":
    evaluate()

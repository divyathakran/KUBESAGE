"""
Guardrails evaluation.

Runs each test case through the real guard() and compares the outcome with
`expected_blocked`.

"Blocked" here means the guard returned the OFF_TOPIC or JAILBREAK refusal.
Greeting / farewell / capabilities replies are friendly canned answers, not
blocks, so they count as "not blocked".
"""

import time

from app.guardrails import rails


# Refusal replies that count as "blocked"
BLOCK_REPLIES = {rails.RESPONSES["OFF_TOPIC"], rails.RESPONSES["JAILBREAK"]}


def _ensure_rails() -> None:
    """
    The eval runs inside the Streamlit process, not FastAPI, so nobody has
    called initialize_rails() yet. Without this, guard() silently passes
    everything ("not initialised") and every adversarial test would fail.
    """
    if rails._guard_llm is None:
        rails.initialize_rails()


def _notify(callback, done: int, total: int, text: str) -> None:
    """Call the progress callback, tolerating different signatures."""
    if callback is None:
        return
    for args in ((done, total, text), (done, total), (done / total, text), (done / total,)):
        try:
            callback(*args)
            return
        except TypeError:
            continue


def run_guardrails_eval(samples: list[dict], progress_callback=None) -> list[dict]:
    """Run every sample through guard(). Returns one result dict per sample."""
    _ensure_rails()

    results = []
    total = len(samples)

    for i, sample in enumerate(samples, start=1):
        _notify(progress_callback, i - 1, total, f"Testing {sample.get('id', i)}...")

        start = time.perf_counter()
        error = None
        try:
            fired, reply = rails.guard(sample["input"])
        except Exception as e:  # keep the whole run alive if one case crashes
            fired, reply, error = False, None, str(e)
        latency = round(time.perf_counter() - start, 3)

        actual_blocked = bool(fired and reply in BLOCK_REPLIES)
        expected_blocked = bool(sample.get("expected_blocked"))
        correct = actual_blocked == expected_blocked

        result = dict(sample)  # keep id, input, type, description, ...
        result.update(
            {
                "expected_blocked": expected_blocked,
                "actual_blocked": actual_blocked,
                "correct": correct,
                "passed": correct,
                "response": reply,
                "latency": latency,
                "latency_s": latency,
                "error": error,
            }
        )
        results.append(result)

        _notify(progress_callback, i, total, f"Tested {sample.get('id', i)}")

    return results


def compute_guardrails_metrics(results: list[dict]) -> dict:
    """
    Treat 'blocked' as the positive class:
      TP = adversarial and blocked      FN = adversarial but let through
      TN = legit and passed             FP = legit but wrongly blocked
    Rates are fractions between 0 and 1; *_pct versions are 0-100.
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = sum(1 for r in results if r["expected_blocked"] and r["actual_blocked"])
    fn = sum(1 for r in results if r["expected_blocked"] and not r["actual_blocked"])
    tn = sum(1 for r in results if not r["expected_blocked"] and not r["actual_blocked"])
    fp = sum(1 for r in results if not r["expected_blocked"] and r["actual_blocked"])

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    latencies = [r["latency"] for r in results if r.get("latency") is not None]

    return {
        "total": total,
        "correct": tp + tn,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "accuracy_pct": round(accuracy * 100, 1),
        "precision_pct": round(precision * 100, 1),
        "recall_pct": round(recall * 100, 1),
        "f1_pct": round(f1 * 100, 1),
        "avg_latency": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
    }
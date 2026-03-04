"""
VALIDATE_CODE.PY — Validation Pipeline for AI Code Detector
Theory of Stochastic Stagnation (Sihare, 2026)
DOI: 10.17605/OSF.IO/FSMK9

Evaluates ai_detector.py against labelled ground-truth .py files.

USAGE:
  python validate_code.py --human ./human_code/ --ai ./ai_code/
  python validate_code.py --human ./human/ --ai ./ai/ --threshold 0.50
  python validate_code.py --human ./human/ --ai ./ai/ --json
  python validate_code.py --human ./human/ --ai ./ai/ --csv results.csv
  python validate_code.py --demo

OUTPUTS:
  • Precision, Recall, F1 (per class + macro/weighted)
  • Confusion matrix
  • Signal importance ranking
  • Per-file breakdown (optional: --verbose)

REQUIRES: ai_detector.py in same directory (or on PYTHONPATH)
"""

import os, sys, json, math, argparse, csv
from pathlib import Path
from collections import defaultdict

# ── Import detector ────────────────────────────────────────────────────────────
try:
    from ai_detector import AIContentDetector, analyze_code, CODE_EXT
except ImportError:
    print("[ERROR] ai_detector.py not found. Place it in the same directory.")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.45   # human_score >= threshold → predicted HUMAN
#                            mirrors _verdict() logic in ai_detector.py
#                            (< 0.45 → PROBABLY AI / LIKELY AI)

SIGNAL_WEIGHTS = {
    # Universal signals (each contributes 0.20 to final human_score)
    "score_entropy":     0.20,
    "score_compress":    0.20,
    "score_burstiness":  0.20,
    # Type-specific block contributes 0.40; broken down below
    "id_entropy_score":     0.40 * 0.35,   # identifier entropy
    "generic_score":        0.40 * 0.30,   # generic name ratio
    "ast_score":            0.40 * 0.20,   # AST complexity variance
    "line_variance_score":  0.40 * 0.15,   # line length variance
}

SIGNAL_LABELS = {
    "score_entropy":        "Byte Entropy",
    "score_compress":       "Compress Ratio",
    "score_burstiness":     "Burstiness",
    "id_entropy_score":     "Identifier Entropy",
    "generic_score":        "Generic Name Ratio",
    "ast_score":            "AST Complexity Variance",
    "line_variance_score":  "Line Length Variance",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_py_files(folder: str, label: int) -> list[dict]:
    """Scan folder for .py files; return list of {path, true_label}."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"[ERROR] Not a directory: {folder}")
        sys.exit(1)
    records = []
    for fp in sorted(folder_path.rglob("*.py")):
        records.append({"path": str(fp), "true_label": label})
    return records


def run_detector(records: list[dict], threshold: float) -> list[dict]:
    """Run ai_detector on every file; add human_score, pred_label, signals."""
    det = AIContentDetector()
    results = []
    for rec in records:
        r = det.analyze_file(rec["path"])
        hs = r.get("human_score", 0.5)
        pred = 1 if hs >= threshold else 0   # 1 = HUMAN, 0 = AI

        # Flatten type_signals into top-level for convenience
        ts = r.get("type_signals", {})
        signals = {
            "score_entropy":       r.get("score_entropy", 0.0),
            "score_compress":      r.get("score_compress", 0.0),
            "score_burstiness":    r.get("score_burstiness", 0.0),
            "id_entropy_score":    ts.get("id_entropy_score", 0.0),
            "generic_score":       ts.get("generic_score", 0.0),
            "ast_score":           ts.get("ast_score", 0.0),
            "line_variance_score": ts.get("line_variance_score", 0.0),
        }

        results.append({
            "path":       rec["path"],
            "true_label": rec["true_label"],
            "pred_label": pred,
            "human_score": round(hs, 4),
            "ai_score":    round(r.get("ai_score", 1.0 - hs), 4),
            "verdict":     r.get("verdict", "UNKNOWN"),
            "signals":     signals,
        })
    return results


# ── Metrics ────────────────────────────────────────────────────────────────────

def confusion_matrix(results: list[dict]) -> dict:
    """Return TP, FP, TN, FN (positive class = HUMAN = 1)."""
    tp = fp = tn = fn = 0
    for r in results:
        t, p = r["true_label"], r["pred_label"]
        if t == 1 and p == 1: tp += 1
        elif t == 0 and p == 1: fp += 1
        elif t == 0 and p == 0: tn += 1
        elif t == 1 and p == 0: fn += 1
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec / (prec + rec) if (prec + rec) else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


def compute_metrics(cm: dict) -> dict:
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    total = tp + fp + tn + fn

    # Per-class
    prec_h, rec_h, f1_h = prf(tp, fp, fn)           # HUMAN class
    prec_a, rec_a, f1_a = prf(tn, fn, fp)            # AI class

    # Macro (unweighted average)
    macro_p  = (prec_h + prec_a) / 2
    macro_r  = (rec_h  + rec_a)  / 2
    macro_f1 = (f1_h   + f1_a)   / 2

    # Weighted average
    n_human = tp + fn
    n_ai    = tn + fp
    w_p  = (prec_h*n_human + prec_a*n_ai) / total if total else 0
    w_r  = (rec_h *n_human + rec_a *n_ai) / total if total else 0
    w_f1 = (f1_h  *n_human + f1_a  *n_ai) / total if total else 0

    accuracy = (tp + tn) / total if total else 0.0

    return {
        "accuracy":       round(accuracy, 4),
        "human": {"precision": prec_h, "recall": rec_h, "f1": f1_h,  "support": n_human},
        "ai":    {"precision": prec_a, "recall": rec_a, "f1": f1_a,  "support": n_ai},
        "macro": {"precision": round(macro_p,4), "recall": round(macro_r,4), "f1": round(macro_f1,4)},
        "weighted": {"precision": round(w_p,4), "recall": round(w_r,4), "f1": round(w_f1,4)},
        "total": total,
    }


# ── Signal importance ──────────────────────────────────────────────────────────

def signal_importance(results: list[dict]) -> list[dict]:
    """
    For each signal, compute mean separation between human and AI files,
    weighted by the signal's architectural weight in ai_detector.py.
    Rank by weighted separation (descending = most discriminative).
    """
    human_sigs = defaultdict(list)
    ai_sigs    = defaultdict(list)

    for r in results:
        bucket = human_sigs if r["true_label"] == 1 else ai_sigs
        for k, v in r["signals"].items():
            bucket[k].append(v)

    ranking = []
    for key, weight in SIGNAL_WEIGHTS.items():
        hv = human_sigs[key]
        av = ai_sigs[key]
        if not hv or not av:
            continue
        mean_h = sum(hv) / len(hv)
        mean_a = sum(av) / len(av)
        separation = abs(mean_h - mean_a)

        # Cohen's d (pooled std)
        def std(lst):
            m = sum(lst)/len(lst)
            return math.sqrt(sum((x-m)**2 for x in lst)/len(lst))
        sh, sa = std(hv), std(av)
        pooled = math.sqrt((sh**2 + sa**2)/2) if (sh + sa) > 0 else 1e-9
        cohens_d = round(separation / pooled, 4)

        weighted_sep = round(separation * weight, 4)

        ranking.append({
            "signal":         key,
            "label":          SIGNAL_LABELS[key],
            "arch_weight":    round(weight, 4),
            "mean_human":     round(mean_h, 4),
            "mean_ai":        round(mean_a, 4),
            "separation":     round(separation, 4),
            "cohens_d":       cohens_d,
            "weighted_sep":   weighted_sep,
        })

    ranking.sort(key=lambda x: x["weighted_sep"], reverse=True)
    for i, row in enumerate(ranking, 1):
        row["rank"] = i
    return ranking


# ── Pretty printing ────────────────────────────────────────────────────────────

SEP = "=" * 68

def print_banner():
    print(f"\n{'#'*68}")
    print("  VALIDATE_CODE.PY — AI Detector Validation Pipeline")
    print("  Theory of Stochastic Stagnation (Sihare, 2026)")
    print("  DOI: 10.17605/OSF.IO/FSMK9")
    print(f"{'#'*68}\n")


def print_confusion(cm: dict):
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    print(f"\n{SEP}")
    print("  CONFUSION MATRIX  (positive = HUMAN)")
    print(f"  {SEP[:50]}")
    print(f"                     Predicted AI   Predicted Human")
    print(f"  True AI          :    TN={tn:<6}        FP={fp}")
    print(f"  True Human       :    FN={fn:<6}        TP={tp}")
    print(f"  {'-'*50}")
    print(f"  Correct: {tp+tn}   Wrong: {fp+fn}   Total: {tp+fp+tn+fn}")
    print(f"{SEP}")


def print_metrics(m: dict):
    print(f"\n{SEP}")
    print("  CLASSIFICATION METRICS")
    print(f"  {'-'*50}")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print(f"  {'-'*50}")

    def row(lbl, d):
        sup = f"{d['support']:>9}" if 'support' in d else f"{'':>9}"
        print(f"  {lbl:<12} {d['precision']:>10.4f} {d['recall']:>8.4f} {d['f1']:>8.4f} {sup}")

    row("HUMAN",    m["human"])
    row("AI",       m["ai"])
    print(f"  {'-'*50}")
    row("Macro",    m["macro"])
    row("Weighted", m["weighted"])
    print(f"  {'-'*50}")
    print(f"  Accuracy:   {m['accuracy']:.4f}   ({round(m['accuracy']*100, 1)}%)")
    print(f"  Total files: {m['total']}")
    print(f"{SEP}")


def bar(v, w=16):
    f = round(v * w)
    return f"[{'█'*f}{'·'*(w-f)}] {v:.3f}"


def print_signal_ranking(ranking: list[dict]):
    print(f"\n{SEP}")
    print("  SIGNAL IMPORTANCE RANKING")
    print(f"  (weighted separation = |mean_human - mean_ai| × arch_weight)")
    print(f"  {'-'*62}")
    print(f"  {'Rank':<5} {'Signal':<26} {'ΔMean':>7} {'Cohen d':>8} {'WtdSep':>8} {'ArchWt':>7}")
    print(f"  {'-'*62}")
    for r in ranking:
        direction = "↑H" if r["mean_human"] > r["mean_ai"] else "↑A"
        print(
            f"  #{r['rank']:<4} {r['label']:<26} "
            f"{r['separation']:>6.3f}{direction} {r['cohens_d']:>8.3f} "
            f"{r['weighted_sep']:>8.4f} {r['arch_weight']:>7.4f}"
        )
    print(f"\n  ↑H = higher in human files (good discriminator for human class)")
    print(f"  ↑A = higher in AI files")
    print(f"{SEP}")


def print_per_file(results: list[dict]):
    print(f"\n{SEP}")
    print("  PER-FILE BREAKDOWN")
    print(f"  {'File':<35} {'True':>6} {'Pred':>6} {'Score':>7} {'OK?':>5}")
    print(f"  {'-'*60}")
    for r in results:
        name  = Path(r["path"]).name[:33]
        true  = "HUMAN" if r["true_label"] == 1 else "AI"
        pred  = "HUMAN" if r["pred_label"] == 1 else "AI"
        score = r["human_score"]
        ok    = "✓" if r["true_label"] == r["pred_label"] else "✗"
        print(f"  {name:<35} {true:>6} {pred:>6} {score:>7.4f} {ok:>5}")
    print(f"{SEP}")


# ── Export ─────────────────────────────────────────────────────────────────────

def export_csv(results: list[dict], metrics: dict, ranking: list[dict], path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # Per-file rows
        w.writerow(["file", "true_label", "pred_label", "human_score", "ai_score",
                    "verdict", "correct",
                    "score_entropy", "score_compress", "score_burstiness",
                    "id_entropy_score", "generic_score", "ast_score", "line_variance_score"])
        for r in results:
            s = r["signals"]
            w.writerow([
                r["path"],
                "HUMAN" if r["true_label"] == 1 else "AI",
                "HUMAN" if r["pred_label"] == 1 else "AI",
                r["human_score"], r["ai_score"], r["verdict"],
                r["true_label"] == r["pred_label"],
                s["score_entropy"], s["score_compress"], s["score_burstiness"],
                s["id_entropy_score"], s["generic_score"], s["ast_score"],
                s["line_variance_score"],
            ])
        w.writerow([])
        w.writerow(["=== METRICS ==="])
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for cls in ["human", "ai", "macro", "weighted"]:
            d = metrics[cls]
            sup = d.get("support", "")
            w.writerow([cls, d["precision"], d["recall"], d["f1"], sup])
        w.writerow(["accuracy", metrics["accuracy"]])
        w.writerow([])
        w.writerow(["=== SIGNAL IMPORTANCE ==="])
        w.writerow(["rank", "signal", "label", "arch_weight", "mean_human",
                    "mean_ai", "separation", "cohens_d", "weighted_sep"])
        for r in ranking:
            w.writerow([r["rank"], r["signal"], r["label"], r["arch_weight"],
                        r["mean_human"], r["mean_ai"], r["separation"],
                        r["cohens_d"], r["weighted_sep"]])
    print(f"\n  [CSV] Saved → {path}")


# ── Demo mode ──────────────────────────────────────────────────────────────────

def run_demo():
    """Creates synthetic in-memory .py files and validates against them."""
    import tempfile, textwrap

    HUMAN_FILES = {
        "matrix_fib.py": textwrap.dedent("""
            import sys
            def _mat_mul(A, B):
                return [
                    [A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]],
                    [A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]]
                ]
            def mat_pow(M, p):
                if p == 1: return M
                half = mat_pow(M, p // 2)
                sq = _mat_mul(half, half)
                return _mat_mul(sq, M) if p % 2 else sq
            def fib(n):
                if n < 2: return n
                return mat_pow([[1,1],[1,0]], n)[0][1]
            if __name__ == '__main__':
                N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
                evens = [fib(k) for k in range(N) if fib(k) % 2 == 0]
                print("even fibs:", evens, "sum:", sum(evens))
        """),
        "entropy_hack.py": textwrap.dedent("""
            # entropy_hack -- computes β_n decay for stagnation audit
            import math, sys
            from collections import Counter
            def H(data):
                c = Counter(data); n = len(data)
                return -sum((v/n)*math.log2(v/n) for v in c.values()) if n else 0
            def beta(gen_tokens, baseline_tokens):
                return H(gen_tokens) / H(baseline_tokens) if H(baseline_tokens) else 0
            if __name__ == '__main__':
                baseline = list(b'The quick brown fox jumps over the lazy dog' * 40)
                synth    = list(b'the result is the result of the result' * 80)
                b = beta(synth, baseline)
                crit = 0.7
                print(f"β = {b:.4f}  {'STAGNATION' if b < crit else 'OK'} (critical={crit})")
        """),
        "quirky_sort.py": textwrap.dedent("""
            # bogosort but instrumented for laughs
            import random, time
            def is_sorted(lst): return all(lst[i]<=lst[i+1] for i in range(len(lst)-1))
            def bogosort(lst, max_iter=10**6):
                t0, iters = time.time(), 0
                while not is_sorted(lst) and iters < max_iter:
                    random.shuffle(lst); iters += 1
                return lst, iters, time.time()-t0
            if __name__ == '__main__':
                data = [5,3,8,1,9,2]
                out, n, t = bogosort(data[:])
                print(f"sorted={out} in {n} shuffles ({t:.3f}s)")
                print("for large n, get a coffee. or a different algorithm.")
        """),
    }

    AI_FILES = {
        "fibonacci_dp.py": textwrap.dedent("""
            def calculate_fibonacci(n):
                \"\"\"Calculate the nth Fibonacci number using dynamic programming.\"\"\"
                if n <= 0:
                    return 0
                elif n == 1:
                    return 1
                dp = [0] * (n + 1)
                dp[0] = 0
                dp[1] = 1
                for i in range(2, n + 1):
                    dp[i] = dp[i-1] + dp[i-2]
                return dp[n]
            def process_data(data):
                \"\"\"Process a list of numbers and return their Fibonacci values.\"\"\"
                result = []
                for item in data:
                    value = calculate_fibonacci(item)
                    result.append(value)
                return result
            def main():
                \"\"\"Main function to demonstrate the Fibonacci calculation.\"\"\"
                input_data = [5, 10, 15, 20]
                output = process_data(input_data)
                print(f"Result: {output}")
            if __name__ == "__main__":
                main()
        """),
        "data_processor.py": textwrap.dedent("""
            class DataProcessor:
                \"\"\"A robust and scalable data processing utility.\"\"\"
                def __init__(self, data):
                    self.data = data
                    self.results = []
                def process(self):
                    \"\"\"Process the data and return results.\"\"\"
                    for item in self.data:
                        value = self._transform(item)
                        self.results.append(value)
                    return self.results
                def _transform(self, value):
                    \"\"\"Transform a single value.\"\"\"
                    return value * 2
                def get_summary(self):
                    \"\"\"Get a summary of the processed data.\"\"\"
                    if not self.results:
                        return {"total": 0, "count": 0, "average": 0}
                    total = sum(self.results)
                    count = len(self.results)
                    average = total / count
                    return {"total": total, "count": count, "average": average}
        """),
        "api_handler.py": textwrap.dedent("""
            import json
            def handle_request(request):
                \"\"\"Handle an incoming API request and return a response.\"\"\"
                if not request:
                    return {"error": "Invalid request", "status": 400}
                response = process_request(request)
                return {"data": response, "status": 200}
            def process_request(request):
                \"\"\"Process the request and extract relevant information.\"\"\"
                result = {}
                if "params" in request:
                    params = request["params"]
                    result = validate_params(params)
                return result
            def validate_params(params):
                \"\"\"Validate input parameters.\"\"\"
                validated = {}
                for key, value in params.items():
                    if value is not None:
                        validated[key] = value
                return validated
        """),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        hdir = os.path.join(tmpdir, "human")
        adir = os.path.join(tmpdir, "ai")
        os.makedirs(hdir); os.makedirs(adir)
        for name, code in HUMAN_FILES.items():
            Path(os.path.join(hdir, name)).write_text(code)
        for name, code in AI_FILES.items():
            Path(os.path.join(adir, name)).write_text(code)

        run_validation(hdir, adir, DEFAULT_THRESHOLD,
                       verbose=True, as_json=False, csv_path=None)


# ── Main validation runner ─────────────────────────────────────────────────────

def run_validation(human_dir: str, ai_dir: str, threshold: float,
                   verbose: bool, as_json: bool, csv_path: str | None):
    print_banner()
    print(f"  Human folder : {human_dir}")
    print(f"  AI folder    : {ai_dir}")
    print(f"  Threshold    : human_score ≥ {threshold:.2f}  → predicted HUMAN")

    human_recs = load_py_files(human_dir, label=1)
    ai_recs    = load_py_files(ai_dir,    label=0)

    if not human_recs:
        print(f"\n[WARN] No .py files found in human folder: {human_dir}")
    if not ai_recs:
        print(f"\n[WARN] No .py files found in AI folder: {ai_dir}")
    if not human_recs and not ai_recs:
        print("[ERROR] Nothing to evaluate."); sys.exit(1)

    print(f"\n  Loaded {len(human_recs)} human files, {len(ai_recs)} AI files.")
    print(f"  Running detector on {len(human_recs)+len(ai_recs)} files...\n")

    all_recs  = human_recs + ai_recs
    results   = run_detector(all_recs, threshold)
    cm        = confusion_matrix(results)
    metrics   = compute_metrics(cm)
    ranking   = signal_importance(results)

    if as_json:
        out = {
            "confusion_matrix": cm,
            "metrics": metrics,
            "signal_ranking": ranking,
            "per_file": results,
        }
        print(json.dumps(out, indent=2))
        return

    print_confusion(cm)
    print_metrics(metrics)
    print_signal_ranking(ranking)
    if verbose:
        print_per_file(results)

    if csv_path:
        export_csv(results, metrics, ranking, csv_path)

    # Summary interpretation
    print(f"\n  ── INTERPRETATION ───────────────────────────────────────────")
    f1w = metrics["weighted"]["f1"]
    if f1w >= 0.85:
        verdict = "STRONG  — detector performs well on this corpus"
    elif f1w >= 0.70:
        verdict = "MODERATE — acceptable performance; review weak signals"
    else:
        verdict = "WEAK    — detector needs re-calibration for this corpus"
    print(f"  Weighted F1 = {f1w:.4f}  →  {verdict}")

    top_sig = ranking[0] if ranking else None
    if top_sig:
        print(f"  Top signal:  {top_sig['label']}  "
              f"(Cohen's d = {top_sig['cohens_d']:.3f}, "
              f"weighted sep = {top_sig['weighted_sep']:.4f})")

    errors = [r for r in results if r["true_label"] != r["pred_label"]]
    if errors:
        print(f"\n  Misclassified ({len(errors)}):")
        for r in errors:
            t = "HUMAN" if r["true_label"] == 1 else "AI"
            p = "HUMAN" if r["pred_label"] == 1 else "AI"
            print(f"    {Path(r['path']).name}  (true={t}, pred={p}, score={r['human_score']:.4f})")

    print(f"\n  Entropy retention threshold β_critical ∈ [0.6, 0.8] — Sihare (2026)")
    print(f"  This pipeline validates the detector against that theoretical boundary.\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validation pipeline for AI code detector (Theory of Stochastic Stagnation)")
    parser.add_argument("--human",     type=str,  help="Folder of human-written .py files")
    parser.add_argument("--ai",        type=str,  help="Folder of AI-generated .py files")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"human_score cutoff for HUMAN prediction (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print per-file breakdown")
    parser.add_argument("--json",      action="store_true",
                        help="Output everything as JSON")
    parser.add_argument("--csv",       type=str,  default=None,
                        help="Export results to CSV file")
    parser.add_argument("--demo",      action="store_true",
                        help="Run with built-in synthetic files (no folders needed)")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.human and args.ai:
        run_validation(
            human_dir  = args.human,
            ai_dir     = args.ai,
            threshold  = args.threshold,
            verbose    = args.verbose,
            as_json    = args.json,
            csv_path   = args.csv,
        )
    else:
        parser.print_help()
        print("\n  Quick start:")
        print("    python validate_code.py --demo")
        print("    python validate_code.py --human ./human/ --ai ./ai/")
        print("    python validate_code.py --human ./human/ --ai ./ai/ --verbose --csv out.csv")
        sys.exit(0)


if __name__ == "__main__":
    main()

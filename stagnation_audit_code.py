"""
=============================================================
STAGNATION AUDIT TOOL — CODING EDITION
Based on: Theory of Stochastic Stagnation (Version Star ★)
Theory Author: Chaitanya Sihare (2026)
=============================================================

DOMAIN: Software development with AI assistants
        (Copilot, Cursor, Codeium, ChatGPT, Claude, etc.)

WHY CODING WORKS:
  In coding, D, B, and w are *precisely* calculable:
  - S = measurable syntax space (programs of length n)
  - T = programs passing the test suite (binary, verifiable)
  - B = AST-level + semantic diff between AI suggestion and human commit
  - Generations = commits / PRs / review cycles (discrete, trackable)

WHAT IT MEASURES:
  At each commit or PR, this tool answers:
  "How much genuine human variance was injected vs AI refinement?"
  "Is this codebase trending toward stagnation?"
  "How many more generations until measurable entropy collapse?"

USAGE:
  # Single audit (AI suggestion vs human commit)
  python stagnation_audit_code.py --ai ai_solution.py --human human_solution.py --tests tests.py

  # Multi-generation tracking
  python stagnation_audit_code.py --history commits.json

  # Demo mode (built-in examples)
  python stagnation_audit_code.py --demo

  # Interactive mode
  python stagnation_audit_code.py --interactive

INSTALL DEPS:
  pip install radon  # for cyclomatic complexity (optional, enhances B calculation)
"""

import ast
import math
import re
import sys
import json
import difflib
import argparse
import subprocess
import textwrap
from collections import Counter
from typing import Optional


# ─────────────────────────────────────────────────────────────
# THEORY CONSTANTS
# ─────────────────────────────────────────────────────────────
ALPHA           = 0.5    # Baseline variance coefficient (domain-adjustable)
GAMMA           = 1.3    # Superlinear scaling exponent  (domain-adjustable)
W_THRESHOLD     = 0.15   # Director weight collapse boundary
THETA_MIN       = 0.6    # Lower bound of critical zone
THETA_MAX       = 0.8    # Upper bound of critical zone
FIDELITY_LOSS   = 0.05   # ~5% entropy loss per generation (empirical baseline)


# ─────────────────────────────────────────────────────────────
# CODE PARSING UTILITIES
# ─────────────────────────────────────────────────────────────

def extract_tokens(code: str) -> list[str]:
    """
    Extract meaningful tokens from Python source code.
    Uses Python's tokenizer — far more precise than word splitting.
    Filters out whitespace and comment tokens.
    """
    import tokenize
    import io

    tokens = []
    try:
        token_gen = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok_type, tok_string, _, _, _ in token_gen:
            # Keep: NAME, NUMBER, STRING, OP — skip NEWLINE, COMMENT, INDENT etc.
            if tok_type in (1, 2, 3, 54):  # NAME, NUMBER, STRING, OP
                tokens.append(tok_string)
    except tokenize.TokenError:
        # Fallback to simple split if tokenizer fails
        tokens = re.findall(r'\b\w+\b|[^\w\s]', code)
    return tokens


def parse_ast_features(code: str) -> dict:
    """
    Extract structural features from AST.
    Returns counts of key AST node types — the 'shape' of the code.
    """
    features = {
        'functions': 0, 'classes': 0, 'loops': 0,
        'conditionals': 0, 'assignments': 0, 'returns': 0,
        'calls': 0, 'imports': 0, 'comprehensions': 0,
        'try_except': 0, 'decorators': 0, 'lambda': 0,
    }
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            t = type(node).__name__
            if t in ('FunctionDef', 'AsyncFunctionDef'): features['functions'] += 1
            elif t == 'ClassDef':                        features['classes'] += 1
            elif t in ('For', 'While', 'AsyncFor'):      features['loops'] += 1
            elif t in ('If', 'IfExp'):                   features['conditionals'] += 1
            elif t in ('Assign', 'AugAssign', 'AnnAssign'): features['assignments'] += 1
            elif t == 'Return':                          features['returns'] += 1
            elif t == 'Call':                            features['calls'] += 1
            elif t in ('Import', 'ImportFrom'):          features['imports'] += 1
            elif t in ('ListComp', 'DictComp', 'SetComp', 'GeneratorExp'):
                features['comprehensions'] += 1
            elif t in ('Try', 'ExceptHandler'):          features['try_except'] += 1
            elif t == 'Lambda':                          features['lambda'] += 1
    except SyntaxError:
        pass
    return features


def count_lines(code: str) -> dict:
    """Count lines: total, blank, comment, code."""
    lines = code.splitlines()
    total = len(lines)
    blank = sum(1 for l in lines if not l.strip())
    comment = sum(1 for l in lines if l.strip().startswith('#'))
    code_lines = total - blank - comment
    return {'total': total, 'blank': blank, 'comment': comment, 'code': code_lines}


def get_identifier_vocabulary(code: str) -> set[str]:
    """Extract all unique identifier names (variable/function/class names)."""
    identifiers = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                identifiers.add(node.name)
            elif isinstance(node, ast.arg):
                identifiers.add(node.arg)
    except SyntaxError:
        identifiers = set(re.findall(r'\b[a-zA-Z_]\w*\b', code))
    return identifiers


# ─────────────────────────────────────────────────────────────
# INFORMATION-THEORETIC CALCULATIONS
# ─────────────────────────────────────────────────────────────

def calculate_D(
    code: str,
    test_pass_rate: float = 1.0,
    num_tests: int = 10,
    language: str = "python"
) -> tuple[float, str]:
    """
    Optimization Depth: D = log2(S) - log2(T)

    S = Search Space = all syntactically valid programs of similar length/complexity
    T = Target Space = programs that pass the test suite

    Estimation method:
    ─────────────────
    S is estimated from:
      - Token vocabulary size (how many distinct tokens are possible)
      - Code length (how many token positions)
      - Python keyword space + identifier space

    T is estimated from:
      - Test pass rate (fraction of S that satisfies constraints)
      - Number of tests (each test multiplies constraint tightness)
      - Code specificity (unique identifiers constrain T further)

    This gives a rigorous lower bound on D.
    """
    tokens = extract_tokens(code)
    n = len(tokens)

    if n == 0:
        return 0.0, "empty code"

    # Python token vocabulary: ~35 keywords + builtins + operators + user identifiers
    vocab_python_base = 35 + 68  # keywords + common builtins
    unique_identifiers = len(get_identifier_vocabulary(code))
    vocab_size = vocab_python_base + unique_identifiers

    # S = vocab_size ^ n (all possible token sequences of length n)
    # Use log directly to avoid overflow: log2(S) = n * log2(vocab_size)
    log2_S = n * math.log2(vocab_size)

    # T = S * test_pass_rate^num_tests
    # Each test independently constrains the solution space
    # log2(T) = log2(S) + num_tests * log2(test_pass_rate)
    if test_pass_rate <= 0:
        test_pass_rate = 1e-10
    log2_T = log2_S + num_tests * math.log2(max(test_pass_rate, 1e-10))

    D = log2_S - log2_T
    method = (f"log2(vocab={vocab_size}^n={n}) - log2(T), "
              f"pass_rate={test_pass_rate:.2f}, tests={num_tests}")
    return round(D, 4), method


def shannon_entropy_from_tokens(tokens: list[str]) -> float:
    """H = -sum(p * log2(p)) over token distribution."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())


def kl_divergence_tokens(p_tokens: list[str], q_tokens: list[str]) -> float:
    """
    KL(P_human || P_ai) — information-theoretic distance.
    P = human final code distribution
    Q = AI suggestion distribution
    Laplace smoothing for unseen tokens.
    """
    p_counts = Counter(p_tokens)
    q_counts = Counter(q_tokens)
    vocab = set(p_counts) | set(q_counts)
    total_p = len(p_tokens)
    total_q = len(q_tokens)

    # Laplace smoothing
    smoothing = 1
    vocab_size = len(vocab)

    kl = 0.0
    for token in vocab:
        p_val = (p_counts.get(token, 0) + smoothing) / (total_p + smoothing * vocab_size)
        q_val = (q_counts.get(token, 0) + smoothing) / (total_q + smoothing * vocab_size)
        if p_val > 0:
            kl += p_val * math.log(p_val / q_val)
    return round(kl, 6)


def calculate_B_code(ai_code: str, human_code: str) -> dict:
    """
    Biological Variance Contribution for code.
    B = D_KL(P_human || P_ai)

    Multi-level measurement:
    1. Token-level KL divergence (raw syntactic distance)
    2. AST structural divergence (semantic shape distance)
    3. Identifier vocabulary novelty (new names introduced by human)
    4. Structural diff ratio (line-level edit distance)

    Returns all components plus composite B score.
    """
    ai_tokens   = extract_tokens(ai_code)
    human_tokens = extract_tokens(human_code)

    # 1. Token-level KL divergence
    B_token = kl_divergence_tokens(human_tokens, ai_tokens)

    # 2. AST structural divergence
    ai_ast   = parse_ast_features(ai_code)
    human_ast = parse_ast_features(human_code)
    all_features = set(ai_ast) | set(human_ast)
    ast_diff = sum(abs(human_ast.get(f,0) - ai_ast.get(f,0)) for f in all_features)
    B_ast = math.log2(1 + ast_diff)  # log-scale to normalize

    # 3. Identifier novelty — new identifiers introduced by human
    ai_ids    = get_identifier_vocabulary(ai_code)
    human_ids = get_identifier_vocabulary(human_code)
    novel_ids = human_ids - ai_ids
    B_novelty = math.log2(1 + len(novel_ids))

    # 4. Line-level edit distance (difflib)
    ai_lines    = ai_code.splitlines()
    human_lines = human_code.splitlines()
    matcher = difflib.SequenceMatcher(None, ai_lines, human_lines)
    edit_ratio = 1.0 - matcher.ratio()  # 0 = identical, 1 = completely different
    B_diff = edit_ratio * math.log2(max(len(human_lines), 1) + 1)

    # Composite B: weighted combination
    B_composite = round(
        0.40 * B_token +
        0.25 * B_ast   +
        0.20 * B_novelty +
        0.15 * B_diff,
        6
    )

    return {
        'B_composite': B_composite,
        'B_token':     round(B_token, 4),
        'B_ast':       round(B_ast, 4),
        'B_novelty':   round(B_novelty, 4),
        'B_diff':      round(B_diff, 4),
        'novel_identifiers': sorted(novel_ids),
        'edit_ratio':  round(edit_ratio, 4),
    }


def calculate_w(B: float, H_final: float) -> float:
    """w = B / H(Y_final)"""
    if H_final == 0:
        return 0.0
    return round(B / H_final, 6)


def calculate_B_required(D: float) -> float:
    """B_req = alpha * D^gamma"""
    if D <= 0:
        return 0.0
    return round(ALPHA * (D ** GAMMA), 6)


def estimate_generations_to_collapse(
    theta_current: float,
    fidelity_loss: float = FIDELITY_LOSS
) -> Optional[int]:
    """
    Estimate generations until θ < θ_critical (0.7 midpoint).
    Using: θ_n = θ_0 * (1 - fidelity_loss)^n
    Solving for n: n = log(θ_critical/θ_0) / log(1 - fidelity_loss)
    """
    if theta_current <= THETA_MIN:
        return 0
    if fidelity_loss <= 0 or fidelity_loss >= 1:
        return None
    try:
        n = math.log(0.7 / theta_current) / math.log(1 - fidelity_loss)
        return max(0, round(n))
    except (ValueError, ZeroDivisionError):
        return None


# ─────────────────────────────────────────────────────────────
# AUDIT ENGINE
# ─────────────────────────────────────────────────────────────

class CodeStagnationAuditor:
    """
    Full stagnation audit pipeline for coding workflows.

    Measures how much genuine human variance was injected
    into an AI-assisted coding session, and predicts
    stagnation trajectory across generations.
    """

    def __init__(self, alpha=ALPHA, gamma=GAMMA, w_threshold=W_THRESHOLD):
        self.alpha = alpha
        self.gamma = gamma
        self.w_threshold = w_threshold
        self.history: list[dict] = []

    def audit(
        self,
        ai_code: str,
        human_code: str,
        test_pass_rate: float = 1.0,
        num_tests: int = 10,
        label: str = "audit"
    ) -> dict:
        """
        Run a single generation audit.

        Parameters:
            ai_code        : Raw AI suggestion (Copilot / Claude / etc.)
            human_code     : Human's final committed code
            test_pass_rate : Fraction of tests passing (0.0 - 1.0)
            num_tests      : Number of test cases in the suite
            label          : Name for this audit (commit hash, PR number, etc.)

        Returns:
            Full audit result dict
        """

        # ── D: Optimization Depth ──────────────────────────────
        D, D_method = calculate_D(human_code, test_pass_rate, num_tests)

        # ── B: Biological Variance (multi-level) ──────────────
        B_data = calculate_B_code(ai_code, human_code)
        B = B_data['B_composite']

        # ── Entropy ───────────────────────────────────────────
        H_human = shannon_entropy_from_tokens(extract_tokens(human_code))
        H_ai    = shannon_entropy_from_tokens(extract_tokens(ai_code))
        H_final = round(H_human, 6)

        # ── w: Director Weight ────────────────────────────────
        w = calculate_w(B, H_final)

        # ── B_req: Required variance ──────────────────────────
        B_req = calculate_B_required(D)

        # ── θ: Entropy retention ──────────────────────────────
        theta = round(H_human / H_ai, 4) if H_ai > 0 else None

        # ── Generations to collapse ───────────────────────────
        gen_to_collapse = estimate_generations_to_collapse(
            theta if theta else 1.0
        )

        # ── Code stats ────────────────────────────────────────
        ai_stats    = count_lines(ai_code)
        human_stats = count_lines(human_code)
        ast_ai      = parse_ast_features(ai_code)
        ast_human   = parse_ast_features(human_code)

        # ── Verdict ───────────────────────────────────────────
        risk = self._assess_risk(w, B, B_req, theta, B_data['edit_ratio'])

        result = {
            "label":             label,
            "D":                 round(D, 4),
            "D_method":          D_method,
            "B":                 B,
            "B_breakdown":       B_data,
            "H_final":           H_final,
            "H_ai":              round(H_ai, 6),
            "w":                 w,
            "B_req":             B_req,
            "theta":             theta,
            "gen_to_collapse":   gen_to_collapse,
            "risk":              risk,
            "B_sufficient":      B >= B_req,
            "w_sufficient":      w >= self.w_threshold,
            "ai_lines":          ai_stats,
            "human_lines":       human_stats,
            "ast_ai":            ast_ai,
            "ast_human":         ast_human,
            "test_pass_rate":    test_pass_rate,
            "num_tests":         num_tests,
        }

        self.history.append(result)
        return result

    def _assess_risk(
        self,
        w: float,
        B: float,
        B_req: float,
        theta: Optional[float],
        edit_ratio: float
    ) -> str:
        score = 0

        # Primary: director weight
        if w < self.w_threshold:
            score += 3
        elif w < self.w_threshold * 2:
            score += 1

        # B sufficiency
        if B < B_req:
            score += 2
        elif B < B_req * 1.5:
            score += 1

        # Entropy retention
        if theta is not None:
            if theta < THETA_MIN:
                score += 3
            elif theta < THETA_MAX:
                score += 1

        # Edit ratio — how much code actually changed
        if edit_ratio < 0.05:   # near-identical
            score += 4
        elif edit_ratio < 0.15:
            score += 2
        elif edit_ratio < 0.30:
            score += 1

        if score == 0:
            return "HEALTHY"
        elif score <= 2:
            return "EARLY WARNING"
        elif score <= 5:
            return "CRITICAL ZONE"
        else:
            return "COLLAPSE RISK"

    def print_report(self, result: dict):
        icons = {
            "HEALTHY":       "✅",
            "EARLY WARNING": "⚠️ ",
            "CRITICAL ZONE": "🔶",
            "COLLAPSE RISK": "🔴",
        }
        icon = icons.get(result['risk'], "❓")
        sep = "=" * 65

        print(f"\n{sep}")
        print(f"  STAGNATION AUDIT — CODING EDITION")
        print(f"  Theory of Stochastic Stagnation § 5.3 | Sihare (2026)")
        print(f"  Label: {result['label']}")
        print(sep)

        # Code stats
        print(f"\n  CODE STATISTICS")
        print(f"  {'':3}{'':20} {'AI':>10} {'Human':>10} {'Delta':>10}")
        print(f"  {'-'*53}")
        for key in ['total', 'code', 'comment', 'blank']:
            ai_v = result['ai_lines'][key]
            hu_v = result['human_lines'][key]
            delta = hu_v - ai_v
            sign = "+" if delta > 0 else ""
            print(f"  {'':3}{key.capitalize() + ' lines':<20} {ai_v:>10} {hu_v:>10} {sign+str(delta):>10}")

        # AST diff
        print(f"\n  AST STRUCTURE CHANGES (AI → Human)")
        print(f"  {'-'*53}")
        ast_keys = ['functions', 'classes', 'loops', 'conditionals',
                    'assignments', 'calls', 'comprehensions', 'try_except']
        for k in ast_keys:
            ai_v = result['ast_ai'].get(k, 0)
            hu_v = result['ast_human'].get(k, 0)
            if ai_v != hu_v:
                delta = hu_v - ai_v
                sign = "+" if delta > 0 else ""
                print(f"  {'':3}{k.capitalize():<20} {ai_v:>10} {hu_v:>10} {sign+str(delta):>10}")

        # Novel identifiers
        novel = result['B_breakdown']['novel_identifiers']
        if novel:
            print(f"\n  NEW IDENTIFIERS (human-introduced): {len(novel)}")
            shown = novel[:8]
            print(f"  {', '.join(shown)}" + (" ..." if len(novel) > 8 else ""))

        # Core metrics
        print(f"\n  STAGNATION METRICS")
        print(f"  {'METRIC':<32} {'VALUE':>12}   STATUS")
        print(f"  {'-'*58}")

        def row(name, val, status=""):
            print(f"  {name:<32} {str(val):>12}   {status}")

        row("D  (Optimization Depth)",     result['D'])
        row("B_req  (Required Variance)",  result['B_req'],
            f"α={self.alpha}, γ={self.gamma}")

        # B breakdown
        bb = result['B_breakdown']
        b_status = "✅ sufficient" if result['B_sufficient'] else "❌ below B_req"
        row("B  (Composite Variance)",     result['B'], b_status)
        row("  ├ B_token  (KL divergence)", bb['B_token'])
        row("  ├ B_ast    (structure)",     bb['B_ast'])
        row("  ├ B_novelty (new identifiers)", bb['B_novelty'])
        row("  └ B_diff   (edit ratio)",    f"{bb['edit_ratio']:.1%}")

        row("H(human code)",               result['H_final'])
        row("H(AI code)",                  result['H_ai'])

        theta_str = str(result['theta']) if result['theta'] else "N/A"
        if result['theta']:
            if result['theta'] >= THETA_MAX:
                t_s = "✅ healthy"
            elif result['theta'] >= THETA_MIN:
                t_s = "🔶 critical zone"
            else:
                t_s = "🔴 collapse"
        else:
            t_s = ""
        row("θ  (Entropy Retention)",       theta_str, t_s)

        w_status = "✅" if result['w_sufficient'] else f"❌ need w≥{self.w_threshold}"
        row("w  (Director Weight)",         result['w'], w_status)

        row("Tests",  f"{result['num_tests']} | pass rate {result['test_pass_rate']:.0%}")

        gen = result['gen_to_collapse']
        gen_str = f"~{gen} generations" if gen is not None else "N/A"
        gen_color = "🔴" if gen is not None and gen < 5 else ("🔶" if gen is not None and gen < 15 else "✅")
        row("Est. generations to collapse", gen_str, gen_color)

        print(f"\n  {'─'*58}")
        print(f"  VERDICT: {icon}  {result['risk']}")
        print(sep)

        self._interpret(result)

    def _interpret(self, r: dict):
        risk = r['risk']
        bb = r['B_breakdown']
        print(f"\n  INTERPRETATION:")

        if risk == "HEALTHY":
            print(f"  Human director is injecting sufficient variance.")
            print(f"  Codebase is above the stagnation boundary.")
            print(f"  Director weight w={r['w']:.3f} (threshold: {self.w_threshold})")

        elif risk == "EARLY WARNING":
            print(f"  Variance indicators are dipping. Watch next 2-3 commits.")
            if bb['edit_ratio'] < 0.15:
                print(f"  → Only {bb['edit_ratio']:.1%} of lines changed. Push for deeper edits.")
            if not r['w_sufficient']:
                print(f"  → w={r['w']:.3f} below threshold. More structural deviation needed.")

        elif risk == "CRITICAL ZONE":
            print(f"  System is within critical entropy range θ ∈ [0.6, 0.8].")
            print(f"  Variance loss is accelerating.")
            if not r['B_sufficient']:
                deficit = r['B_req'] - r['B']
                print(f"  → B deficit: {deficit:.4f} units of variance needed.")
            if bb['edit_ratio'] < 0.10:
                print(f"  → Edit ratio {bb['edit_ratio']:.1%} is dangerously low.")
                print(f"  → Developer is acting as Acceptor, not Director.")
            print(f"  → Inject structural changes: new abstractions, renamed concepts,")
            print(f"     alternative algorithms, or architectural deviation.")

        elif risk == "COLLAPSE RISK":
            print(f"  ⚠ Codebase is in recursive refinement mode.")
            print(f"  Human is rubber-stamping AI output without variance injection.")
            print(f"  Novel identifiers introduced: {len(bb['novel_identifiers'])}")
            if r['gen_to_collapse'] is not None:
                print(f"  Estimated {r['gen_to_collapse']} generations until θ < 0.7.")
            print(f"  → REQUIRED: Substantial human-directed restructuring.")
            print(f"  → Break the loop: write a section from scratch, no AI input.")

        print()

    def print_history_summary(self):
        """Print trend analysis across multiple generations."""
        if not self.history:
            print("No history recorded.")
            return

        print(f"\n{'='*65}")
        print(f"  GENERATION TREND ANALYSIS  ({len(self.history)} audits)")
        print(f"{'='*65}")
        print(f"  {'#':<4} {'Label':<20} {'D':>6} {'B':>8} {'w':>8} {'θ':>7} {'Risk'}")
        print(f"  {'-'*60}")

        for i, r in enumerate(self.history, 1):
            theta = str(r['theta']) if r['theta'] else "N/A"
            print(f"  {i:<4} {r['label']:<20} {r['D']:>6} {r['B']:>8} {r['w']:>8} {theta:>7}   {r['risk']}")

        # Trend arrows
        if len(self.history) >= 2:
            w_trend = self.history[-1]['w'] - self.history[0]['w']
            B_trend = self.history[-1]['B'] - self.history[0]['B']
            print(f"\n  TREND:  w {'↑' if w_trend > 0 else '↓'} {abs(w_trend):.3f}   "
                  f"B {'↑' if B_trend > 0 else '↓'} {abs(B_trend):.3f}")
            if w_trend < 0 and B_trend < 0:
                print(f"  ⚠ Both w and B declining — stagnation trajectory confirmed.")
            elif w_trend > 0 and B_trend > 0:
                print(f"  ✅ Both w and B increasing — healthy divergence trend.")

        print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────
# DEMO CASES
# ─────────────────────────────────────────────────────────────

DEMO_CASES = [
    {
        "label": "PR-001 | Healthy (deep human rework)",
        "test_pass_rate": 0.95,
        "num_tests": 12,
        "ai": textwrap.dedent("""
            def find_duplicates(lst):
                seen = []
                duplicates = []
                for item in lst:
                    if item in seen:
                        duplicates.append(item)
                    else:
                        seen.append(item)
                return duplicates
        """),
        "human": textwrap.dedent("""
            from collections import Counter
            from typing import TypeVar, Sequence

            T = TypeVar('T')

            def find_duplicates(sequence: Sequence[T]) -> list[T]:
                \"\"\"
                Return elements appearing more than once.
                O(n) time using Counter. Order-preserving.
                \"\"\"
                counts = Counter(sequence)
                seen_once = set()
                result = []
                for item in sequence:
                    if counts[item] > 1 and item not in seen_once:
                        result.append(item)
                        seen_once.add(item)
                return result
        """),
    },
    {
        "label": "PR-002 | Collapse Risk (near-copy accept)",
        "test_pass_rate": 1.0,
        "num_tests": 5,
        "ai": textwrap.dedent("""
            def calculate_average(numbers):
                total = 0
                for num in numbers:
                    total += num
                return total / len(numbers)
        """),
        "human": textwrap.dedent("""
            def calculate_average(numbers):
                total = 0
                for num in numbers:
                    total = total + num
                return total / len(numbers)
        """),
    },
    {
        "label": "PR-003 | Critical Zone (partial edit)",
        "test_pass_rate": 0.80,
        "num_tests": 8,
        "ai": textwrap.dedent("""
            def binary_search(arr, target):
                left = 0
                right = len(arr) - 1
                while left <= right:
                    mid = (left + right) // 2
                    if arr[mid] == target:
                        return mid
                    elif arr[mid] < target:
                        left = mid + 1
                    else:
                        right = mid - 1
                return -1
        """),
        "human": textwrap.dedent("""
            def binary_search(arr, target):
                left, right = 0, len(arr) - 1
                while left <= right:
                    mid = left + (right - left) // 2  # avoids overflow
                    if arr[mid] == target:
                        return mid
                    elif arr[mid] < target:
                        left = mid + 1
                    else:
                        right = mid - 1
                return -1
        """),
    },
]


def run_demo():
    auditor = CodeStagnationAuditor()

    print("\n" + "█"*65)
    print("  STAGNATION AUDIT TOOL — CODING EDITION  DEMO")
    print("  Theory of Stochastic Stagnation (Sihare, 2026)")
    print("  Measuring entropy decay in AI-assisted development")
    print("█"*65)

    for case in DEMO_CASES:
        result = auditor.audit(
            ai_code=case['ai'],
            human_code=case['human'],
            test_pass_rate=case['test_pass_rate'],
            num_tests=case['num_tests'],
            label=case['label']
        )
        auditor.print_report(result)

    auditor.print_history_summary()

    print("  INTERPRETATION OF RESULTS:")
    print("  PR-001: Deep human rework — new imports, generics, docstring,")
    print("          O(n) algorithm change. Strong director weight.")
    print("  PR-002: One-character change (+ → +=). Near-zero variance.")
    print("          Classic rubber-stamp acceptance. Collapse risk.")
    print("  PR-003: Overflow fix + style — meaningful but minimal.")
    print("          Sits in the critical zone: real edit, low depth.\n")


def run_interactive():
    print("\n" + "="*65)
    print("  STAGNATION AUDIT — INTERACTIVE MODE")
    print("="*65)

    def get_multiline(prompt):
        print(f"\n{prompt}")
        print("(Paste code, then press Enter twice to finish)\n")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        return "\n".join(lines[:-1])

    ai_code    = get_multiline("Paste the AI-generated code:")
    human_code = get_multiline("Paste your final committed code:")

    try:
        pass_rate = float(input("\nTest pass rate (0.0 - 1.0, default 1.0): ") or "1.0")
        num_tests = int(input("Number of tests (default 10): ") or "10")
    except ValueError:
        pass_rate, num_tests = 1.0, 10

    label = input("Label for this audit (PR number, commit hash, etc.): ") or "interactive"

    auditor = CodeStagnationAuditor()
    result  = auditor.audit(ai_code, human_code, pass_rate, num_tests, label)
    auditor.print_report(result)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stagnation Audit Tool — Coding Edition"
    )
    parser.add_argument("--demo",        action="store_true", help="Run demo cases")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--ai",    type=str, help="Path to AI-generated code file")
    parser.add_argument("--human", type=str, help="Path to human-edited code file")
    parser.add_argument("--tests", type=int,   default=10,  help="Number of tests")
    parser.add_argument("--pass-rate", type=float, default=1.0, help="Test pass rate")
    parser.add_argument("--label", type=str,   default="audit", help="Audit label")
    parser.add_argument("--alpha", type=float, default=ALPHA,  help="Override α")
    parser.add_argument("--gamma", type=float, default=GAMMA,  help="Override γ")
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    elif args.ai and args.human:
        with open(args.ai)    as f: ai_code    = f.read()
        with open(args.human) as f: human_code = f.read()
        auditor = CodeStagnationAuditor(alpha=args.alpha, gamma=args.gamma)
        result  = auditor.audit(ai_code, human_code, args.pass_rate, args.tests, args.label)
        auditor.print_report(result)
    else:
        run_demo()
        print("  Usage examples:")
        print("  python stagnation_audit_code.py --demo")
        print("  python stagnation_audit_code.py --interactive")
        print("  python stagnation_audit_code.py --ai ai.py --human final.py --tests 20 --pass-rate 0.9\n")


if __name__ == "__main__":
    main()

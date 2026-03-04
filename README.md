# Theory of Stochastic Stagnation
**A framework for measuring entropy decay in AI-assisted systems**

> *"Intelligence oriented exclusively toward error minimization risks stochastic stagnation.
> The future of intelligence is symbiotic."*
> — Chaitanya Sihare, 2026

**Paper:** [OSF DOI: 10.17605/OSF.IO/FSMK9](https://osf.io/fsmk9)
**Author:** Chaitanya Sihare | ORCID: [0009-0000-0340-2492](https://orcid.org/0009-0000-0340-2492)
**Affiliation:** PG College Seoni, Madhya Pradesh, India

---

## What is this?

Most AI detectors are trained classifiers. They learn "AI output looks like this" from labeled datasets. When models update, they go blind. They can't explain their verdicts.

This project takes a different approach. It is grounded in **information theory** — Shannon entropy, KL divergence, compressibility — signals that measure something fundamental about content regardless of which model generated it.

The core claim:

> AI systems systematically narrow the entropy distribution of their outputs through optimization. This is measurable, predictable, and has consequences for anyone using AI-assisted workflows.

This is not just a detector. It is a **framework** for understanding human-AI collaboration mathematically.

---

## The Theory

When AI systems train recursively on their own outputs, they eliminate low-probability tail events — the rare, creative, unexpected tokens that drive innovation. The system enters **Stochastic Stagnation**: a state of permanent convergence toward narrowed distributions.

**Key quantities (Section 5.3 of the paper):**

| Symbol | Name | Formula |
|--------|------|---------|
| D | Optimization Depth | log₂(S) − log₂(T) |
| B | Biological Variance | D_KL(P_human ‖ P_ai) |
| w | Director Weight | B / H(Y_final) |
| B_req | Required Variance | α · D^γ |
| θ | Entropy Retention | H(Yₙ) / H(Y₀) |

**Stagnation audit rule:** If w < 0.15, the system is trending toward collapse.

**Law of Stagnation:** B_req ≥ 0.5 · D^1.3

The relationship is **superlinear** (γ > 1) — confirmed both theoretically and by simulation (empirical fit: α ≈ 1.04, γ ≈ 2.64). The exact parameters are domain-dependent; the superlinear structure is stable.

---

## Tools

### 1. `stagnation_audit.py` — Text Stagnation Auditor
Measures how much genuine human variance exists in an AI-assisted text workflow.

```bash
python stagnation_audit.py
python stagnation_audit.py --interactive
```

Outputs: D, B, w, B_req, θ, and a stagnation verdict (HEALTHY / EARLY WARNING / CRITICAL ZONE / COLLAPSE RISK).

---

### 2. `stagnation_audit_code.py` — Code Stagnation Auditor
Designed for software development workflows. Measures entropy decay in AI-assisted coding (Copilot, Cursor, Claude, ChatGPT).

```bash
python stagnation_audit_code.py --demo
python stagnation_audit_code.py --ai ai_suggestion.py --human final_commit.py --tests 20 --pass-rate 0.9
```

Why coding works precisely:
- **S** = all syntactically valid programs of length n
- **T** = programs passing the test suite (binary, measurable)
- **B** = multi-level: token KL divergence + AST structural distance + identifier novelty + edit ratio
- **Generations** = commits / PRs (discrete, trackable)

---

### 3. `ai_detector.py` — Universal AI Content Detector
Analyzes any file type using entropy-based signals. No training data required.

```bash
python ai_detector.py --demo
python ai_detector.py --file mycode.py
python ai_detector.py --file photo.png
python ai_detector.py --dir ./my_project/
python ai_detector.py --file document.txt --json
```

**Supported file types:**
- Source code (Python, JS, Java, C, Go, Rust, ...)
- Text / Markdown / documents
- Images (PNG, JPEG, BMP, WebP)
- Any binary file (byte-level analysis)

**Signals used:**

| Signal | What it measures |
|--------|-----------------|
| Byte entropy | Raw information density |
| Compressibility | Predictability of content |
| Burstiness | Inter-chunk entropy variance |
| Identifier entropy | Vocabulary richness of code |
| Generic name ratio | AI naming patterns (result, data, helper...) |
| AST complexity variance | Structural regularity of code |
| Spatial entropy std | Block-level image uniformity |
| Channel correlation | RGB channel independence |
| Noise entropy | Camera sensor noise vs AI smoothness |

**Real results so far:**
- Real phone photo (gym equipment) → 68% Human ✅
- AI-generated infographic with human direction → 58% Ambiguous ✅ (correctly uncertain)
- AI code with specific human prompt → 61% Human ✅ (theory confirmed — human direction raises score)
- AI code with generic prompt → 38% AI ✅

---

### 4. React App (`ai_detector_final.jsx`)
Browser-based demo. No installation needed. Two-stage detection: entropy signals + semantic analysis.

Paste any Python code → instant verdict with full signal breakdown and stagnation layer.

---

## Why this approach is different

| Feature | Trained Classifiers (GPTZero, Turnitin) | This Framework |
|---------|----------------------------------------|----------------|
| Explainability | Black box | Every signal explained |
| Model dependency | Blind to new models until retrained | Model-agnostic |
| File types | Text only | Code, text, images, binary |
| Theoretical basis | Empirical pattern matching | Information theory |
| Human contribution | Not measured | Directly measured (w score) |
| Stagnation tracking | Not available | Core feature |

---

## Installation

```bash
pip install pillow numpy scipy
```

No other dependencies beyond Python standard library.

---

## Validation Status

| Component | Status |
|-----------|--------|
| Theory published | ✅ OSF preprint with DOI |
| Simulation (superlinear γ>1) | ✅ Confirmed |
| Text stagnation audit | ✅ Built and tested |
| Code stagnation audit | ✅ Built and tested |
| Universal detector | ✅ Built and tested |
| Image validation study | 🔄 In progress |
| Code validation study | 🔄 In progress |
| Peer review | 🔄 Not yet submitted |

This is honest research in progress. The framework is theoretically grounded. Empirical validation is ongoing.

---

## The key insight this framework adds

When a developer accepts an AI suggestion without editing it, their **Director Weight w approaches zero**. Over many commits, the codebase entropy collapses. The team loses the capacity to generate genuinely novel solutions — not because they forgot how, but because they stopped practicing the cognitive moves that generate novelty.

This is measurable. It is now measured.

---

## Citation

```bibtex
@misc{sihare2026stochastic,
  title={Theory of Stochastic Stagnation},
  author={Sihare, Chaitanya},
  year={2026},
  doi={10.17605/OSF.IO/FSMK9},
  url={https://osf.io/fsmk9}
}
```

---

## Contact

Chaitanya Sihare
PG College Seoni, Madhya Pradesh, India
ORCID: 0009-0000-0340-2492
OSF: https://osf.io/fsmk9

---

*Validation study in progress. Contributions, critiques, and collaborations welcome.*

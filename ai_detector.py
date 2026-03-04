"""
UNIVERSAL AI CONTENT DETECTOR
Based on: Theory of Stochastic Stagnation (Sihare, 2026)

Every file is information. Entropy reveals origin.

AI content sits in a NARROW entropy band - too clean, too regular, too converged.
Human content is messier: domain-specific, bursty, personally shaped.

SUPPORTED: Python/JS/any code | Text/Markdown | Images (PNG/JPG) | Any binary

USAGE:
  python ai_detector.py --demo
  python ai_detector.py --file mycode.py
  python ai_detector.py --file photo.png
  python ai_detector.py --dir ./project/
  python ai_detector.py --file report.txt --json

INSTALL: pip install pillow --break-system-packages
"""

import ast, io, json, math, os, re, zlib, argparse
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
    HAS_IMAGE = True
except ImportError:
    HAS_IMAGE = False

# ── Constants ──────────────────────────────────────────────────────────────────
CODE_EXT  = {'.py','.js','.ts','.java','.c','.cpp','.cs','.go','.rb','.rs','.php'}
TEXT_EXT  = {'.txt','.md','.rst','.html','.xml','.json','.yaml','.yml','.csv'}
IMAGE_EXT = {'.png','.jpg','.jpeg','.bmp','.webp','.tiff','.gif'}

# ── Universal byte signals ─────────────────────────────────────────────────────

def byte_entropy(data: bytes) -> float:
    if not data: return 0.0
    counts = Counter(data); total = len(data)
    return -sum((c/total)*math.log2(c/total) for c in counts.values())

def compress_ratio(data: bytes) -> float:
    if len(data) < 16: return 1.0
    return len(zlib.compress(data, level=9)) / len(data)

def burstiness(data: bytes, chunk=256) -> float:
    if len(data) < chunk*2: return 0.5
    chunks = [data[i:i+chunk] for i in range(0, len(data)-chunk, chunk)]
    ents = [byte_entropy(c) for c in chunks if c]
    if len(ents) < 2: return 0.5
    m = sum(ents)/len(ents)
    std = (sum((e-m)**2 for e in ents)/len(ents))**0.5
    return round(min(1.0, std/0.5), 4)

def score_entropy(h: float) -> float:
    if h < 3.0: return 0.10
    if h < 5.0: return 0.40
    if 6.2 <= h <= 7.6: return 0.25   # AI cluster zone
    if h > 7.8: return 0.70
    return 0.60

def score_compress(r: float) -> float:
    if r < 0.30: return 0.10
    if r < 0.65: return 0.25
    if r < 0.80: return 0.55
    return 0.75

# ── Code signals ───────────────────────────────────────────────────────────────

AI_NAMES = {
    'result','results','data','value','values','temp','tmp','item','items',
    'element','helper','utils','output','input','param','params','args','kwargs',
    'obj','node','key','val','num','count','total','index','idx','lst','arr',
    'response','request','handler','manager','processor','execute','compute'
}

def analyze_code(src: str, ext: str = '.py') -> dict:
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b', src)
    s = {}

    # Identifier entropy
    if words:
        wc = Counter(words); n = len(words)
        ie = -sum((c/n)*math.log2(c/n) for c in wc.values())
        s['identifier_entropy'] = round(ie, 4)
        s['id_entropy_score']   = min(1.0, ie/6.0)
    else:
        s['identifier_entropy'] = 0.0
        s['id_entropy_score']   = 0.5

    # Generic name ratio
    id_set = set(w.lower() for w in words)
    gr = len(id_set & AI_NAMES) / max(len(id_set), 1)
    s['generic_ratio'] = round(gr, 4)
    s['generic_score'] = max(0.0, 1.0 - gr*2)

    # AST complexity variance (Python only)
    if ext == '.py':
        try:
            tree = ast.parse(src)
            cxs = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cx = sum(1 for n in ast.walk(node)
                             if isinstance(n,(ast.If,ast.For,ast.While,ast.Try,ast.ExceptHandler)))
                    cxs.append(cx)
            if len(cxs) >= 2:
                mc = sum(cxs)/len(cxs)
                std = (sum((c-mc)**2 for c in cxs)/len(cxs))**0.5
                s['complexity_variance'] = round(std, 4)
                s['ast_score']           = min(1.0, std/3.0)
            else:
                s['complexity_variance'] = 0.0
                s['ast_score']           = 0.4
        except SyntaxError:
            s['complexity_variance'] = 0.0
            s['ast_score']           = 0.5
    else:
        s['complexity_variance'] = 0.0
        s['ast_score']           = 0.5

    # Line length variance
    lines = [l for l in src.splitlines() if l.strip()]
    if len(lines) >= 3:
        ll = [len(l) for l in lines]; ml = sum(ll)/len(ll)
        std = (sum((l-ml)**2 for l in ll)/len(ll))**0.5
        s['line_variance']       = round(std, 2)
        s['line_variance_score'] = min(1.0, std/25.0)
    else:
        s['line_variance']       = 0.0
        s['line_variance_score'] = 0.5

    s['type_specific_score'] = round(
        s['id_entropy_score']*0.35 + s['generic_score']*0.30 +
        s['ast_score']*0.20 + s['line_variance_score']*0.15, 4)
    return s

# ── Text signals ───────────────────────────────────────────────────────────────

AI_PHRASES = [
    'furthermore','moreover','additionally','in conclusion','to summarize',
    'it is important','it should be noted','in other words','as a result',
    'therefore','consequently','in summary','overall','leverage','utilize',
    'facilitate','optimize','streamline','robust','scalable','delve','crucial',
    'certainly','absolutely','it is worth noting','it is essential'
]

def analyze_text(text: str) -> dict:
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    s = {}

    ttr = len(set(words))/len(words) if words else 0.0
    s['ttr']       = round(ttr, 4)
    s['ttr_score'] = min(1.0, ttr*2.0)

    if len(sents) >= 3:
        sl = [len(s_.split()) for s_ in sents]; msl = sum(sl)/len(sl)
        std = (sum((l-msl)**2 for l in sl)/len(sl))**0.5
        s['sent_length_std']   = round(std, 2)
        s['sent_variance_score'] = min(1.0, std/12.0)
    else:
        s['sent_length_std']   = 0.0
        s['sent_variance_score'] = 0.5

    tl = text.lower()
    hits = sum(1 for p in AI_PHRASES if p in tl)
    density = hits / max(len(words)/100, 1)
    s['ai_phrase_density'] = round(density, 4)
    s['phrase_score']      = max(0.0, 1.0 - density*0.3)

    punct = [c for c in text if c in '.,;:!?-()[]{}"\'/']
    if punct:
        pc = Counter(punct); pt = len(punct)
        pe = -sum((c/pt)*math.log2(c/pt) for c in pc.values())
        s['punct_entropy'] = round(pe, 4)
        s['punct_score']   = min(1.0, pe/3.0)
    else:
        s['punct_entropy'] = 0.0
        s['punct_score']   = 0.5

    s['type_specific_score'] = round(
        s['ttr_score']*0.30 + s['sent_variance_score']*0.25 +
        s['phrase_score']*0.30 + s['punct_score']*0.15, 4)
    return s

# ── Image signals ──────────────────────────────────────────────────────────────

def ch_ent(vals: list) -> float:
    c = Counter(vals); t = len(vals)
    return -sum((v/t)*math.log2(v/t) for v in c.values()) if t else 0.0

def corr(x: list, y: list) -> float:
    if len(x) < 2: return 0.0
    mx, my = sum(x)/len(x), sum(y)/len(y)
    num = sum((a-mx)*(b-my) for a,b in zip(x,y))
    dx = sum((a-mx)**2 for a in x)**0.5
    dy = sum((b-my)**2 for b in y)**0.5
    return num/(dx*dy) if dx*dy > 0 else 0.0

def analyze_image(data: bytes) -> dict:
    s = {}
    if not HAS_IMAGE:
        s['type_specific_score'] = 0.5
        s['note'] = 'pip install pillow to enable image analysis'
        return s
    try:
        img = Image.open(io.BytesIO(data)).convert('RGB')
        w, h = img.size
        s['dimensions'] = f"{w}x{h}"
        pixels = list(img.getdata())
        rv = [p[0] for p in pixels]
        gv = [p[1] for p in pixels]
        bv = [p[2] for p in pixels]

        # Channel entropy
        mce = (ch_ent(rv)+ch_ent(gv)+ch_ent(bv))/3
        s['channel_entropy']       = round(mce, 4)
        s['channel_entropy_score'] = 0.3 if mce > 7.5 else (0.5 if mce > 6.5 else 0.7)

        # Channel correlation
        step = max(1, len(rv)//10000)
        rs,gs,bs = rv[::step][:10000], gv[::step][:10000], bv[::step][:10000]
        mc = (abs(corr(rs,gs))+abs(corr(rs,bs))+abs(corr(gs,bs)))/3
        s['channel_correlation']  = round(mc, 4)
        s['correlation_score']    = max(0.1, 1.0 - mc*0.8)

        # Spatial block entropy variance
        bsz = max(32, min(w,h)//8)
        gray = list(img.convert('L').getdata())
        block_ents = []
        for row in range(0, h-bsz, bsz):
            for col in range(0, w-bsz, bsz):
                block = [gray[(row+r)*w+(col+c)]
                         for r in range(bsz) for c in range(bsz)
                         if (row+r)*w+(col+c) < len(gray)]
                if block: block_ents.append(ch_ent(block))
        if len(block_ents) >= 4:
            mbe = sum(block_ents)/len(block_ents)
            std = (sum((e-mbe)**2 for e in block_ents)/len(block_ents))**0.5
            s['spatial_entropy_std']   = round(std, 4)
            s['spatial_variance_score'] = min(1.0, std/1.5)
        else:
            s['spatial_entropy_std']   = 0.0
            s['spatial_variance_score'] = 0.5

        # Noise floor
        diffs = [abs(int(gray[i])-int(gray[i-1])) for i in range(1, min(len(gray),50000))]
        ne = ch_ent(diffs) if diffs else 0.0
        s['noise_entropy'] = round(ne, 4)
        s['noise_score']   = 0.75 if ne > 5.0 else (0.5 if ne > 3.5 else 0.2)

        # Dimension regularity (AI images often use standard sizes)
        AI_DIMS = {256,384,512,640,768,1024,1280,1536,1920,2048}
        s['dimension_score'] = 0.3 if (w in AI_DIMS and h in AI_DIMS) else 0.7

        s['type_specific_score'] = round(
            s['channel_entropy_score']*0.20 + s['correlation_score']*0.25 +
            s['spatial_variance_score']*0.30 + s['noise_score']*0.20 +
            s['dimension_score']*0.05, 4)

    except Exception as e:
        s['type_specific_score'] = 0.5
        s['error'] = str(e)
    return s

# ── Detector ───────────────────────────────────────────────────────────────────

class AIContentDetector:

    def analyze_file(self, filepath: str) -> dict:
        p = Path(filepath)
        if not p.exists(): return {'error': f'Not found: {filepath}', 'verdict': 'ERROR'}
        return self._analyze(p.read_bytes(), p.suffix.lower(), p.name, str(filepath))

    def analyze_bytes(self, data: bytes, ext: str = '', filename: str = '') -> dict:
        return self._analyze(data, ext.lower(), filename, filename)

    def _analyze(self, data: bytes, ext: str, filename: str, label: str) -> dict:
        hb    = byte_entropy(data)
        cr    = compress_ratio(data)
        burst = burstiness(data)
        se    = score_entropy(hb)
        sc    = score_compress(cr)

        file_type = 'binary'
        ts = {}
        if ext in CODE_EXT:
            file_type = 'code'
            try: ts = analyze_code(data.decode('utf-8', errors='replace'), ext)
            except Exception as e: ts = {'type_specific_score': 0.5, 'error': str(e)}
        elif ext in TEXT_EXT:
            file_type = 'text'
            try: ts = analyze_text(data.decode('utf-8', errors='replace'))
            except Exception as e: ts = {'type_specific_score': 0.5, 'error': str(e)}
        elif ext in IMAGE_EXT:
            file_type = 'image'
            ts = analyze_image(data)
        else:
            ts = {'type_specific_score': 0.5}

        tss   = ts.get('type_specific_score', 0.5)
        final = round(se*0.20 + sc*0.20 + burst*0.20 + tss*0.40, 4)
        verdict, confidence = self._verdict(final)

        return {
            'label': label, 'file_type': file_type, 'extension': ext,
            'file_size': len(data),
            'byte_entropy': round(hb,4), 'compress_ratio': round(cr,4),
            'burstiness': round(burst,4),
            'score_entropy': round(se,4), 'score_compress': round(sc,4),
            'score_burstiness': round(burst,4), 'score_type': round(tss,4),
            'type_signals': ts,
            'human_score': final, 'ai_score': round(1.0-final, 4),
            'verdict': verdict, 'confidence': confidence
        }

    def _verdict(self, s: float):
        if s >= 0.75: return 'LIKELY HUMAN',        'High'
        if s >= 0.60: return 'PROBABLY HUMAN',      'Medium'
        if s >= 0.45: return 'AMBIGUOUS',           'Low'
        if s >= 0.30: return 'PROBABLY AI',         'Medium'
        return          'LIKELY AI-GENERATED', 'High'

    def print_report(self, result: dict):
        if result.get('verdict') == 'ERROR':
            print(f"  ERROR: {result.get('error')}"); return

        icons = {'LIKELY HUMAN':'[HUMAN]','PROBABLY HUMAN':'[~HUMAN]',
                 'AMBIGUOUS':'[???]','PROBABLY AI':'[~AI]','LIKELY AI-GENERATED':'[AI]'}
        icon = icons.get(result['verdict'], '?')
        sep = '=' * 65

        def bar(v, w=20):
            f = round(v*w)
            return f"[{'#'*f}{'.'*(w-f)}] {v:.2f}"

        print(f"\n{sep}")
        print(f"  AI CONTENT DETECTOR | Stochastic Stagnation Framework (Sihare 2026)")
        print(f"  {Path(result['label']).name}  |  {result['file_type'].upper()}  |  {result['file_size']:,} bytes")
        print(sep)
        print(f"\n  UNIVERSAL SIGNALS  (0.0 = AI  -->  1.0 = Human)")
        print(f"  {'Byte Entropy':<22} raw={result['byte_entropy']:.3f}   {bar(result['score_entropy'])}")
        print(f"  {'Compress Ratio':<22} raw={result['compress_ratio']:.3f}   {bar(result['score_compress'])}")
        print(f"  {'Burstiness':<22} raw={result['burstiness']:.3f}   {bar(result['score_burstiness'])}")
        print(f"  {'Type-Specific':<22}          {bar(result['score_type'])}")

        ts = result['type_signals']
        ft = result['file_type']

        if ft == 'code' and ts:
            print(f"\n  CODE SIGNALS")
            for lbl, rk, sk in [
                ('Identifier Entropy',   'identifier_entropy',   'id_entropy_score'),
                ('Generic Name Ratio',   'generic_ratio',        'generic_score'),
                ('Complexity Variance',  'complexity_variance',  'ast_score'),
                ('Line Length Variance', 'line_variance',        'line_variance_score'),
            ]:
                if rk in ts and sk in ts:
                    print(f"  {lbl:<22} raw={ts[rk]:.3f}   {bar(ts[sk])}")

        elif ft == 'text' and ts:
            print(f"\n  TEXT SIGNALS")
            for lbl, rk, sk in [
                ('Type-Token Ratio',     'ttr',              'ttr_score'),
                ('Sentence Length Std',  'sent_length_std',  'sent_variance_score'),
                ('AI Phrase Density',    'ai_phrase_density','phrase_score'),
                ('Punctuation Entropy',  'punct_entropy',    'punct_score'),
            ]:
                if rk in ts and sk in ts:
                    print(f"  {lbl:<22} raw={ts[rk]:.3f}   {bar(ts[sk])}")

        elif ft == 'image' and ts:
            print(f"\n  IMAGE SIGNALS")
            if 'dimensions' in ts:
                print(f"  Dimensions: {ts['dimensions']}")
            for lbl, rk, sk in [
                ('Channel Entropy',      'channel_entropy',    'channel_entropy_score'),
                ('Channel Correlation',  'channel_correlation','correlation_score'),
                ('Spatial Entropy Std',  'spatial_entropy_std','spatial_variance_score'),
                ('Noise Entropy',        'noise_entropy',      'noise_score'),
            ]:
                if rk in ts and sk in ts:
                    print(f"  {lbl:<22} raw={ts[rk]:.3f}   {bar(ts[sk])}")
            if 'note' in ts: print(f"  Note: {ts['note']}")
            if 'error' in ts: print(f"  Error: {ts['error']}")

        print(f"\n  {'-'*62}")
        print(f"  Human {bar(result['human_score'])}  ({round(result['human_score']*100)}%)")
        print(f"  AI    {bar(result['ai_score'])}  ({round(result['ai_score']*100)}%)")
        print(f"\n  VERDICT ({result['confidence']} confidence): {icon}  {result['verdict']}")
        print(sep)

    def scan_directory(self, dirpath: str) -> list:
        results = []
        all_ext = CODE_EXT | TEXT_EXT | IMAGE_EXT
        for root, _, files in os.walk(dirpath):
            for fname in files:
                if Path(fname).suffix.lower() in all_ext:
                    results.append(self.analyze_file(os.path.join(root, fname)))
        return results

    def print_directory_summary(self, results: list):
        if not results: return
        sep = '=' * 65
        print(f"\n{sep}\n  DIRECTORY SCAN  ({len(results)} files)\n{sep}")
        print(f"  {'File':<35} {'Type':<8} {'Human%':>7}  Verdict")
        print(f"  {'-'*60}")
        for r in sorted(results, key=lambda x: x['human_score']):
            print(f"  {Path(r['label']).name[:33]:<35} {r['file_type']:<8} "
                  f"{round(r['human_score']*100):>6}%  {r['verdict']}")
        avg = sum(r['human_score'] for r in results)/len(results)
        flagged = sum(1 for r in results if r['human_score'] < 0.45)
        print(f"\n  Avg human score: {avg:.2f} ({round(avg*100)}%)")
        print(f"  Files flagged AI: {flagged}/{len(results)}\n{sep}\n")


# ── Demo ───────────────────────────────────────────────────────────────────────

def run_demo():
    d = AIContentDetector()
    print("\n" + "#"*65)
    print("  UNIVERSAL AI CONTENT DETECTOR -- DEMO")
    print("  Theory of Stochastic Stagnation (Sihare, 2026)")
    print("  Every file is information. Entropy reveals origin.")
    print("#"*65)

    AI_CODE = b"""
def calculate_fibonacci(n):
    if n <= 0: return 0
    elif n == 1: return 1
    dp = [0] * (n + 1)
    dp[0] = 0; dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

def process_data(data):
    result = []
    for item in data:
        value = calculate_fibonacci(item)
        result.append(value)
    return result

def main():
    input_data = [5, 10, 15, 20]
    output = process_data(input_data)
    print(f"Result: {output}")

if __name__ == "__main__":
    main()
"""

    HUMAN_CODE = b"""
import sys

def _mat_mul(A, B):
    # 2x2 only, inlined for speed
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
    print(f"even fibs up to fib({N}):", evens, "| sum:", sum(evens))
"""

    AI_TEXT = b"""Artificial intelligence is transforming the way we approach
problem-solving in the modern world. Furthermore, it is important to note
that AI systems leverage advanced algorithms to optimize outcomes across
various domains. Additionally, machine learning models facilitate the
processing of large datasets to generate meaningful insights. In conclusion,
it is crucial to implement robust AI governance frameworks to ensure
responsible development. Moreover, stakeholders should utilize these tools
to streamline workflows and enhance productivity overall."""

    HUMAN_TEXT = b"""I have been staring at this codebase for three weeks and
starting to think the original architect had a personal grudge against future
maintainers. The abstraction layers make sense individually - I get why each
decision was made - but together they create this weird accordion of
indirection where changing one thing requires touching six files in three
packages. What really gets me is the naming. Half the classes are named after
what they *do*, half after what they *are*. And the comments! Every function
has a docstring that restates the signature in different words. Processes the
item. Thanks. Would not have guessed."""

    for label, data, ext in [
        ("AI-generated Python",  AI_CODE,   ".py"),
        ("Human-written Python", HUMAN_CODE,".py"),
        ("AI-generated text",    AI_TEXT,   ".txt"),
        ("Human-written text",   HUMAN_TEXT,".txt"),
    ]:
        print(f"\n--- {label} ---")
        d.print_report(d.analyze_bytes(data, ext, label))

    print("\n  CAVEAT: Scores reflect entropy signatures, not legal authorship.")
    print("  A formulaic human scores low. A heavily-edited AI scores high.")
    print("  Use as a probabilistic signal, not a verdict.\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Universal AI Content Detector")
    parser.add_argument("--file", type=str,         help="Analyze a single file")
    parser.add_argument("--dir",  type=str,         help="Scan a directory")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    det  = AIContentDetector()

    if args.file:
        r = det.analyze_file(args.file)
        if args.json:
            print(json.dumps({k:v for k,v in r.items() if k != 'type_signals'}, indent=2))
        else:
            det.print_report(r)
    elif args.dir:
        rs = det.scan_directory(args.dir)
        for r in rs: det.print_report(r)
        det.print_directory_summary(rs)
    else:
        run_demo()
        print("  python ai_detector.py --file mycode.py")
        print("  python ai_detector.py --file photo.png")
        print("  python ai_detector.py --dir ./project/")
        print("  python ai_detector.py --file doc.txt --json\n")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the bundled offline local-corpus JSONL files for the demo."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "local_corpus"
OUT.mkdir(parents=True, exist_ok=True)

ENGLISH = [
    "The transformer architecture is the foundation of modern LLMs. Self-attention models long-range dependencies without recurrence.",
    "Curriculum learning schedules data by difficulty. Early stages prefer clean text; annealing reserves high-value examples.",
    "Tokenization maps text to IDs. A frozen tokenizer hash guarantees identical encodings across runs.",
    "Data contamination occurs when eval examples leak into training. Content hashes and firewalls prevent this.",
    "Packing fills fixed-length sequences with useful tokens. Document boundaries use EOS markers.",
    "A consumption ledger records every batch. Ledger offsets enable crash recovery and historical replay.",
    "Gradient accumulation simulates larger batches. Global batch equals microbatch times accumulation times GPUs.",
    "Checkpoints must bind model weights to dataloader state including RNG and ledger offsets.",
    "OPUS scores candidate batches. Protected floors can override rejection for scarce lanes.",
    "Useful loss-bearing tokens per second is the key dataloader throughput metric.",
    "Immutable shards store token IDs. Content and tokenizer hashes seal meaning and payload.",
    "Replay regenerates historical batches. Matching hashes prove exact reconstruction.",
    "Deduplication uses exact hashing and MinHash for near-duplicates.",
    "Loss masks mark gradient-bearing positions. Padding and BOS are excluded.",
    "Prefetch hides storage latency while workers prepare the next batches.",
    "Per-token perplexity reveals localized difficulty hidden by shard averages.",
    "Validation shards may be read but never become gradient-bearing data.",
    "Long-context training is expensive; unused positions waste high-value compute.",
    "License and provenance metadata travel with every admitted shard.",
    "A learning ledger links loss outcomes back to source shards for V6 planning.",
    "Microbatches are per-GPU units; ledgers reconstruct every unit inside a step.",
    "Deterministic ordering from seed plus append-only ledgers handle messy restarts.",
    "Mixture targets allocate budget across code, reasoning, Indic, and English lanes.",
    "RoPE encodings rotate queries and keys by position; batch position IDs must match.",
    "Forking starts a new ledger branch from an earlier checkpoint with explicit divergence.",
]

CODE = [
    "def pack_docs(docs, seq_len, eos):\n    out = []\n    cur = []\n    for d in docs:\n        if len(cur) + len(d) + 1 > seq_len:\n            out.append(cur)\n            cur = []\n        cur.extend(d + [eos])\n    if cur:\n        out.append(cur)\n    return out\n",
    "class ShardStore:\n    def __init__(self, path):\n        self.bin = path + '.bin'\n        self.idx = path + '.idx'\n\n    def write(self, tokens):\n        import numpy as np\n        np.asarray(tokens, dtype=np.uint32).tofile(self.bin)\n",
    "def causal_mask(n):\n    import numpy as np\n    return np.tril(np.ones((n, n), dtype=np.int32))\n",
    "def opus_decide(score, floor_override=False):\n    if floor_override:\n        return 'ACCEPT'\n    return 'ACCEPT' if score > 0.2 else 'REJECT'\n",
    "def save_rng():\n    import random\n    import numpy as np\n    return {'py': random.getstate(), 'np': np.random.get_state()}\n",
    "class Adam:\n    def __init__(self, w, lr=1e-3):\n        self.w = w\n        self.m = 0 * w\n        self.v = 0 * w\n        self.t = 0\n        self.lr = lr\n\n    def step(self, g):\n        self.t += 1\n        self.m = 0.9 * self.m + 0.1 * g\n        self.v = 0.999 * self.v + 0.001 * (g * g)\n        self.w -= self.lr * self.m / (self.v ** 0.5 + 1e-8)\n",
    "def verify_replay(a, b):\n    keys = ['input_ids', 'loss_mask', 'attention_mask', 'position_ids', 'doc_ids', 'token_spans']\n    return all(a[k] == b[k] for k in keys)\n",
    "def admission(manifest, tok_hash, eval_ids):\n    if manifest['tokenizer_hash'] != tok_hash:\n        return False\n    if manifest['shard_id'] in eval_ids:\n        return False\n    return True\n",
    "def efficiency(loss_mask):\n    useful = sum(sum(row) for row in loss_mask)\n    total = len(loss_mask) * len(loss_mask[0])\n    return useful / total if total else 0.0\n",
    "def interpolate(w0, w1, alpha):\n    return {k: (1 - alpha) * w0[k] + alpha * w1[k] for k in w0}\n",
]
CODE += [f"def helper_{i}(x):\n    return x * {i} + {i}\n" for i in range(15)]

REASONING = [
    "Mean Value Theorem: exists c with f'(c)=(f(b)-f(a))/(b-a) when f is continuous on [a,b] and differentiable on (a,b).",
    "Master Theorem: T(n)=aT(n/b)+Theta(n^{log_b a}) yields Theta(n^{log_b a} log n).",
    "Bayes: P(H|D)=P(D|H)P(H)/P(D). MAP maximizes the posterior density.",
    "Cross-entropy loss is -log p(true token). Perplexity is exp of average loss.",
    "Hoeffding bounds sample-mean deviation with high probability after O(log(1/delta)/eps^2) samples.",
    "SVD yields best low-rank approximation in Frobenius and spectral norms.",
    "Convex local minima are global; projection onto closed convex sets is unique.",
    "Dijkstra finds shortest paths with nonnegative weights using a priority queue.",
    "Knapsack DP: dp[i][w]=max(dp[i-1][w], v_i+dp[i-1][w-w_i]).",
    "Picard-Lindelof gives unique local ODE solutions under Lipschitz conditions.",
]
REASONING += [f"Claim {i}: for n>={i}, the sum 1+2+...+n equals n(n+1)/2 by induction." for i in range(15)]

INDIC = [
    "यह एक शैक्षिक हिंदी पाठ है। टोकनाइज़र को देवनागरी के लिए सुरक्षित सामान्यीकरण चाहिए।",
    "मशीन लर्निंग मॉडल को विविध भाषाओं के डेटा की आवश्यकता होती है।",
    "मूल्यांकन डेटा को कभी प्रशिक्षण बैच में नहीं मिलाया जाना चाहिए।",
    "चेकपॉइंट में मॉडल वेट के साथ डेटालोडर ऑफसेट भी सहेजना चाहिए।",
    "ओपस चयन उपयोगी बैचों को स्वीकार करता है और संरक्षित फ्लोर दुर्लभ लेन बचाता है।",
    "पैकिंग दक्षता उपयोगी टोकनों का अनुपात है। अधिक पैडिंग बर्बाद कंप्यूट है।",
    "रीप्ले में बैच हैश, टोकन स्पैन और मास्क मूल रन से मेल खाने चाहिए।",
    "पाठ्यक्रम ठंडे प्रारंभ से तर्क-भारी, संतुलित, कोड-भारी और एनीलिंग तक जाता है।",
]
INDIC += [f"इंडिक दस्तावेज़ संख्या {i}: डिजिटल समावेशन के लिए खुले मॉडल महत्वपूर्ण हैं।" for i in range(17)]

SOURCES = {
    "english.jsonl": ("local-fineweb-edu", ENGLISH),
    "code.jsonl": ("local-starcoder-python", CODE),
    "reasoning.jsonl": ("local-open-web-math", REASONING),
    "indic.jsonl": ("local-indic-sangraha", INDIC),
}

if __name__ == "__main__":
    for name, (source, rows) in SOURCES.items():
        path = OUT / name
        with path.open("w", encoding="utf-8") as f:
            for text in rows:
                f.write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
        print(f"Wrote {path} ({len(rows)} docs)")

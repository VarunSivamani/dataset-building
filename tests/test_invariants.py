#!/usr/bin/env python3
"""Invariant checks for the Session 6 Training Data Execution System."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_demo import (
    CONFIG,
    AdamOptimizer,
    BinaryShardStore,
    FrozenTokenizer,
    TinyLM,
    admission_gate,
    batch_fingerprint,
    capture_rng_state,
    compile_mixture_schedule,
    create_shards,
    curriculum_weights,
    generate_corpus,
    pack_batch,
    restore_rng_state,
    seed_everything,
)
import numpy as np
import random


class TestTokenizer(unittest.TestCase):
    def test_tokenizer_determinism(self):
        t1 = FrozenTokenizer(seed=42)
        t2 = FrozenTokenizer(seed=42)
        text = "Hello world — हिंदी + code def f(): pass"
        self.assertEqual(t1.encode(text), t2.encode(text))
        self.assertEqual(t1.tokenizer_hash, t2.tokenizer_hash)

    def test_tokenizer_hash_mismatch_rejects_shard(self):
        docs = generate_corpus(40, seed=42, prefer_hf=False)
        tok = FrozenTokenizer(seed=42)
        with tempfile.TemporaryDirectory() as td:
            _, shards, manifests = create_shards(docs, tok, Path(td) / "shards", shard_size=8)
            # Corrupt tokenizer hash on one train shard
            for m in manifests:
                if not m.is_eval:
                    m.tokenizer_hash = "deadbeef" * 8
                    break
            admitted, blocked, decisions = admission_gate(manifests, tok, set())
            self.assertTrue(any(not d["checks"]["tokenizer_hash"]["passed"] for d in decisions))
            self.assertTrue(any(m.tokenizer_hash.startswith("deadbeef") for m in blocked))


class TestEvalFirewall(unittest.TestCase):
    def test_eval_shards_never_enter_training(self):
        docs = generate_corpus(60, seed=42, prefer_hf=False)
        tok = FrozenTokenizer(seed=42)
        with tempfile.TemporaryDirectory() as td:
            _, _, manifests = create_shards(docs, tok, Path(td) / "shards", shard_size=8)
            admitted, blocked, _ = admission_gate(manifests, tok, set())
            self.assertTrue(any(m.is_eval for m in blocked))
            self.assertTrue(all(not m.is_eval and m.lane != "eval" for m in admitted))


class TestBinaryShards(unittest.TestCase):
    def test_bin_idx_roundtrip(self):
        docs = generate_corpus(40, seed=7, prefer_hf=False)
        tok = FrozenTokenizer(seed=7)
        with tempfile.TemporaryDirectory() as td:
            store, shards, manifests = create_shards(docs, tok, Path(td), shard_size=8)
            m = [x for x in manifests if not x.is_eval][0]
            arr = shards[m.shard_id]
            self.assertTrue(Path(m.bin_path).exists())
            self.assertTrue(Path(m.idx_path).exists())
            self.assertTrue(np.issubdtype(arr.dtype, np.integer))
            self.assertEqual(len(arr), m.total_tokens)
            store.close()


class TestPacking(unittest.TestCase):
    def _setup(self):
        docs = generate_corpus(80, seed=42, prefer_hf=False)
        tok = FrozenTokenizer(seed=42)
        td = tempfile.mkdtemp()
        store, shards, manifests = create_shards(docs, tok, Path(td) / "shards", shard_size=8)
        admitted, _, _ = admission_gate(manifests, tok, set())
        schedule = compile_mixture_schedule(admitted, 20, seed=42)
        mmap = {m.shard_id: shards[m.shard_id] for m in admitted}
        man = {m.shard_id: m for m in admitted}
        return td, store, mmap, man, schedule, tok

    def test_multi_doc_packing_and_efficiency(self):
        td, store, shards, manifests, schedule, tok = self._setup()
        try:
            entry = next(e for e in schedule if e["opus_decision"] == "ACCEPT")
            batch = pack_batch(entry, shards, manifests, CONFIG["batch_size"], CONFIG["seq_len"], CONFIG["seed"])
            # efficiency formula
            useful = sum(sum(r) for r in batch.loss_mask)
            total = CONFIG["batch_size"] * CONFIG["seq_len"]
            self.assertAlmostEqual(batch.packing_efficiency, useful / total)
            # at least one sequence should include multiple docs when corpus allows
            multi = any(len(docs) > 1 for docs in batch.doc_ids)
            self.assertTrue(multi or batch.packing_efficiency > 0.5)
            # token spans present
            self.assertTrue(all(len(spans) >= 1 for spans in batch.token_spans))
        finally:
            store.close()
            shutil.rmtree(td, ignore_errors=True)

    def test_loss_mask_excludes_padding_and_bos(self):
        td, store, shards, manifests, schedule, tok = self._setup()
        try:
            entry = next(e for e in schedule if e["opus_decision"] == "ACCEPT")
            batch = pack_batch(entry, shards, manifests, CONFIG["batch_size"], CONFIG["seq_len"], CONFIG["seed"])
            for seq, loss in zip(batch.input_ids, batch.loss_mask):
                for tok_id, m in zip(seq, loss):
                    if tok_id == CONFIG["pad_token"] or tok_id == CONFIG["bos_token"]:
                        self.assertEqual(m, 0)
        finally:
            store.close()
            shutil.rmtree(td, ignore_errors=True)


class TestCurriculumAndFloors(unittest.TestCase):
    def test_smooth_curriculum_stages(self):
        names = set()
        for step in range(50):
            name, weights = curriculum_weights(step, 50)
            names.add(name.split("->")[0])
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertTrue({"cold", "reasoning-heavy", "balanced", "code-heavy", "final-annealing"} & names)

    def test_protected_floors_maintained(self):
        docs = generate_corpus(100, seed=42, prefer_hf=False)
        tok = FrozenTokenizer(seed=42)
        with tempfile.TemporaryDirectory() as td:
            _, shards, manifests = create_shards(docs, tok, Path(td), shard_size=10)
            admitted, _, _ = admission_gate(manifests, tok, set())
            schedule = compile_mixture_schedule(admitted, 120, seed=42)
            counts = {l: 0 for l in CONFIG["lanes"]}
            for e in schedule:
                if e["opus_decision"] == "ACCEPT":
                    counts[e["lane"]] += 1
            total = max(1, sum(counts.values()))
            for lane, floor in CONFIG["protected_floors"].items():
                self.assertGreaterEqual(counts[lane] / total, floor * 0.25)


class TestResumeAndReplay(unittest.TestCase):
    def test_resume_expected_batch_id(self):
        docs = generate_corpus(60, seed=42, prefer_hf=False)
        tok = FrozenTokenizer(seed=42)
        with tempfile.TemporaryDirectory() as td:
            _, shards, manifests = create_shards(docs, tok, Path(td), shard_size=8)
            admitted, _, _ = admission_gate(manifests, tok, set())
            schedule = compile_mixture_schedule(admitted, 30, seed=42)
            # Simulate checkpoint at step 10 -> next accepted batch id
            next_step = 11
            expected = f"batch_{next_step:06d}"
            entry = schedule[next_step]
            if entry["opus_decision"] != "REJECT":
                mmap = {m.shard_id: shards[m.shard_id] for m in admitted}
                man = {m.shard_id: m for m in admitted}
                batch = pack_batch(entry, mmap, man, CONFIG["batch_size"], CONFIG["seq_len"], CONFIG["seed"])
                self.assertEqual(batch.batch_id, expected)

    def test_replay_identical_fields(self):
        docs = generate_corpus(60, seed=42, prefer_hf=False)
        tok = FrozenTokenizer(seed=42)
        with tempfile.TemporaryDirectory() as td:
            _, shards, manifests = create_shards(docs, tok, Path(td), shard_size=8)
            admitted, _, _ = admission_gate(manifests, tok, set())
            schedule = compile_mixture_schedule(admitted, 20, seed=42)
            mmap = {m.shard_id: shards[m.shard_id] for m in admitted}
            man = {m.shard_id: m for m in admitted}
            for entry in schedule:
                if entry["opus_decision"] == "REJECT":
                    continue
                a = batch_fingerprint(pack_batch(entry, mmap, man, CONFIG["batch_size"], CONFIG["seq_len"], CONFIG["seed"]))
                b = batch_fingerprint(pack_batch(entry, mmap, man, CONFIG["batch_size"], CONFIG["seq_len"], CONFIG["seed"]))
                for field in ["batch_hash", "input_ids", "attention_mask", "loss_mask", "position_ids", "doc_ids", "token_spans"]:
                    self.assertEqual(a[field], b[field], field)


class TestRNGAndOptimizer(unittest.TestCase):
    def test_rng_checkpoint_restore(self):
        seed_everything(123)
        _ = random.random()
        _ = np.random.rand()
        blob = capture_rng_state()
        a = random.random()
        b = float(np.random.rand())
        restore_rng_state(blob)
        self.assertEqual(random.random(), a)
        self.assertEqual(float(np.random.rand()), b)

    def test_adam_state_roundtrip(self):
        model = TinyLM(512, dim=16, seed=1)
        opt = AdamOptimizer(model.parameters(), lr=1e-2)
        x = [[1, 2, 3, 4]]
        mask = [[1, 1, 1, 1]]
        _, _, grads = model.forward_backward(x, mask)
        opt.step(grads)
        state = opt.state_dict()
        h1 = opt.state_hash()
        opt2 = AdamOptimizer(model.parameters(), lr=1e-2)
        opt2.load_state_dict(state)
        self.assertEqual(opt2.state_hash(), h1)
        self.assertEqual(opt2.t, state["t"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

# Session 6 — Training Data Execution System

**ERA V5 — Building the Training Dataset**

A compact yet complete **Training Data Execution System** for V5. The aim is not scale.
The aim is to show the data stack is **correct, reproducible, auditable, and efficient**.

```bash
pip install -r requirements.txt
python run_demo.py
```

That single command rebuilds the full demonstration and the `submission_artifacts/`
directory with no manual steps.

---

## The Assignment Path

The system covers the full pipeline:

```text
documents
  -> tokenized shards
  -> manifests
  -> mixture schedule
  -> packing
  -> batches
  -> training
  -> consumption ledger
  -> learning ledger
  -> checkpoint
  -> crash
  -> resume
  -> replay
  -> audit
```

A small corpus, tokenizer and model are used intentionally. All evidence must be
**produced by the run**, not hard-coded.

---

## What the System Demonstrates

| Requirement | How this repository proves it |
|-------------|-------------------------------|
| Immutable tokenized shards with manifests | `.bin` / `.idx` shards + `manifests/shard_*.json` |
| Frozen tokenizer and content hashes | `tokenizer/tokenizer.json` + SHA-256 identity / payload hashes |
| Packing policies for different data types | Multi-doc packing per lane with EOS boundaries |
| Correct loss / attention masks and position ids | `loss_mask`, `attention_mask`, lane-local `position_ids` |
| Curriculum stages, lane weights, protected floors | Smooth curriculum + floor enforcement in `schedule.jsonl` |
| Evaluation and validation firewalls | Eval shards are blocked in `manifests/admission.json` |
| OPUS accept / reject / defer / protected-floor override | `ledgers/opus_decisions.jsonl` |
| Training consumption and learning ledgers | `ledgers/consumption.jsonl`, `ledgers/learning.jsonl` |
| Token- or sample-level loss tracking | Learning ledger ties loss to shard / doc / span |
| Checkpoints tied to ledger offsets | `checkpoints/ckpt_step_*/metadata.json` |
| Crash recovery without skip or repeat | Intentional crash + `resume_next_batch_matched` |
| Replay of the same historical stream | `replay_hash_matched` over batch ids, spans and hashes |
| Fork from an earlier checkpoint | `evidence/fork_proof.json` |
| Packing utilization and useful loss-bearing tokens/sec | `performance.json` |

The run **intentionally crashes**, restores from the saved checkpoint and shows the
next batch is exactly the expected one. It also **replays** an earlier range and
verifies that reconstructed batch ids, token spans and hashes match the original.

---

## Submission Contents

This repository includes:

1. **Full implementation** — `run_demo.py` (control plane + data plane)
2. **Concise README** — this file (architecture and design choices)
3. **Single command** — `python run_demo.py`
4. **Automated tests** — `python tests/test_invariants.py`
5. **Generated execution log** — `submission_artifacts/run.log`
6. **Machine-readable evidence bundle** — `evidence.json` + `evidence.md`
7. **Generated manifests, ledgers, checkpoints and performance reports**

Optional (offline-first by default):

```bash
python run_demo.py --prefer-hf   # try Hugging Face first, still falls back locally
pip install tiktoken transformers datasets   # optional backends
```

---

## Generated Artifact Layout

`python run_demo.py` creates:

```text
submission_artifacts/
  run.log
  evidence.json
  evidence.md
  manifests/
  ledgers/
  checkpoints/
  performance.json
```

Expanded tree (also generated):

```text
submission_artifacts/
├── run.log
├── evidence.json
├── evidence.md
├── performance.json
├── tokenizer/tokenizer.json
├── shards/*.bin + *.idx
├── manifests/
│   ├── shard_*.json
│   ├── admission.json
│   └── schedule.jsonl
├── ledgers/
│   ├── consumption.jsonl
│   ├── learning.jsonl
│   └── opus_decisions.jsonl
├── checkpoints/ckpt_step_*/{model.npz, optimizer.pkl, metadata.json}
└── evidence/{resume,replay,fork}_proof.json
```

---

## Execution Log (`run.log`)

`run.log` captures the full event sequence, including:

- shards created
- manifests validated
- evaluation data blocked
- mixture compiled
- batches packed
- OPUS decisions recorded
- checkpoint saved
- crash simulated
- run resumed
- historical stream replayed
- branch forked
- audit completed
- performance measured

Key markers appear as:

```text
[PASS] tokenizer_hash_verified
[PASS] eval_shard_blocked
[PASS] checkpoint_saved
[PASS] resume_next_batch_matched
[PASS] replay_hash_matched
```

---

## Evidence Bundle

### `evidence.json`

Machine-readable summary of whether each major requirement passed and where its
supporting proof lives. Keys include:

- `tokenizer_integrity`
- `evaluation_firewall`
- `packing_correctness`
- `mixture_compliance`
- `opus_audit_trail`
- `crash_recovery`
- `replay`
- `learning_trace`
- `throughput`

### `evidence.md`

Human-readable summary in assignment table form:

| Requirement | Result | Evidence |
|-------------|--------|----------|
| Tokenizer integrity | PASS/FAIL | Manifest record |
| Evaluation firewall | PASS/FAIL | Blocked-shard event |
| Packing correctness | PASS/FAIL | Packed-batch report |
| Mixture compliance | PASS/FAIL | Planned vs actual shares |
| OPUS audit trail | PASS/FAIL | Candidate decision records |
| Crash recovery | PASS/FAIL | Expected and resumed batch ids |
| Replay | PASS/FAIL | Original and replay hashes |
| Learning trace | PASS/FAIL | Loss linked to source data |
| Throughput | PASS/FAIL | Performance report |

The bundle is **produced by the code**. Hard-coded evidence is not accepted.

---

## Architecture and Design Decisions

### Control plane vs data plane

| Plane | Responsibility |
|-------|----------------|
| **Control plane** | Admission firewall, curriculum/OPUS mixture compiler, checkpoint + ledger policy |
| **Data plane** | Tokenizer, immutable shards, packing/masks, TinyLM + Adam consumer |

### End-to-end design

1. **Documents** — lane-tagged corpus (`code`, `reasoning`, `indic`, `english`) with
   eval flags; HF optional, local JSONL is default, synthetic is last fallback.

2. **Frozen tokenizer** — `tiktoken` → HF `gpt2` → offline BPE simulation; identity
   JSON is hashed with SHA-256.

3. **Immutable shards** — contiguous `uint32` token files plus document index and
   JSON manifests (tokenizer hash, content hash, provenance).

4. **Admission** — fail-closed on tokenizer mismatch; eval shards never enter
   loss-bearing training.

5. **Mixture schedule** — smooth curriculum (`cold` → `reasoning-heavy` →
   `balanced` → `code-heavy` → `annealing`) with protected floors and OPUS
   accept / reject / defer / floor-override.

6. **Packing** — multi-document concatenation with EOS; attention/loss masks;
   document-local position ids; batch fingerprint hash.

7. **Training + ledgers** — consumption (what was consumed), learning
   (loss ↔ source spans), OPUS decisions (why).

8. **Checkpoint** — model + Adam (`m`/`v`/`t`) + Python/NumPy RNG + ledger offset
   + batch id.

9. **Crash → resume** — restore the coupled state; assert the next batch id
   equals the scheduled successor.

10. **Replay / fork / audit** — rebuild historical fingerprints; fork from an
    earlier checkpoint; emit evidence and performance metrics.

### Design principles

- **Determinism** — fixed seed plus explicit schedule rows
- **Content addressing** — tokenizer and shard payloads are content-hashed
- **Append-only audit** — consumption / learning / OPUS ledgers
- **State coupling** — checkpoint has no meaning without ledger offset, RNG and optimizer
- **Fail-closed admission** — mismatch or eval overlap ⇒ reject

### Why this shape

| Decision | Rationale |
|----------|-----------|
| Binary `.bin/.idx` shards | Immutable, hashable, production-like indexed datasets |
| Multi-doc packing + masks | Proves utilization and that PAD/BOS do not contribute to loss |
| OPUS + floors in schedule | Mixture policy is runnable and auditable, not descriptive |
| Real Adam + RNG in ckpt | Resume and replay stay scientifically comparable |
| Offline-first corpus | Reviewer can run without network or HF auth |

---

## Repository Layout

```text
S6/
├── run_demo.py                 # Full demonstration entry point
├── requirements.txt            # numpy required; optional backends listed
├── README.md
├── data/
│   ├── generate_local_corpus.py
│   └── local_corpus/*.jsonl
├── tests/test_invariants.py    # Invariant test suite
└── submission_artifacts/       # Regenerated by run_demo.py
```

---

## Scoring Alignment (1,000 Points)

| Area | Points | Evidence in this repo |
|------|-------:|------------------------|
| End-to-end execution | 150 | `python run_demo.py` → full `submission_artifacts/` |
| Shards, manifests, tokenizer integrity | 100 | shards + manifests + tokenizer hash |
| Packing, masks, batch correctness | 150 | packed batches + masks + fingerprints |
| Mixture schedule, protected floors, OPUS | 150 | `schedule.jsonl` + OPUS ledger |
| Consumption and learning ledgers | 150 | consumption + learning JSONL |
| Checkpoint, crash, resume, replay, fork | 150 | checkpoints + resume/replay/fork proofs |
| Evaluation and validation firewall | 50 | blocked eval in admission |
| Throughput and packing efficiency | 50 | `performance.json` |
| Tests, evidence quality, documentation | 50 | tests + generated evidence + this README |
| **Total** | **1,000** | |

A subsystem is awarded full credit only when its outcome is backed by
**reproducible evidence**.

Common failure modes the demo is built to catch:

- Resume repeats or skips a batch → crash recovery fails
- Replay yields different batch hashes → replay fails
- Evaluation data leaks into a loss-bearing batch → firewall fails
- Packing/throughput figures cannot be reconstructed → throughput fails

---

## Evaluation Steps (How Graders Should Check)

### Step 1: Run

```bash
python run_demo.py
```

Recreates the entire `submission_artifacts/` directory.

### Step 2: Check evidence

Compare `run.log`, `evidence.json` and `evidence.md` against the generated
manifests, ledgers, checkpoints and `performance.json`.

### Step 3: Inspect code

Confirm evidence is emitted by the implementation (`run_demo.py`), not mocked
or hard-coded.

---

## Automated Tests

```bash
python tests/test_invariants.py
```

Covers tokenizer integrity, eval firewall, binary shards, packing/masks,
curriculum floors, resume/replay fingerprints and RNG/Adam restore.

---

## Operator Checklist

1. `python run_demo.py` finishes without manual steps
2. `submission_artifacts/` is fully regenerated
3. Log shows `[PASS] resume_next_batch_matched`
4. Log shows `[PASS] replay_hash_matched`
5. Eval shards appear in the blocked admission set
6. `python tests/test_invariants.py` is green
7. `evidence.md` requirement table is PASS for all rows

---

## Completeness Criterion

The task is complete when the system can demonstrate:

- **what** it consumed,
- **why** it consumed it,
- **what** the model learned from it,
- **how** the run can be reconstructed.

```bash
python run_demo.py
```

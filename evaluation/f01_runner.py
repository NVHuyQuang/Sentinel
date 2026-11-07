"""F01 vertical slice runner: record (or replay) -> detect -> triage -> investigate -> evaluate.

    py -3.12 -m evaluation.f01_runner                 # record fresh snapshots, then run
    py -3.12 -m evaluation.f01_runner --replay        # reuse snapshots in evaluation/work
    py -3.12 -m evaluation.f01_runner --primary laya  # real Laya (needs checkpoint/device)

The hidden manifest is read only after the pipeline (and its investigation) has finished.
"""

import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path

from evaluation.score import evaluate, render_markdown
from services.decision_engine.providers.laya import LayaAdapter
from services.decision_engine.providers.rules import RulesProvider
from services.investigation.stub import investigate
from services.pipeline import fit_detectors, run_pipeline
from services.telemetry.backend import SnapshotBackend
from simulator.fault_controller import GroundTruthStore
from simulator.recorder import SimConfig, record_snapshot

TRAIN_SEED, TEST_SEED = 11, 12
RUN_ID = "f01-demo"

# Laya answers as documented in docs/laya-contract.md: category LATENCY, SEV4, priority ~0.5.
_ANSWER = {
    "category": {"type": "choice", "choice": "LATENCY", "probabilities": {"LATENCY": 0.6},
                 "confidence": 0.2, "answer_confidence": 0.6,
                 "action": {"act_probability": 1.0}},
    "severity": {"type": "choice", "choice": "SEV4", "probabilities": {"SEV4": 0.7},
                 "confidence": 0.4, "answer_confidence": 0.7,
                 "action": {"act_probability": 1.0}},
    "priority_score": {"type": "score", "score": 4.5, "probabilities": {"4": 0.5},
                       "confidence": 0.0, "answer_confidence": 0.5,
                       "action": {"act_probability": 1.0}},
    "investigate": {"type": "noul", "noul": 0.9, "confidence": 0.9, "answer_confidence": 0.9,
                    "action": {"act_probability": 1.0}},
    "deployment_relevant": {"type": "noul", "noul": 0.1, "confidence": 0.1,
                            "answer_confidence": 0.1, "action": {"act_probability": 1.0}},
}


class StubLayaRouter:
    """Stand-in Router so replay never needs a GPU. Not a recording of real Laya output."""

    def predict(self, text, questions):
        return {"model": "laya-stub-router", "answers": copy.deepcopy(_ANSWER),
                "routing": {"repo": "stub"}}


def make_primary(kind: str, device: str):
    if kind == "rules":
        return RulesProvider()
    if kind == "laya":
        return LayaAdapter(device=device)
    return LayaAdapter(device="cpu", timeout_ms=5000, router=StubLayaRouter())


def record_all(work: Path) -> None:
    record_snapshot(work / "train", SimConfig(seed=TRAIN_SEED, duration_s=3000))
    record_snapshot(work / "test", SimConfig(seed=TEST_SEED, onset_s=600),
                    GroundTruthStore(work / "ground_truth"), RUN_ID)


async def run(work: Path, out: Path, primary_kind: str = "stub-laya", device: str = "cuda") -> dict:
    """Replay the snapshots in `work` (record_all must have run; it owns its own event loop)."""
    train = SnapshotBackend(work / "train")
    test = SnapshotBackend(work / "test")
    primary = make_primary(primary_kind, device)
    result = await run_pipeline(test, fit_detectors(train), primary, RulesProvider(), investigate)
    # Pipeline and investigation are complete; only now may the evaluator open the manifest.
    manifest = GroundTruthStore(work / "ground_truth").read(RUN_ID)
    report = evaluate(manifest, test, result, primary_kind)
    out.mkdir(parents=True, exist_ok=True)
    (out / "f01_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    (out / "f01_report.md").write_text(render_markdown(report))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("evaluation/work"))
    ap.add_argument("--out", type=Path, default=Path("evaluation/reports"))
    ap.add_argument("--primary", choices=["stub-laya", "laya", "rules"], default="stub-laya")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--replay", action="store_true")
    a = ap.parse_args(argv)
    if not a.replay:
        record_all(a.work)
    report = asyncio.run(run(a.work, a.out, a.primary, a.device))
    print(render_markdown(report))
    return 0 if report["verdict"]["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

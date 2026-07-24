"""Round-trip check on the local metric harness.

Builds a tiny synthetic CTC sequence where the prediction misses one cell in the final
frame, then asserts the scorer reports exactly that error and the AOGM matches the
hand-computed weighted cost. Guards against silently mis-wiring traccuracy's API or
mislabelling the AOGM weights (which already bit us once -- the edge weights were swapped).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from ctrack.eval.ctc import error_breakdown, score_ctc

pytest.importorskip("traccuracy")


def _write_seq(d: Path, track_file: str, drop_cell2_at_last_frame: bool) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for t in range(3):
        im = np.zeros((32, 32), dtype=np.uint16)
        im[5:10, 5 + t : 10 + t] = 1                       # cell 1: drifts right
        if not (drop_cell2_at_last_frame and t == 2):
            im[20:25, 20:25] = 2                            # cell 2: static
        tifffile.imwrite(d / f"mask{t:03d}.tif", im)
    end2 = 1 if drop_cell2_at_last_frame else 2
    (d / track_file).write_text(f"1 0 2 0\n2 0 {end2} 0\n")
    return d


def test_scorer_reports_the_one_injected_miss(tmp_path: Path) -> None:
    gt = _write_seq(tmp_path / "GT", "man_track.txt", drop_cell2_at_last_frame=False)
    pred = _write_seq(tmp_path / "RES", "res_track.txt", drop_cell2_at_last_frame=True)

    res = score_ctc(gt, pred)

    # exactly one missed cell and the one edge leading to it; nothing else
    assert res["fn_nodes"] == 1
    assert res["fn_edges"] == 1
    assert res["fp_nodes"] == res["ns_nodes"] == res["fp_edges"] == res["ws_edges"] == 0

    # 1 x 10 (FN node) + 1 x 1.5 (FN edge)
    assert res["AOGM"] == pytest.approx(11.5)
    assert sum(d["cost"] for d in error_breakdown(res).values()) == pytest.approx(res["AOGM"])

    assert 0.0 < res["TRA"] < 1.0
    assert "CHOTA" in res


def test_perfect_prediction_scores_one(tmp_path: Path) -> None:
    gt = _write_seq(tmp_path / "GT", "man_track.txt", drop_cell2_at_last_frame=False)
    pred = _write_seq(tmp_path / "RES", "res_track.txt", drop_cell2_at_last_frame=False)

    res = score_ctc(gt, pred)

    assert res["AOGM"] == pytest.approx(0.0)
    assert res["TRA"] == pytest.approx(1.0)
    assert res["DET"] == pytest.approx(1.0)

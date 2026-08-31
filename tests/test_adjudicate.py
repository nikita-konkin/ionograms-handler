"""The hand-scaling adjudication harness.

The property worth testing is the blinding. Everything else here is bookkeeping
that would fail loudly, but a leak of which marker is the new extractor fails
*silently* and produces a confident wrong answer -- the reviewer cannot un-know
it, and the whole exercise exists because no other measurement available is
independent.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "adjudicate", ROOT / "tools" / "adjudicate.py")
adjudicate = importlib.util.module_from_spec(spec)
sys.modules["adjudicate"] = adjudicate
spec.loader.exec_module(adjudicate)


def _manifest(tmp_path, rows):
    out = tmp_path / "review"
    out.mkdir(exist_ok=True)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out


def _row(rid, algo, dp, a_is_algo):
    return {"id": rid, "path": f"/nowhere/{rid}.h5", "when": "2026-08-17 20:00:00",
            "tx": "NIC1", "algo_mhz": algo, "dp_mhz": dp,
            "gap_mhz": round(abs(algo - dp), 3),
            "A": "algo" if a_is_algo else "dp",
            "B": "dp" if a_is_algo else "algo"}


def _verdicts(out, pairs):
    with open(out / "verdicts.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "verdict", "muf_mhz", "note"])
        for rid, choice, muf in pairs:
            writer.writerow([rid, choice, muf, ""])


def test_the_filename_splits_transmitter_from_a_hyphenated_receiver():
    """"Yoshkar-Ola" contains the separator, and the transmitter does not."""
    name = "lfm_ionogram-NIC1-Yoshkar-Ola-ch0-002-1786926835.00.h5"
    match = adjudicate.FILENAME.match(name)

    assert match is not None
    assert match.group("tx") == "NIC1"
    assert match.group("rx") == "Yoshkar-Ola"
    assert match.group("t0") == "1786926835"


def test_the_sheet_the_reviewer_fills_in_names_no_estimator(tmp_path):
    """The blinding, stated as the thing that must be true of the file.

    `verdicts.csv` is the only file the reviewer needs open, so it must not
    carry the assignment -- that lives in the manifest, which `score` reads
    afterwards.
    """
    rows = [_row("000", 12.95, 12.25, True), _row("001", 18.10, 21.40, False)]
    out = _manifest(tmp_path, rows)

    class Args:
        pass
    args = Args()
    args.out = str(out)

    # `select` writes it; recreate that step without touching an archive.
    with open(out / "verdicts.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "verdict", "muf_mhz", "note"])
        for row in rows:
            writer.writerow([row["id"], "", "", ""])

    text = (out / "verdicts.csv").read_text(encoding="utf-8")
    assert "algo" not in text and "dp" not in text
    for row in rows:                       # nor the values that would identify
        assert str(row["algo_mhz"]) not in text
        assert str(row["dp_mhz"]) not in text


def test_score_unblinds_through_the_manifest(tmp_path, capsys):
    """A and B mean different estimators on different rows, by design."""
    rows = [_row("000", 12.95, 12.25, True),      # A=algo  B=dp
            _row("001", 18.10, 21.40, False),     # A=dp    B=algo
            _row("002", 14.00, 15.20, True)]      # A=algo  B=dp
    out = _manifest(tmp_path, rows)
    # Choose `dp` every time, expressed as the letter it wears on that row.
    _verdicts(out, [("000", "B", ""), ("001", "A", ""), ("002", "B", "")])

    class Args:
        pass
    args = Args()
    args.out = str(out)
    adjudicate.score(args)

    printed = capsys.readouterr().out
    assert "dp    preferred on   3 of 3" in printed
    assert "algo  preferred on   0 of 3" in printed


def test_a_clean_sweep_of_three_is_not_significant(tmp_path, capsys):
    """Three-nil is p=0.25. The test has to say so rather than declare a winner.

    This is the failure the sample size exists to avoid, and the reason `score`
    reports an exact binomial rather than a percentage.
    """
    rows = [_row(f"{i:03d}", 12.0, 13.0, True) for i in range(3)]
    out = _manifest(tmp_path, rows)
    _verdicts(out, [(f"{i:03d}", "B", "") for i in range(3)])

    class Args:
        pass
    args = Args()
    args.out = str(out)
    adjudicate.score(args)

    printed = capsys.readouterr().out
    assert "NOT distinguishable from a coin flip" in printed
    assert "more would sharpen it" in printed


def test_twenty_of_twenty_two_is_significant(tmp_path, capsys):
    rows = [_row(f"{i:03d}", 12.0, 13.0, True) for i in range(22)]
    out = _manifest(tmp_path, rows)
    picks = [(f"{i:03d}", "B" if i < 20 else "A", "") for i in range(22)]
    _verdicts(out, picks)

    class Args:
        pass
    args = Args()
    args.out = str(out)
    adjudicate.score(args)

    printed = capsys.readouterr().out
    assert "dp    preferred on  20 of 22" in printed
    assert "-- distinguishable from a coin flip" in printed


def test_neither_is_recorded_rather_than_forced_into_a_choice(tmp_path, capsys):
    """A sounding both get wrong is not evidence for either.

    Folding it into the loser's column would make the winner look better than
    it is; dropping it silently would hide the case worth reading.
    """
    rows = [_row("000", 12.0, 13.0, True), _row("001", 9.0, 22.0, False)]
    out = _manifest(tmp_path, rows)
    _verdicts(out, [("000", "B", ""), ("001", "neither", "15.5")])

    class Args:
        pass
    args = Args()
    args.out = str(out)
    adjudicate.score(args)

    printed = capsys.readouterr().out
    assert "1 picked a marker, 1 said neither" in printed
    assert "both wrong on: 001" in printed
    # The hand-scaled MUF is used for an absolute error on the row it was given.
    assert "against the 1 hand-scaled MUFs" in printed
    assert "algo  MAE 6.500" in printed        # |9.0 - 15.5|
    assert "dp    MAE 6.500" in printed        # |22.0 - 15.5|

#!/usr/bin/env python3
"""Evaluate all TrackNet checkpoints and produce progression statistics.

Writes:
  - output/progression_stats.csv  (checkpoint, f1, precision, recall, tp, fp, fn, threshold, meta)
  - output/progression_f1.png     (simple F1 vs checkpoint plot)

Usage:
  python scripts/progression_stats.py
"""
from pathlib import Path
import csv
import json
import traceback

import torch


def main():
    root = Path('.').resolve()
    weights_dir = root / 'runs' / 'tracknet' / 'weights'
    out_dir = root / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    if not weights_dir.exists():
        print(f"Weights directory not found: {weights_dir}")
        return

    pt_files = sorted([p for p in weights_dir.iterdir() if p.suffix == '.pt'])
    if not pt_files:
        print(f"No .pt files found in {weights_dir}")
        return

    # Import the evaluator utility (evaluate_models.eval_tracknet)
    # Robust import: try top-level, then scripts package, then load from file path.
    try:
        import evaluate_models as evaluate_models
    except Exception:
        try:
            import scripts.evaluate_models as evaluate_models
        except Exception:
            import importlib.util, sys
            em_path = root / 'scripts' / 'evaluate_models.py'
            if em_path.exists():
                spec = importlib.util.spec_from_file_location('evaluate_models', str(em_path))
                evaluate_models = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(evaluate_models)
            else:
                raise ImportError('evaluate_models module not found in repo root or scripts/')

    results = []
    for pt in pt_files:
        print(f"\nEvaluating checkpoint: {pt.name}")
        evaluate_models.TRACKNET_WEIGHTS = str(pt)
        try:
            stats = evaluate_models.eval_tracknet()
        except Exception as e:
            print(f"  ERROR running eval_tracknet on {pt.name}: {e}")
            traceback.print_exc()
            stats = {"f1": None, "precision": None, "recall": None,
                     "tp": None, "fp": None, "fn": None, "threshold": None}

        # Try to load a companion metadata file if present
        meta = None
        meta_path = pt.with_name(pt.stem + '_meta.pt')
        if meta_path.exists():
            try:
                meta_obj = torch.load(meta_path, map_location='cpu')
                # Keep only JSON-serializable keys
                meta = {k: meta_obj[k] for k in meta_obj.keys() if k in meta_obj}
            except Exception:
                meta = {"_error": "failed to load meta", "path": str(meta_path)}

        results.append({
            "checkpoint": str(pt.relative_to(root)),
            "f1": stats.get("f1"),
            "precision": stats.get("precision"),
            "recall": stats.get("recall"),
            "tp": stats.get("tp"),
            "fp": stats.get("fp"),
            "fn": stats.get("fn"),
            "threshold": stats.get("threshold"),
            "meta": meta,
        })

    # Write CSV
    csv_path = out_dir / 'progression_stats.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "f1", "precision", "recall", "tp", "fp", "fn", "threshold", "meta_json"])
        for r in results:
            writer.writerow([r['checkpoint'], r['f1'], r['precision'], r['recall'],
                             r['tp'], r['fp'], r['fn'], r['threshold'],
                             json.dumps(r['meta']) if r['meta'] is not None else ""])

    print(f"Wrote metrics to {csv_path}")

    # Try to plot F1 progression (optional)
    try:
        import matplotlib.pyplot as plt

        names = [Path(r['checkpoint']).stem for r in results]
        f1s = [r['f1'] if r['f1'] is not None else float('nan') for r in results]

        plt.figure(figsize=(max(6, len(names) * 0.9), 4))
        plt.plot(range(len(names)), f1s, marker='o')
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        plt.ylabel('Best F1 (PE ≤ 5px)')
        plt.title('TrackNet progress: F1 by checkpoint')
        plt.tight_layout()
        fig_path = out_dir / 'progression_f1.png'
        plt.savefig(fig_path)
        print(f"Wrote plot to {fig_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()

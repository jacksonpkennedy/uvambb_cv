#!/usr/bin/env python3
"""Evaluate multiple YOLO weight files on Roboflow val and TrackNet val.

Produces CSV in `output/yolo_variants_eval.csv` and an optional F1 plot.

Usage:
  python scripts/eval_yolo_variants.py            # auto-detect common checkpoints
  python scripts/eval_yolo_variants.py --weights runs/detect/train/weights/best.pt yolo11s.pt
"""
from pathlib import Path
import argparse
import csv
import json
import traceback


def find_default_weights(root: Path):
    candidates = [
        root / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt',
        root / 'runs' / 'detect' / 'train' / 'weights' / 'last.pt',
        root / 'yolo11s.pt',
        root / 'yolo11n.pt',
        root / 'yolo11m.pt',
    ]
    return [p for p in candidates if p.exists()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='*', help='Paths to YOLO .pt weight files')
    parser.add_argument('--plot', action='store_true', help='Save simple F1 plot')
    args = parser.parse_args()

    root = Path('.').resolve()
    out_dir = root / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.weights and len(args.weights) > 0:
        weights = [Path(w) for w in args.weights]
    else:
        weights = find_default_weights(root)

    if not weights:
        print('No weight files provided or found. Pass --weights PATHS or place checkpoints in runs/detect/train/weights or project root.')
        return

    # Import evaluate_models and use its functions (robust to file moves)
    try:
        import evaluate_models as evaluate_models
    except Exception:
        try:
            import scripts.evaluate_models as evaluate_models
        except Exception:
            import importlib.util
            em_path = Path('.') / 'scripts' / 'evaluate_models.py'
            if em_path.exists():
                spec = importlib.util.spec_from_file_location('evaluate_models', str(em_path))
                evaluate_models = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(evaluate_models)
            else:
                raise ImportError('evaluate_models module not found in repo root or scripts/')

    rows = []
    for w in weights:
        w = w.resolve()
        print(f'\n=== Evaluating YOLO weights: {w} ===')
        # Roboflow (curated) eval
        try:
            evaluate_models.YOLO_WEIGHTS = str(w)
            map50, per_class, mean_f1, mean_conf = evaluate_models.eval_yolo()
        except Exception as e:
            print(f'Roboflow eval failed for {w}: {e}')
            traceback.print_exc()
            map50 = mean_f1 = mean_conf = None
            per_class = {}

        # TrackNet-val (apples-to-apples)
        try:
            evaluate_models.YOLO_WEIGHTS = str(w)
            tn_stats = evaluate_models.eval_yolo_on_tracknet_val()
        except Exception as e:
            print(f'TrackNet-val eval failed for {w}: {e}')
            traceback.print_exc()
            tn_stats = {"f1": None, "precision": None, "recall": None,
                        "tp": None, "fp": None, "fn": None, "threshold": None}

        rows.append({
            'checkpoint': str(w.relative_to(root)),
            'roboflow_map50': map50,
            'roboflow_mean_f1': mean_f1,
            'roboflow_mean_conf': mean_conf,
            'roboflow_per_class': per_class,
            'tn_f1': tn_stats.get('f1'),
            'tn_precision': tn_stats.get('precision'),
            'tn_recall': tn_stats.get('recall'),
            'tn_threshold': tn_stats.get('threshold'),
            'tn_tp': tn_stats.get('tp'),
            'tn_fp': tn_stats.get('fp'),
            'tn_fn': tn_stats.get('fn'),
        })

    csv_path = out_dir / 'yolo_variants_eval.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['checkpoint','roboflow_map50','roboflow_mean_f1','roboflow_mean_conf','roboflow_per_class_json',
                         'tn_f1','tn_precision','tn_recall','tn_threshold','tn_tp','tn_fp','tn_fn'])
        for r in rows:
            writer.writerow([r['checkpoint'], r['roboflow_map50'], r['roboflow_mean_f1'], r['roboflow_mean_conf'],
                             json.dumps(r['roboflow_per_class']), r['tn_f1'], r['tn_precision'], r['tn_recall'],
                             r['tn_threshold'], r['tn_tp'], r['tn_fp'], r['tn_fn']])

    print(f'Wrote results to {csv_path}')

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            names = [Path(r['checkpoint']).stem for r in rows]
            f1s = [r['tn_f1'] if r['tn_f1'] is not None else float('nan') for r in rows]
            plt.figure(figsize=(max(6, len(names)*0.9),4))
            plt.bar(range(len(names)), f1s)
            plt.xticks(range(len(names)), names, rotation=45, ha='right')
            plt.ylabel('YOLO F1 on TrackNet val (PE ≤ 5px)')
            plt.title('YOLO variants: TrackNet-val F1')
            plt.tight_layout()
            figp = out_dir / 'yolo_variants_tn_f1.png'
            plt.savefig(figp)
            print(f'Wrote plot to {figp}')
        except Exception as e:
            print('Plotting failed:', e)


if __name__ == '__main__':
    main()

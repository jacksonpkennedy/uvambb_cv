"""Download latest model weights from W&B Artifacts.

Usage:
    python fetch_weights.py --model tracknet
    python fetch_weights.py --model yolo
    python fetch_weights.py --model all
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODELS = {
    "tracknet": {
        "artifact": "tracknet-best:latest",
        "dest": "runs/tracknet/weights",
    },
    "yolo": {
        "artifact": "yolo-best:latest",
        "dest": "runs/detect/train/weights",
    },
}


def fetch(model_name: str):
    import wandb

    project = os.environ.get("WANDB_PROJECT", "uvambb-cv")
    entity = os.environ.get("WANDB_ENTITY")

    if model_name == "all":
        for name in MODELS:
            fetch(name)
        return

    if model_name not in MODELS:
        print(f"Unknown model: {model_name}. Choose from: {', '.join(MODELS)}, all")
        return

    cfg = MODELS[model_name]
    dest = Path(cfg["dest"])
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {model_name} weights from W&B...")
    run = wandb.init(
        project=project,
        entity=entity,
        job_type="download",
        name=f"fetch-{model_name}",
    )
    try:
        artifact = run.use_artifact(cfg["artifact"])
        artifact.download(root=str(dest))
        print(f"Downloaded to {dest}/")
        meta = artifact.metadata
        if meta:
            print(f"  Metadata: {meta}")
    except wandb.errors.CommError as e:
        print(f"Failed to fetch artifact: {e}")
        print("Make sure the artifact exists and you have access.")
    finally:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model weights from W&B")
    parser.add_argument("--model", default="all",
                        choices=["tracknet", "yolo", "all"],
                        help="Which model weights to download (default: all)")
    args = parser.parse_args()
    fetch(args.model)

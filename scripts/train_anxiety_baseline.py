from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccac.baselines.anxiety_baseline import BaselineConfig, train_anxiety_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CCAC T1/T2/T3 -> T4 anxiety baseline")
    parser.add_argument("--dataset-path", type=str, default="datasets")
    parser.add_argument("--output-dir", type=str, default="artifacts/baselines/anxiety_wavlm_dinov2_small")
    parser.add_argument("--audio-feature-name", type=str, default="audio_wavlm_base")
    parser.add_argument("--video-feature-name", type=str, default="video_dinov2_small")
    parser.add_argument("--target-label-column", type=str, default="t4_anxiety_level")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--temporal-hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-power", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-feature-cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BaselineConfig(
        dataset_path=str(Path(args.dataset_path).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        audio_feature_name=args.audio_feature_name,
        video_feature_name=args.video_feature_name,
        target_label_column=args.target_label_column,
        hidden_dim=args.hidden_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weight_power=args.class_weight_power,
        label_smoothing=args.label_smoothing,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        num_folds=args.num_folds,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        feature_cache=not args.no_feature_cache,
    )
    result = train_anxiety_baseline(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

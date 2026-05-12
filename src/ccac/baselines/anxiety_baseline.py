from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset


STAGES = ("T1", "T2", "T3")
CLIP_TYPES = ("A01", "B01", "B02", "B03")
TARGET_LABEL_COLUMN = "t4_anxiety_level"


@dataclass(frozen=True)
class BaselineConfig:
    dataset_path: str
    output_dir: str
    audio_feature_name: str = "audio_wavlm_base"
    video_feature_name: str = "video_dinov2_small"
    target_label_column: str = TARGET_LABEL_COLUMN
    subject_id_column: str = "subject_id"
    hidden_dim: int = 256
    temporal_hidden_dim: int = 192
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    class_weight_power: float = 1.0
    label_smoothing: float = 0.0
    batch_size: int = 32
    epochs: int = 80
    patience: int = 12
    num_folds: int = 5
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    feature_cache: bool = True


class LongitudinalDataset(Dataset):
    def __init__(self, features: np.ndarray, mask: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.mask = torch.from_numpy(mask).bool()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[index], self.mask[index], self.labels[index]


class LongitudinalAnxietyModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, temporal_hidden_dim: int, dropout: float):
        super().__init__()
        self.clip_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.clip_attention = nn.Linear(hidden_dim, 1)
        self.stage_position = nn.Parameter(torch.zeros(len(STAGES), hidden_dim))
        nn.init.normal_(self.stage_position, mean=0.0, std=0.02)
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        fusion_dim = hidden_dim * 2 + temporal_hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor, clip_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.clip_encoder(inputs)
        stage_repr = self._pool_stage(encoded, clip_mask)
        stage_repr = stage_repr + self.stage_position.unsqueeze(0)
        temporal_out, _ = self.temporal_encoder(stage_repr)
        pooled_temporal = temporal_out.mean(dim=1)
        pooled_stage = stage_repr.mean(dim=1)
        final_stage = stage_repr[:, -1, :]
        fused = torch.cat([pooled_temporal, pooled_stage, final_stage], dim=-1)
        return self.classifier(fused)

    def _pool_stage(self, encoded: torch.Tensor, clip_mask: torch.Tensor) -> torch.Tensor:
        logits = self.clip_attention(encoded).squeeze(-1)
        logits = logits.masked_fill(~clip_mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * clip_mask.float()
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        stage_repr = (encoded * weights.unsqueeze(-1)).sum(dim=2)
        missing_stage = clip_mask.sum(dim=-1, keepdim=True) == 0
        return torch.where(missing_stage, torch.zeros_like(stage_repr), stage_repr)


class BaselineFeatureBuilder:
    def __init__(self, audio_feature_name: str, video_feature_name: str):
        self.audio_feature_name = audio_feature_name
        self.video_feature_name = video_feature_name
        self.audio_columns: dict[tuple[str, str], list[str]] = {}
        self.video_columns: dict[tuple[str, str], list[str]] = {}
        self.audio_dim = 0
        self.video_dim = 0

    def fit(self, frame: pd.DataFrame) -> "BaselineFeatureBuilder":
        self.audio_columns = {
            (stage, clip): _find_feature_columns(frame, stage, clip, self.audio_feature_name)
            for stage in STAGES
            for clip in CLIP_TYPES
        }
        self.video_columns = {
            (stage, clip): _find_feature_columns(frame, stage, clip, self.video_feature_name)
            for stage in STAGES
            for clip in CLIP_TYPES
        }
        self.audio_dim = max((len(columns) for columns in self.audio_columns.values()), default=0)
        self.video_dim = max((len(columns) for columns in self.video_columns.values()), default=0)
        if self.audio_dim <= 0:
            raise ValueError(f"no columns found for audio feature: {self.audio_feature_name}")
        if self.video_dim <= 0:
            raise ValueError(f"no columns found for video feature: {self.video_feature_name}")
        return self

    @property
    def input_dim(self) -> int:
        return self.audio_dim + self.video_dim

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        features = np.zeros((len(frame), len(STAGES), len(CLIP_TYPES), self.input_dim), dtype=np.float32)
        clip_mask = np.zeros((len(frame), len(STAGES), len(CLIP_TYPES)), dtype=bool)
        for stage_index, stage in enumerate(STAGES):
            for clip_index, clip_type in enumerate(CLIP_TYPES):
                audio_block, audio_present = self._extract_block(frame, self.audio_columns[(stage, clip_type)], self.audio_dim)
                video_block, video_present = self._extract_block(frame, self.video_columns[(stage, clip_type)], self.video_dim)
                features[:, stage_index, clip_index, : self.audio_dim] = audio_block
                features[:, stage_index, clip_index, self.audio_dim :] = video_block
                clip_mask[:, stage_index, clip_index] = audio_present | video_present
        return features, clip_mask

    @staticmethod
    def _extract_block(frame: pd.DataFrame, columns: list[str], target_dim: int) -> tuple[np.ndarray, np.ndarray]:
        block = np.zeros((len(frame), target_dim), dtype=np.float32)
        present = np.zeros(len(frame), dtype=bool)
        if not columns:
            return block, present
        values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        block[:, : values.shape[1]] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        present = np.isfinite(values).any(axis=1)
        return block, present


def train_anxiety_baseline(config: BaselineConfig) -> dict[str, Any]:
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(config.dataset_path)
    if _is_release_dataset(dataset_path):
        frame, features, clip_mask, label_mapping, input_dim = _load_release_train_val(config, dataset_path)
        labels = frame["_label_index"].to_numpy(dtype=np.int64)
    else:
        frame = pd.read_csv(config.dataset_path, dtype={config.subject_id_column: str})
        frame = frame.dropna(subset=[config.target_label_column]).reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"dataset has no rows with target label: {config.target_label_column}")
        labels, label_mapping = _encode_labels(frame[config.target_label_column])
        builder = BaselineFeatureBuilder(config.audio_feature_name, config.video_feature_name).fit(frame)
        features, clip_mask = builder.transform(frame)
        input_dim = builder.input_dim
    fold_indices = _build_folds(labels, config.num_folds, config.seed)

    oof_probabilities = np.zeros((len(frame), len(label_mapping)), dtype=np.float32)
    oof_predictions = np.full(len(frame), fill_value=-1, dtype=np.int64)
    metrics: list[dict[str, Any]] = []
    fold_states: list[dict[str, Any]] = []

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        scaler = _fit_scaler(features[train_idx], clip_mask[train_idx])
        train_features = _apply_scaler(features[train_idx], scaler)
        val_features = _apply_scaler(features[val_idx], scaler)
        train_dataset = LongitudinalDataset(train_features, clip_mask[train_idx], labels[train_idx])
        val_dataset = LongitudinalDataset(val_features, clip_mask[val_idx], labels[val_idx])
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        model = LongitudinalAnxietyModel(
            input_dim=input_dim,
            num_classes=len(label_mapping),
            hidden_dim=config.hidden_dim,
            temporal_hidden_dim=config.temporal_hidden_dim,
            dropout=config.dropout,
        ).to(device)
        class_weights = _class_weights(labels[train_idx], len(label_mapping), config.class_weight_power).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        best_state = None
        best_metric = -math.inf
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, config.epochs + 1):
            _train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics, probabilities = _evaluate(model, val_loader, criterion, device, len(label_mapping))
            macro_f1 = float(val_metrics["macro_f1"])
            if macro_f1 > best_metric:
                best_metric = macro_f1
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    "model": model.state_dict(),
                    "scaler_mean": scaler[0].tolist(),
                    "scaler_std": scaler[1].tolist(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                }
                torch.save(best_state, fold_dir / "best_model.pt")
                np.save(fold_dir / "val_probabilities.npy", probabilities)
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

        if best_state is None:
            raise RuntimeError(f"fold {fold_id} failed to produce a valid checkpoint")

        model.load_state_dict(best_state["model"])
        val_metrics, probabilities = _evaluate(model, val_loader, criterion, device, len(label_mapping))
        predictions = probabilities.argmax(axis=1)
        oof_probabilities[val_idx] = probabilities
        oof_predictions[val_idx] = predictions
        fold_metric = {
            "fold": fold_id,
            "best_epoch": best_epoch,
            **val_metrics,
        }
        metrics.append(fold_metric)
        fold_states.append(best_state)
        (fold_dir / "metrics.json").write_text(json.dumps(fold_metric, ensure_ascii=False, indent=2), encoding="utf-8")

    label_by_index = {index: label for label, index in label_mapping.items()}
    oof_labels = np.asarray([label_by_index[int(index)] for index in oof_predictions], dtype=object)
    overall_metrics = _classification_metrics(labels, oof_predictions)
    metrics_df = pd.DataFrame(metrics)
    oof_df = pd.DataFrame(
        {
            config.subject_id_column: frame[config.subject_id_column].astype(str),
            "true_label": frame[config.target_label_column].astype(str),
            "pred_label": oof_labels,
        }
    )
    for class_index in range(len(label_mapping)):
        oof_df[f"prob_class_{class_index}"] = oof_probabilities[:, class_index]

    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "baseline_config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "feature_input_dim": input_dim,
                "label_mapping": label_mapping,
                "fold_metrics_mean": metrics_df.mean(numeric_only=True).to_dict(),
                "overall_oof_metrics": overall_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "classification_report.txt").write_text(
        classification_report(
            labels,
            oof_predictions,
            target_names=[label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])],
            zero_division=0,
        ),
        encoding="utf-8",
    )
    if _is_release_dataset(dataset_path):
        _write_release_test_predictions(config, dataset_path, output_dir, fold_states, label_mapping, input_dim, device)
    return {
        "feature_input_dim": input_dim,
        "label_mapping": label_mapping,
        "fold_metrics": metrics,
        "overall_oof_metrics": overall_metrics,
    }


def _is_release_dataset(path: Path) -> bool:
    return path.is_dir() and (path / "train_val" / "labels.csv").exists()


def _load_release_train_val(config: BaselineConfig, dataset_root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, int], int]:
    labels_path = dataset_root / "train_val" / "labels.csv"
    frame = pd.read_csv(labels_path)
    required = {"anon_school", "anon_class", "anon_person", config.target_label_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"release labels missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=[config.target_label_column]).reset_index(drop=True)
    frame[config.subject_id_column] = frame[["anon_school", "anon_class", "anon_person"]].agg("/".join, axis=1)
    labels, label_mapping = _encode_labels(frame[config.target_label_column])
    frame["_label_index"] = labels

    cache_path = _release_cache_path(dataset_root, config, split="train_val")
    if config.feature_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cached_subjects = cached["subjects"].astype(str).tolist()
        if cached_subjects == frame[config.subject_id_column].astype(str).tolist():
            return frame, cached["features"].astype(np.float32), cached["clip_mask"].astype(bool), label_mapping, int(cached["input_dim"])

    features, clip_mask, input_dim = _build_release_features(dataset_root, "train_val", frame, config.audio_feature_name, config.video_feature_name)
    if config.feature_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            subjects=frame[config.subject_id_column].astype(str).to_numpy(dtype=object),
            features=features,
            clip_mask=clip_mask,
            input_dim=np.asarray(input_dim, dtype=np.int64),
        )
    return frame, features, clip_mask, label_mapping, input_dim


def _write_release_test_predictions(
    config: BaselineConfig,
    dataset_root: Path,
    output_dir: Path,
    fold_states: list[dict[str, Any]],
    label_mapping: dict[str, int],
    input_dim: int,
    device: torch.device,
) -> None:
    test_path = dataset_root / "test" / "subjects.csv"
    if not test_path.exists() or not fold_states:
        return
    test_frame = pd.read_csv(test_path)
    if test_frame.empty:
        return
    test_frame[config.subject_id_column] = test_frame[["anon_school", "anon_class", "anon_person"]].agg("/".join, axis=1)
    cache_path = _release_cache_path(dataset_root, config, split="test")
    if config.feature_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cached_subjects = cached["subjects"].astype(str).tolist()
        if cached_subjects == test_frame[config.subject_id_column].astype(str).tolist():
            test_features = cached["features"].astype(np.float32)
            test_mask = cached["clip_mask"].astype(bool)
        else:
            test_features, test_mask, _ = _build_release_features(dataset_root, "test", test_frame, config.audio_feature_name, config.video_feature_name)
    else:
        test_features, test_mask, _ = _build_release_features(dataset_root, "test", test_frame, config.audio_feature_name, config.video_feature_name)
        if config.feature_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                subjects=test_frame[config.subject_id_column].astype(str).to_numpy(dtype=object),
                features=test_features,
                clip_mask=test_mask,
                input_dim=np.asarray(input_dim, dtype=np.int64),
            )

    probabilities = []
    num_classes = len(label_mapping)
    for state in fold_states:
        scaler = (np.asarray(state["scaler_mean"], dtype=np.float32), np.asarray(state["scaler_std"], dtype=np.float32))
        scaled = _apply_scaler(test_features, scaler)
        dataset = LongitudinalDataset(scaled, test_mask, np.zeros(len(test_frame), dtype=np.int64))
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        model = LongitudinalAnxietyModel(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=config.hidden_dim,
            temporal_hidden_dim=config.temporal_hidden_dim,
            dropout=config.dropout,
        ).to(device)
        model.load_state_dict(state["model"])
        probabilities.append(_predict_probabilities(model, loader, device, num_classes))
    ensemble = np.mean(np.stack(probabilities, axis=0), axis=0)
    label_by_index = {index: label for label, index in label_mapping.items()}
    pred_index = ensemble.argmax(axis=1)
    pred_label = [label_by_index[int(index)] for index in pred_index]
    output = test_frame[["anon_school", "anon_class", "anon_person", config.subject_id_column]].copy()
    output["pred_label"] = pred_label
    for class_index in range(num_classes):
        output[f"prob_class_{class_index}"] = ensemble[:, class_index]
    output.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")


def _release_cache_path(dataset_root: Path, config: BaselineConfig, split: str) -> Path:
    name = f"{config.audio_feature_name}__{config.video_feature_name}__{config.target_label_column}.npz"
    return dataset_root / "metadata" / "baseline_cache" / split / name


def _build_release_features(
    dataset_root: Path,
    split: str,
    frame: pd.DataFrame,
    audio_feature_name: str,
    video_feature_name: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    audio_dim = _infer_release_feature_dim(dataset_root, split, audio_feature_name, prefer_npy=True)
    video_dim = _infer_release_feature_dim(dataset_root, split, video_feature_name, prefer_npy=True)
    input_dim = audio_dim + video_dim
    features = np.zeros((len(frame), len(STAGES), len(CLIP_TYPES), input_dim), dtype=np.float32)
    clip_mask = np.zeros((len(frame), len(STAGES), len(CLIP_TYPES)), dtype=bool)
    for row_index, row in enumerate(frame.itertuples(index=False)):
        subject_parts = (row.anon_school, row.anon_class, row.anon_person)
        for stage_index, stage in enumerate(STAGES):
            for clip_index, clip_type in enumerate(CLIP_TYPES):
                audio = _load_release_vector(dataset_root, split, audio_feature_name, subject_parts, stage, clip_type, audio_dim)
                video = _load_release_vector(dataset_root, split, video_feature_name, subject_parts, stage, clip_type, video_dim)
                features[row_index, stage_index, clip_index, :audio_dim] = audio
                features[row_index, stage_index, clip_index, audio_dim:] = video
                clip_mask[row_index, stage_index, clip_index] = bool(np.isfinite(audio).any() or np.isfinite(video).any())
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), clip_mask, input_dim


def _infer_release_feature_dim(dataset_root: Path, split: str, feature_name: str, prefer_npy: bool) -> int:
    feature_root = dataset_root / split / feature_name
    if not feature_root.exists():
        raise ValueError(f"release feature not found: {feature_name}")
    pattern = "pooled.npy" if prefer_npy else "pooled.json"
    candidate = next(feature_root.rglob(pattern), None)
    if candidate is not None:
        return int(_read_feature_vector(candidate).shape[0])
    candidate = next(feature_root.rglob("pooled.json"), None)
    if candidate is None:
        raise ValueError(f"feature has no pooled.npy or pooled.json files: {feature_name}")
    return int(_read_feature_vector(candidate).shape[0])


def _load_release_vector(
    dataset_root: Path,
    split: str,
    feature_name: str,
    subject_parts: tuple[str, str, str],
    stage: str,
    clip_type: str,
    target_dim: int,
) -> np.ndarray:
    base = dataset_root / split / feature_name / subject_parts[0] / subject_parts[1] / subject_parts[2] / stage / clip_type
    npy_path = base / "pooled.npy"
    json_path = base / "pooled.json"
    if npy_path.exists():
        values = _read_feature_vector(npy_path)
    elif json_path.exists():
        values = _read_feature_vector(json_path)
    else:
        values = np.zeros(target_dim, dtype=np.float32)
    vector = np.zeros(target_dim, dtype=np.float32)
    length = min(target_dim, values.shape[0])
    vector[:length] = values[:length]
    return vector


def _read_feature_vector(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    values = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(values, dict):
        items = [values[key] for key in sorted(values)]
    elif isinstance(values, list):
        items = values
    else:
        raise ValueError(f"unsupported pooled feature JSON: {path}")
    return np.asarray([float(item) for item in items], dtype=np.float32).reshape(-1)


def _find_feature_columns(frame: pd.DataFrame, stage: str, clip_type: str, feature_name: str) -> list[str]:
    pattern = re.compile(rf"^{stage.lower()}_{clip_type.lower()}_{re.escape(feature_name)}_(\d+)$")
    matched: list[tuple[int, str]] = []
    for column in frame.columns:
        found = pattern.match(column)
        if found:
            matched.append((int(found.group(1)), column))
    matched.sort(key=lambda item: item[0])
    return [column for _, column in matched]


def _encode_labels(series: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    normalized = series.astype(str).str.strip()
    labels = sorted(normalized.unique().tolist())
    mapping = {label: index for index, label in enumerate(labels)}
    encoded = normalized.map(mapping).to_numpy(dtype=np.int64)
    return encoded, mapping


def _build_folds(labels: np.ndarray, num_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    num_samples = len(labels)
    if num_samples < 2:
        raise ValueError("baseline requires at least two labeled subjects")
    class_counts = pd.Series(labels).value_counts()
    max_folds = max(2, min(num_folds, num_samples))
    if len(class_counts) > 1 and class_counts.min() >= 2:
        fold_count = min(max_folds, int(class_counts.min()))
        splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(num_samples), labels))
    fold_count = min(max_folds, num_samples)
    splitter = KFold(n_splits=fold_count, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(num_samples)))


def _fit_scaler(features: np.ndarray, clip_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1])
    flat_mask = clip_mask.reshape(-1)
    if flat_mask.any():
        valid = flat[flat_mask]
    else:
        valid = flat
    mean = valid.mean(axis=0)
    std = valid.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_scaler(features: np.ndarray, scaler: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = scaler
    return ((features - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)).astype(np.float32)


def _class_weights(labels: np.ndarray, num_classes: int, power: float) -> torch.Tensor | None:
    if power <= 0:
        return None
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (num_classes * counts)
    weights = np.power(weights, power)
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    model.train()
    for batch_features, batch_mask, batch_labels in dataloader:
        batch_features = batch_features.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_features, batch_mask)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    losses: list[float] = []
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch_features, batch_mask, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_mask = batch_mask.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features, batch_mask)
            loss = criterion(logits, batch_labels)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            losses.append(float(loss.item()))
            all_labels.append(batch_labels.cpu().numpy())
            all_probabilities.append(probabilities)
    labels = np.concatenate(all_labels, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0) if all_probabilities else np.zeros((0, num_classes), dtype=np.float32)
    predictions = probabilities.argmax(axis=1) if len(probabilities) else np.zeros(0, dtype=np.int64)
    metrics = _classification_metrics(labels, predictions)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, probabilities


def _predict_probabilities(model: nn.Module, dataloader: DataLoader, device: torch.device, num_classes: int) -> np.ndarray:
    model.eval()
    all_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch_features, batch_mask, _ in dataloader:
            batch_features = batch_features.to(device)
            batch_mask = batch_mask.to(device)
            logits = model(batch_features, batch_mask)
            all_probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    if not all_probabilities:
        return np.zeros((0, num_classes), dtype=np.float32)
    return np.concatenate(all_probabilities, axis=0)


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


def _resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

import numpy as np
import pandas as pd

from ccac.baselines.anxiety_baseline import (
    BaselineFeatureBuilder,
    LongitudinalAnxietyModel,
    _build_folds,
)


def _toy_frame() -> pd.DataFrame:
    rows = []
    for index in range(6):
        row = {
            "subject_id": f"s{index}",
            "t4_anxiety_level": "low" if index % 2 == 0 else "high",
        }
        for stage in ("t1", "t2", "t3"):
            for clip in ("a01", "b01", "b02", "b03"):
                for dim in range(3):
                    row[f"{stage}_{clip}_audio_wavlm_base_{dim:04d}"] = float(index + dim)
                for dim in range(2):
                    row[f"{stage}_{clip}_video_dinov2_small_{dim:04d}"] = float(index - dim)
        rows.append(row)
    return pd.DataFrame(rows)


def test_feature_builder_produces_expected_tensor_shape() -> None:
    frame = _toy_frame()
    builder = BaselineFeatureBuilder("audio_wavlm_base", "video_dinov2_small").fit(frame)
    features, clip_mask = builder.transform(frame)

    assert features.shape == (6, 3, 4, 5)
    assert clip_mask.shape == (6, 3, 4)
    assert clip_mask.all()


def test_model_forward_shape_matches_class_count() -> None:
    frame = _toy_frame()
    builder = BaselineFeatureBuilder("audio_wavlm_base", "video_dinov2_small").fit(frame)
    features, clip_mask = builder.transform(frame)
    model = LongitudinalAnxietyModel(
        input_dim=builder.input_dim,
        num_classes=2,
        hidden_dim=16,
        temporal_hidden_dim=12,
        dropout=0.1,
    )
    logits = model(
        inputs=np_to_tensor(features[:2]),
        clip_mask=np_to_tensor(clip_mask[:2], is_bool=True),
    )
    assert tuple(logits.shape) == (2, 2)


def test_fold_builder_prefers_stratified_splits_when_possible() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    folds = _build_folds(labels, num_folds=5, seed=42)
    assert len(folds) == 3


def np_to_tensor(array: np.ndarray, is_bool: bool = False):
    import torch

    tensor = torch.from_numpy(array)
    return tensor.bool() if is_bool else tensor.float()

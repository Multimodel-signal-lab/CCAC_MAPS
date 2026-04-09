# CCAC_MAPS
Official Repository of Multimodal Adolescent Psychological States Challenge for CCAC

## Background
This challenge focuses on the tasks of "multimodal temporal evolution prediction" and long-term early warning for adolescent mental health. It aims to break through the limitations of traditional single-point state assessments 
through long-term sequence modeling. By doing so, this track hopes to overcome the scientific research challenges of long-term psychological evolution analysis and to identify high-risk, susceptible students with deteriorating 
tendencies as early as possible, thereby providing a solid scientific foundation for early intervention by schools and families. The competition is built upon a longitudinal tracking cohort of 2,000 adolescents spanning a 3-year, 
4-phase period, featuring continuous induced-paradigm audio-visual data and DASS scale data. It provides a unified evaluation benchmark and multimodal temporal data resources.

## Task Introduction
### Task Content
This task requires participating models to dynamically predict the future psychological conditions and emotional evolution trends of subjects within a long-term longitudinal tracking environment, utilizing their historical multimodal
data and scale labels. Specifically, models must perform temporal modeling on the induced-paradigm audio-visual features and psychological scale data of adolescents across the first three phases (T1, T2, and T3). By extracting deep, dynamic
psychological representations and evolutionary patterns, the model should accurately predict the individual's depression level in the final phase (T4), which in turn reflects their underlying emotional change trajectory.

### Input and Output
- **Input**: Chronologically ordered historical multimodal features and psychological assessment baselines of a single user. Specifically, this includes the subject's multimodal features (e.g., acoustic descriptors, visual facial features) as well as
the corresponding psychological scale data across the three historical phases: T1, T2, and T3.
- **Output**: The user's depression level in phase T4.

## Annotations：
### File Types
- A01：	"The North Wind and the Sun" standardized reading passage.
  - 有一回，北风跟太阳在那儿争论谁的本领大。说着说着，来了一个过路的，身上穿了一件厚袍子。他们俩就商量好了，说谁能先叫这个过路的把他的袍子脱下来，就算是他的本领大。北风就使劲吹起来，拼命地吹。可是，他吹得越厉害，那个人就把他的袍子裹得越紧。到末了儿，北风没辙了，只好就算了。一会儿，太阳出来一晒，那个人马上就把袍子脱了下来。所以，北风不得不承认，还是太阳比他的本领大。
- B01：Please describe how your day went yesterday.
  - 请描述一下，你昨天过的怎么样?
- B02：Please describe your happiest memory from the past week.
  - 请描述一下，现在回想最近一周最开心的记忆?
- B03：Please describe your saddest memory from the past week.
  - 请描述一下，现在回想最近一周最悲伤的记忆?

### Auxiliary Attributes
1. 家庭结构（Family structure）
   1. 1=核心家庭，Nuclear
   2. 2=大家庭, Extended
   3. 3=单亲家庭, Single-parent
   4. 4=重组家庭，Blended
   5. 5=隔代家庭，Skipped-generation
   6. 6=其他，Other
2. 是否是独生子女（Only child status）
   1. Whether the respondent is an only child (1：Yes/0：No).
3. 如非独生子女,是否感受到父母有所偏爱？（Parental favoritism	If not an only child）
   1. 1=偏爱兄弟姐妹，Favoring siblings
   2. 2=无偏爱，No favoritism
   3. 3=偏爱自己，Favoring self
4. 本学期相比上个学期，学习成绩变动情况（Academic performance change Compared with previous semester）: 
   1. 1=进步，Improved, 
   2. 2=退步，Declined, 
   3. 3=稳定，Stable.
5. 本学期相比上个学期，情绪变动情况（Emotional state change Compared with previous semester）: 
   1. 1=变好，Better
   2. 2=变差，Worse
   3. 3=无变化，No change

## Features Description
### 1. Audio
#### 1.1 `mel_mfcc`
- Contents:
  - Log-Mel spectrogram: 80 dims
  - MFCC: 13 dims
- Time resolution: 25 Hz
- `sequence.npz`:
  - `mel_features`: `(T, 80)`
  - `mfcc_features`: `(T, 13)`
  - `timestamps_ms`: `(T,)`
  - `valid_mask`: `(T,)`
- Field meanings:
  - `mel_features[:, 0:80]`: log energy over 80 Mel filter banks
  - `mfcc_features[:, 0:13]`: 13 MFCC coefficients
- `pooled` statistics:
  - `mean/std/p10/p50/p90` are computed separately for Mel and MFCC features
  - Theoretical total dimensionality: `80*5 + 13*5 = 465`
  - Column naming patterns:
    - `mel_00_mean ... mel_79_p90`
    - `mfcc_00_mean ... mfcc_12_p90`

#### 1.2 `vad`
- Extraction method: `webrtcvad`
- Time resolution: 25 Hz
- `sequence.npz`:
  - `features`: `(T, 1)`
- Single-dimension semantics:
  - Dimension 0: `vad_decision`
    - `1` indicates speech
    - `0` indicates silence / non-speech
- `pooled.json` fields:
  - `speech_ratio`
  - `total_speech_duration`
  - `total_silence_duration`
  - `num_speech_segments`
  - `num_silence_segments`
  - `pause_count`
  - `mean_pause_duration`
  - `max_pause_duration`
  - `long_pause_count`
  - Other outputs:
    - `segments.json`: start/end timestamps of speech and silence segments

#### 1.3 `egemaps`
- Extraction method: `openSMILE eGeMAPSv02 functionals`
- Total `pooled` dimensionality: `88`
- Description:
  - These 88 dimensions are standardized acoustic statistical features covering F0, loudness, spectral slope, spectral flux, formant-related statistics, and voiced/unvoiced segment statistics.
- Example column names:
  - `F0semitoneFrom27.5Hz_sma3nz_amean`
  - `F0semitoneFrom27.5Hz_sma3nz_stddevNorm`
  - `loudnessPeaksPerSec`
  - `VoicedSegmentsPerSec`
  - `MeanVoicedSegmentLengthSec`
  - `equivalentSoundLevel_dBp`

#### 1.4 `ssl_embed`
- Extraction method: final-layer hidden states from a speech self-supervised model, linearly interpolated and resampled to 25 Hz
- `sequence.npz`:
  - `features`: `(T, D)`
  - `timestamps_ms`: `(T,)`
  - `valid_mask`: `(T,)`
  - `embed_dim`: scalar
  - `model_name`: scalar
- Per-frame semantics:
  - Each frame is a `D`-dimensional speech representation vector produced by the corresponding pretrained model.
- `pooled` statistics:
  - `mean/std/p10/p50/p90` are computed for each embedding dimension
  - Total dimensionality: `5 * D`
  - Column naming pattern: `embed_0000_mean ... embed_(D-1)_p90`

Supported models and dimensions:

| model_tag | HuggingFace / model identifier | Per-frame dimension  |
|---|---|---:|
| `wavlm-base` | `microsoft/wavlm-base` | 768 |
| `chinese-hubert-base` | `TencentGameMate/chinese-hubert-base` | 768 |
| `chinese-hubert-large` | `TencentGameMate/chinese-hubert-large` | 1024 |
| `chinese-wav2vec2-base` | `TencentGameMate/chinese-wav2vec2-base` | 768 |
| `chinese-wav2vec2-large` | `TencentGameMate/chinese-wav2vec2-large` | 1024 |
| `wav2vec2-chinese-xlsr` | `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` | 1024 |

## 2. Video Features
### 2.1 `qc_stats`

- Pretrained dependency: indirectly depends on upstream `InsightFace buffalo_l` outputs (`face_meta`)
- Extraction method: quality statistics computed from `face_meta.parquet`
- `sequence.npz`:
  - `features`: `(T, 4)`
  - `feature_names = [quality_score, blur_score, brightness, det_score]`
- 4-D mapping:
  - Dimension 0: `quality_score`, overall quality score
  - Dimension 1: `blur_score`, blur/sharpness score
  - Dimension 2: `brightness`
  - Dimension 3: `det_score`, face detection confidence
- Main `pooled.json` fields:
  - `total_frames`
  - `detected_frames`
  - `detection_rate`
  - `valid_frames`
  - `valid_ratio`
  - `is_low_quality`
  - `blur_mean`, `blur_std`, `blur_min`, `blur_max`
  - `brightness_mean`, `brightness_std`
  - `quality_mean`, `quality_std`
  - `det_score_mean`, `det_score_std`
  - `face_area_ratio_mean`
  - `num_selected_frames`

### 2.2 `headpose_geom`

- Pretrained dependency: indirectly depends on 5-point landmarks from upstream `InsightFace buffalo_l`
- Extraction method: geometric approximation based on 5-point landmarks
- `sequence.npz`:
  - `features`: `(T, 5)`
  - `feature_names = [yaw, pitch, roll, ear_mean, mar]`
- 5-D mapping:
  - Dimension 0: `yaw`, left-right head rotation
  - Dimension 1: `pitch`, up-down head motion
  - Dimension 2: `roll`, head tilt
  - Dimension 3: `ear_mean`, mean EAR of both eyes, approximately reflecting eye openness
  - Dimension 4: `mar`, mouth opening ratio
- `pooled` statistics:
  - `mean/std/min/max` are computed for each of `yaw/pitch/roll/ear_mean/mar`
  - Plus `valid_ratio`
  - Total dimensionality: `5*4 + 1 = 21`

### 2.3 `face_behavior`

- Pretrained dependency: indirectly depends on upstream `InsightFace buffalo_l` outputs (`face_meta`)
- Extraction method: behavior statistics based on `ear / mar / yaw / pitch / quality_score`
- `sequence.npz`:
  - `features`: `(T, 5)`
- 5-D mapping:
  - Dimension 0: `ear`
  - Dimension 1: `mar`
  - Dimension 2: `yaw`
  - Dimension 3: `pitch`
  - Dimension 4: `quality_score`
- `pooled.parquet` statistical fields:
  - `blink_count`
  - `blink_rate_per_min`
  - `avg_blink_duration_frames`
  - `mouth_open_ratio`
  - `mouth_movement_std`
  - `speech_activity_ratio`
  - `gaze_stability_score`
  - `yaw_range`
  - `pitch_range`
  - `saccade_count`
  - `expression_variability`
  - `expression_change_count`
- Total dimensionality: `12`

### 2.4 `vad_agg`

- Pretrained model: no
- Extraction method: align and aggregate audio `vad` onto the video timeline
- `sequence.npz`:
  - `features`: `(T, 4)`
  - `feature_names = [speech_prob, speech_activity, local_speech_ratio, speech_transition]`
- 4-D mapping:
  - Dimension 0: `speech_prob`, aligned speech probability/intensity
  - Dimension 1: `speech_activity`, binary speech activity
  - Dimension 2: `local_speech_ratio`, local-window speech ratio
  - Dimension 3: `speech_transition`, frame-to-frame speech state transition intensity
- `pooled.parquet` statistical fields:
  - `speech_ratio`
  - `avg_speech_prob`
  - `speech_transitions`
  - `silence_ratio`
  - `local_speech_ratio_mean`
  - `local_speech_ratio_std`

### 2.5 `body_pose`

- Pretrained model: `MediaPipe PoseLandmarker`
  - Preferred: `pose_landmarker_full.task`
  - Fallback: `pose_landmarker_lite.task`
- `sequence.npz`:
  - `features`: `(T, 27)`
  - `landmark_names = [nose, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip]`
- 27-D mapping:
  - Each landmark outputs 3 dimensions: `(x, y, visibility)`
  - There are 9 upper-body landmarks, so total dimensionality is `27 = 9 * 3`
- Flattened order:
  - `nose_x, nose_y, nose_visibility`
  - `left_shoulder_x, left_shoulder_y, left_shoulder_visibility`
  - `right_shoulder_x, right_shoulder_y, right_shoulder_visibility`
  - `left_elbow_x, left_elbow_y, left_elbow_visibility`
  - `right_elbow_x, right_elbow_y, right_elbow_visibility`
  - `left_wrist_x, left_wrist_y, left_wrist_visibility`
  - `right_wrist_x, right_wrist_y, right_wrist_visibility`
  - `left_hip_x, left_hip_y, left_hip_visibility`
  - `right_hip_x, right_hip_y, right_hip_visibility`
- Main `pooled.json` fields:
  - `num_frames`
  - `num_valid_frames`
  - `valid_ratio`
  - `landmarks_mean`: 27 dimensions
  - `landmarks_std`: 27 dimensions

### 2.6 `global_motion`

- Pretrained model: no
- Extraction method: OpenCV Farneback optical flow + frame differencing
- `sequence.npz`:
  - `features`: `(T, 4)`
  - `feature_names = [flow_mag_mean, flow_mag_std, flow_angle_mean, frame_diff_mean]`
- 4-D mapping:
  - Dimension 0: `flow_mag_mean`, mean optical-flow magnitude
  - Dimension 1: `flow_mag_std`, std of optical-flow magnitude
  - Dimension 2: `flow_angle_mean`, mean optical-flow direction
  - Dimension 3: `frame_diff_mean`, mean pixel difference between adjacent frames
- Main `pooled.json` fields:
  - `num_frames`
  - `flow_magnitude_mean`
  - `flow_magnitude_std`
  - `flow_magnitude_max`
  - `frame_diff_mean`
  - `frame_diff_max`
  - `motion_energy`

### 2.7 `vision_ssl_embed`

- Pretrained model: yes
- Extraction method: aligned face images are fed into a pretrained vision model; `pooler_output` or `CLS token` is used
- `sequence.npz`:
  - `features`: `(T, D)`
  - `timestamps_ms`: `(T,)`
  - `valid_mask`: `(T,)`
  - `embed_dim`: scalar
  - `model_name`: scalar
- Per-frame semantics:
  - Each frame is a `D`-dimensional high-level visual representation vector encoding face appearance, pose, texture, and expression-related patterns. It does not correspond to a single handcrafted geometric variable.
- `pooled` statistics:
  - `mean/std/p10/p50/p90` are computed for each embedding dimension
  - Total dimensionality: `5 * D`
  - Column naming pattern: `embed_0000_mean ... embed_(D-1)_p90`

Supported models and dimensions:

| model_tag | HuggingFace / model identifier | Per-frame dimension |
|---|---|---:|
| `dinov2-small` | `facebook/dinov2-small` | 384 |
| `dinov2-base` | `facebook/dinov2-base` | 768 |
| `dinov2-large` | `facebook/dinov2-large` | 1024 |
| `vit-mae-base` | `facebook/vit-mae-base` | 768 |
| `vit-base-patch16-224` | `google/vit-base-patch16-224` | 768 |
| `siglip-base-patch16-224` | `google/siglip-base-patch16-224` | 768 |
| `siglip-so400m-patch14-384` | `google/siglip-so400m-patch14-384` | 1152 |
| `clip-vit-base-patch32` | `openai/clip-vit-base-patch32` | 768 |
| `clip-vit-large-patch14` | `openai/clip-vit-large-patch14` | 1024 |

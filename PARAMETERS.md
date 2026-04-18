# Pipeline Parameter Reference

All tunable constants live at the top of `main.py` (lines ~61–114) unless noted.
Resolution-dependent values are specified at **1080p** and auto-scale to other resolutions via `frame_w / 1920.0`.

---

## TrackNet Ball Detection

| Constant | Default | Effect |
|---|---|---|
| `TRACKNET_WEIGHTS` | `best_overall.pt` | Which checkpoint to load |
| `TRACKNET_CONF_THRESH` | `0.7` | Heatmap peak must exceed this to count as a detection. **Lower = more detections, more FP. Higher = fewer detections, more FN.** Optimal found via `--sweep-conf`. |
| `TRACKNET_BALL_RADIUS` | `15 px` | Half-size of the synthetic bbox built around TrackNet's centroid output. Purely visual. |

---

## Ball Validator — per-detection gates (`BallValidator.filter`)

Gates run in order. A detection is dropped the moment any gate fails.

### Gate 1 — Court region
| Constant | Default | Effect |
|---|---|---|
| `BALL_COURT_X_PAD` | `0.05` (5% of frame_w) | Horizontal slack beyond the detected court polygon edges. Increase if ball at sideline is wrongly rejected. |

### Gate 2 — Velocity-aware teleport
| Constant | Default | Effect |
|---|---|---|
| `BALL_MAX_TELEPORT_PX` | `200 px` | Base max movement per frame (at 1080p). Scales with current speed: `allowed = max(base, speed×2.5) × gap`. |
| `BALL_TELEPORT_GRACE` | `10 frames` | After this many frames without a detection, teleport check resets (ball is treated as new). |
| Cold-start absolute cap | `700 px` (hardcoded) | Max single cold-start jump allowed regardless of gap. |
| Cold-start per-frame cap | `350 px × gap` (hardcoded) | Per-frame cold-start budget — capped by the absolute limit above. |
| Cold-start speed threshold | `15 px/frame` (hardcoded) | If current speed is below this, a cold-start large jump is allowed (assumes track was lost, not teleport). |

### Gate 3 — Bbox size consistency
| Constant/Value | Default | Effect |
|---|---|---|
| Size ratio upper bound | `2.5×` (hardcoded) | Ball bbox diagonal can't be more than 2.5× the recent median. |
| Size ratio lower bound | `0.4×` (hardcoded) | Ball bbox diagonal can't be less than 0.4× the recent median. |
| Consecutive violations to reject | `2` (hardcoded) | Single-frame size anomaly (motion blur, occlusion) passes through; only reject on 2nd consecutive hit. |
| Recent size window | `10 frames` (hardcoded) | How many accepted bbox diagonals to keep for the median. |

### Gate 4 — Backboard constraint
| Value | Default | Effect |
|---|---|---|
| Near-hoop window | `30 frames` (hardcoded) | How long after the ball was near a hoop to apply the backboard constraint. |
| Hoop proximity margin | `8% of frame_w` (hardcoded in `_near_any_hoop`) | How close ball must be to a hoop centre to trigger the near-hoop state. |
| Above-hoop slack | `3% of frame_h` (hardcoded) | Ball must be this far above hoop_top to trigger the backboard side constraint. |

---

## Ball Validator — candidate selection & re-acquisition

### Gate 5 — Best-candidate scoring (`_score`)
Selects the single best detection when multiple candidates exist (currently only relevant once top-K TrackNet is added).

| Value | Default | Effect |
|---|---|---|
| Trajectory min speed | `5 px/frame × scale` (hardcoded) | Below this speed, trajectory alignment is ignored and raw confidence wins. |
| Alignment weight | `0.5 + 0.5 × alignment` (hardcoded) | Score = conf × (0.5–1.0). At perfect alignment score = conf; at worst alignment score = 0.5×conf. |

### Gate 6 — Re-acquisition trajectory coherence
| Constant/Value | Default | Effect |
|---|---|---|
| `BALL_VELOCITY_HISTORY` | `5 frames` | Frames of per-frame (vx,vy) deltas to average for trajectory prediction. |
| Re-acquisition tolerance | `max_teleport × max(1.5, gap×1.0)` (hardcoded) | How far the detection can be from the projected position. Increase multiplier to accept more rebounds/deflections. |
| Spatial clustering fallback speed | `5 px/frame` (hardcoded) | When speed is below this (no velocity info), re-acquisition uses spatial cluster vote instead of trajectory. |

### Gate 7 — Wrist proximity (stationary balls)
| Constant/Value | Default | Effect |
|---|---|---|
| Stationary speed threshold | `8 px/frame × scale` (hardcoded) | Gate only fires when ball speed is below this. Fast-moving balls skip the wrist check. |
| `WRIST_PROX` | `80 px × scale` (hardcoded in `_near_any_wrist`) | Ball must be within this radius of a visible wrist. Only fires when at least one wrist is detectable. |
| `KP_CONF_THRESH` | `0.30` | Minimum keypoint confidence to consider a wrist "visible". |

### Gate 8 — Head-lock rejection
| Constant | Default | Effect |
|---|---|---|
| `BALL_HEADLOCK_WINDOW` | `20 frames` | History window to evaluate ball-to-head offset stability. |
| `BALL_HEADLOCK_MAX_VAR` | `6.0 px` | Max std-dev of ball-head offset to classify as "locked". Decrease to catch subtler locks; increase to only catch rigid locks. |
| `BALL_HEADLOCK_PROX` | `80 px` | Ball must be within this distance of a head keypoint to start tracking offset. |
| `BALL_HEADLOCK_HEAD_KPS` | `(0,1,2,3,4)` | COCO keypoints checked: nose, eyes, ears. |
| `HEADLOCK_PERSIST_FRAMES` | `5 frames` | Consecutive head-lock hits before track state is fully cleared. Single hits are soft-rejected (drop detection, keep state). |

---

## Ball Tracker — Kalman filter (`BallTracker`)

All values are hardcoded inside `BallTracker.__init__`.

| Value | Default | Effect |
|---|---|---|
| `_kf_R` (measurement noise) | `15.0 × I₂` | How much to trust TrackNet's centroid vs the KF prediction. **Higher = smoother but laggier; lower = more responsive but jittery.** |
| `_kf_Q` (process noise) | `diag(0.5, 0.5, 4.0, 4.0)` | Uncertainty added per step. Position noise 0.5 (model position is reliable); velocity noise 4.0 (ball accelerates often). Higher velocity noise = KF trusts measurements more during acceleration. |
| `_kf_P` (initial covariance) | `100 × I₄` | Confidence in initial state. High value = first few frames trust measurements more than prediction. |
| KF reset multiplier | `2.5 × kf_speed` (hardcoded) | KF resets if new measurement is more than `max(MAX_TELEPORT, speed×2.5)` from prediction. |
| `_VEL_HISTORY` | `5 frames` | Frames of velocity stored by tracker (separate from validator's velocity history). |

---

## Ball Interpolator (`BallInterpolator`)

| Constant | Default | Effect |
|---|---|---|
| `BALL_INTERP_BUFFER` | `20 frames` | Look-ahead window (~333ms at 60fps). Interpolator waits this long before finalising output so it can see future anchors. **Increase for longer look-ahead; increases latency.** |
| `BALL_INTERP_MAX_GAP` | `8 frames` | Max consecutive missing frames to fill. Gaps larger than this are left empty. |
| `STABLE_WINDOW` | `8 frames` (inside class) | Frames to look back/forward when checking endpoint stability. |
| `MIN_STABLE` | `2 of 8 frames` (inside class) | How many frames in the window must agree (within STABLE_RADIUS) to consider an endpoint stable. Lower = interpolates more aggressively. |
| `STABLE_RADIUS` | `180 px` at 1080p (inside class) | Max spread within a stable cluster. Increase if valid anchors are being rejected due to slight jitter. |
| Arc interpolation threshold | `gap > 3 frames` (hardcoded) | Gaps above 3 frames use parabolic arc fill; shorter gaps use linear. |

---

## Ball Re-ID (`BALL_REID_*`)

| Constant | Default | Effect |
|---|---|---|
| `BALL_REID_BUFFER` | `240 frames` (~4s at 60fps) | How long a lost ball track is remembered before being discarded. |
| `BALL_REID_DIST` | `500 px` | Max pixel distance to re-associate a returning ball to its old track. |

---

## Player / Referee Tracking

| Constant | Default | Effect |
|---|---|---|
| `REID_BUFFER_FRAMES` | `600 frames` (~10s) | How long a lost player track is remembered. |
| `REID_DIST_THRESH` | `300 px` | Max distance for player re-association. |
| `IOU_CRASH_THRESH` | `0.40` | IoU above this between two player boxes triggers crash/merge handling. |
| `VEL_HISTORY_LEN` | `15 frames` | Velocity history for player motion prediction. |
| `VEL_EXIT_THRESH` | `80 px` | Player exit velocity threshold. |

---

## Pose / Keypoints

| Constant | Default | Effect |
|---|---|---|
| `KP_CONF_THRESH` | `0.30` | Minimum YOLO-Pose keypoint confidence for any keypoint to be used (wrist check, head-lock, skeleton draw). |

---

## OCR (Jersey Numbers)

| Constant | Default | Effect |
|---|---|---|
| `OCR_INTERVAL` | `30 frames` | Run EasyOCR every N frames. Lower = more accurate but slower. |
| `OCR_CONFIRM_COUNT` | `2 reads` | Consistent reads needed before locking a jersey number to a player ID. |

---

## Court Detection

| Constant | Default | Effect |
|---|---|---|
| `COURT_MIN_LINES` | `4` | Minimum Hough lines to consider court visible. |
| `COURT_WOOD_RATIO` | `0.15` | Minimum fraction of wood-coloured (tan/brown) pixels required. |

---

## Quick-reference: most impactful for ball annotation quality

| Problem | Parameter to adjust |
|---|---|
| Too few detections / ball disappears | Lower `TRACKNET_CONF_THRESH` |
| Too many false positives (faces, crowds) | Raise `TRACKNET_CONF_THRESH` |
| Ball drops out after fast pass/shot | Lower `BALL_MAX_TELEPORT_PX` or raise re-acquisition tolerance multiplier (`1.5` → `2.0`) |
| Ball flickers in/out on stationary holds | Lower `BALL_TEMPORAL_MIN` (1 instead of 2) or raise `BALL_TEMPORAL_RADIUS` |
| Interpolation doesn't fill gaps | Lower `MIN_STABLE` (1) or raise `STABLE_RADIUS` / `BALL_INTERP_MAX_GAP` |
| Box jitters frame-to-frame | Raise `_kf_R` (trust KF more) |
| Box lags behind fast ball | Lower `_kf_R` (trust measurements more) |
| Ball locked on bald head | Lower `BALL_HEADLOCK_MAX_VAR` or `BALL_HEADLOCK_PROX` |
| Wrist check wrongly drops held ball | Raise `WRIST_PROX` or lower `KP_CONF_THRESH` |

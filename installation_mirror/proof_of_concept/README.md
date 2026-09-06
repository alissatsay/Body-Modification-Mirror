# Proof of Concept: Hand-Gesture Gain Control

Early pilot experiment that took the clinical warp-mirror pipeline (see
`../../clinical_mirror/`) and asked one question: instead of changing the
warp strength with keyboard presets, can a person control it by moving
their hands apart or together in front of the mirror? This was the first
gesture-control prototype and it directly informed the gesture-driven
interaction model used in the full installation (`../run_installation.py`).

## How it works

Hold both hands open and still in front of the camera for about 2 seconds
to "arm" gesture control (a ring around each palm fills up to show the
charge progress). Once armed, moving your hands apart or together adjusts
the warp gain (`uGain`) live. Stop moving for a moment and it disarms
back to idle.

## Files

- **`hand_gesture_gain_control.py`** — the primary version. Full palm-circle
  UI (idle/active color states, charge-up ring), two-open-hands arming
  gesture, fullscreen deployment on a second display.
- **`hand_gesture_gain_control_optimized.py`** — a performance-tuned variant
  of the same pilot: processes at a lower internal resolution, precomputes
  and reuses warp-map buffers instead of rebuilding them every frame,
  decimates the pose/hands/segmentation models (runs them every N frames
  and reuses the last result), and uses uint8 bitwise compositing instead
  of float32 blending. It also prints a rolling FPS/percentile breakdown
  every ~2 seconds — useful for tuning frame rate on real hardware. Kept
  separately because it demonstrates a genuinely different technique, not
  because it's a duplicate.

Both files are self-contained (they don't import from `clinical_mirror/` or
`../helpers/`) since they're historical snapshots of an experiment rather
than part of the live pipeline — the intent is to preserve them as
reference/demo material without wiring them into anything that could break
if the shared helpers change later.

## Demoing on a vertical vs. horizontal screen

Both files have an `ORIENTATION` flag near the top:

```python
ORIENTATION = "vertical"  # "vertical" or "horizontal"
```

Setting it to `"vertical"` rotates the camera feed 90° and outputs a
portrait frame (matching a rotated/vertical display); `"horizontal"` skips
the rotation and outputs a landscape frame. Everything else (gesture
detection, warp behavior, UI) is unaffected by this flag.

## What was consolidated away

This started as 6 near-duplicate scripts in `New_code/`
(`GLSLwarpCombinationBackground.py`, `GLSL_CB_rotated_HT.py`, `GLSL_HT.py`,
`GLSL_HT_Stats.py`, `GLSL_HT_Stats_Optimized.py`, `GLSL_HT_UI`), each one
commit further along the same experiment (confirmed via `git log --follow`
history and function-set diffing):

1. `GLSLwarpCombinationBackground.py` — the base warp-mirror script, no
   rotation, no hand tracking.
2. `GLSL_CB_rotated_HT.py` — adds a fixed 90° rotation only (despite the
   "HT" in the name, it has no hand-tracking code yet).
3. `GLSL_HT.py` — adds configurable rotation plus the first working
   hand-tracking gain control (a simple text banner, no palm UI).
4. `GLSL_HT_Stats.py` / `GLSL_HT_Stats_Optimized.py` — add the palm-circle
   UI and two-hand-open arming gesture, plus (in both) FPS/percentile
   instrumentation; `Stats_Optimized` additionally adds the performance
   techniques described above and fully supersedes `Stats.py`.
5. `GLSL_HT_UI` — the most recently touched of the six, with the same palm
   UI as the Stats files but without the profiling instrumentation, at a
   higher (4K) output resolution and with the deployment window/fullscreen
   setup.

`GLSL_HT_UI` and `GLSL_HT_Stats_Optimized.py` were the two that each added
something the other didn't (full UI + deployment settings vs. performance
technique + profiling), so both were kept and renamed to the two files
above; the other four were strict subsets and were removed (still
recoverable from git history — nothing was force-deleted).

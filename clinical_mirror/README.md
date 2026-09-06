# Clinical Mirror

The camera-based digital mirror used for the HSRC-approved user study and the
ongoing pre-clinical work. MediaPipe tracks your pose and segments you from
the background; a Gaussian-weighted pixel remap then stretches a region of
the body (centered on the hips, tapering out) to simulate a change in body
shape in real time.

## Files

- **`live_mirror.py`** — the main version. Run this to demo or run a session.
- **`conditional_warp.py`** — an alternate technique: instead of warping the
  whole frame and compositing over a captured background, this blends the
  warp map itself against an identity map using the segmentation mask, so
  only the person gets warped and the real background stays live behind
  them. Kept separate because it's a genuinely different approach, not a
  mode of `live_mirror.py`.
- **`compare_original_vs_warped.py`** — a side-by-side (original vs. warped)
  view, for documentation/figures rather than running a session.
- **`helpers.py`** — the shared functions (warp math, pose measurements,
  segmentation compositing, capture utilities) that everything above and
  `dataset_generation/` are built on.
- **`dataset_generation/`** — offline scripts that capture a photo pair and
  batch-render it across a sweep of warp strengths, for building a dataset
  rather than running a live session. See below.

## Running live_mirror.py

From the repo root, with the virtual environment active:

```
source .venv/bin/activate
python clinical_mirror/live_mirror.py
```

It opens your webcam, asks you to step out of frame to capture a clean
background, then starts the mirror.

## Controls

| Key     | Action                                                        |
|---------|----------------------------------------------------------------|
| `+`/`-` | increase / decrease warp strength (uGain) by 0.05              |
| `0`–`6` | jump straight to a preset warp strength (0.00 – 0.60)           |
| `u`/`d` | nudge the warp's vertical center up/down, overriding pose tracking |
| `b`     | toggle background mode: still ↔ combination                    |
| `s`     | toggle frame-saving on/off                                      |
| `q`     | quit                                                            |

## Background modes

- **still** (default) — the warped person is composited over the one
  background frame captured at startup.
- **combination** — the top of the frame is the captured background; the
  bottom (below a line derived from your hand/arm position) shows the live,
  unwarped frame, so you can reveal the real background below a certain
  height. Set `INITIAL_BACKGROUND_MODE` at the top of the file, or toggle it
  live with `b`.

## Saving frames

Off by default — this carries over from earlier data-collection scripts,
not something a normal demo needs. Press `s` to turn it on; frames save
every `SAVE_INTERVAL_SEC` seconds (default 5) to `saved_frames/live_mirror/`.

## dataset_generation/

Four standalone scripts, each capturing a photo pair (or loading one from
disk) and rendering it across many warp strengths — for building a static
image dataset, not for running a live mirror session:

- **`generate_dynamic_dataset.py`** — captures live, sweeps warp strength,
  composites over the captured background via segmentation.
- **`generate_greenscreen_dataset.py`** — same sweep, but keys the person
  out with a green screen (HSV chroma-key) instead of segmentation.
- **`generate_nobackground_dataset.py`** — same sweep, no background
  compositing at all (raw warped frames only).
- **`generate_still_warp.py`** — loads an existing photo pair from disk and
  renders a single warp strength, rather than capturing live or sweeping.

Run any of these from the repo root, the same way as `live_mirror.py`.

## Not moved in here yet

`New_code/` still holds a separate, more recent lineage of scripts (portrait
rotation, stats/profiling) built for the actual pre-clinical study — that
hasn't been folded into this structure yet.

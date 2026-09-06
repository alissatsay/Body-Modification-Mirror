# Reshape Repeat — Installation Mirror

The interactive installation piece ("Reshape Repeat"): a full-body camera
mirror where the person's silhouette is triangulated into a mesh and
deformed with as-rigid-as-possible (ARAP) sculpting, driven by two-hand
gestures (pinch to grab and pull, open palm/peace sign for mode and
session control). This has been shown publicly as an exhibited
installation. It's a separate system from `../clinical_mirror/` (which does
a simpler Gaussian pixel-warp, not mesh sculpting, and is used for the
clinical study rather than exhibition), though the earlier gesture-control
prototype in `proof_of_concept/` did start from the clinical mirror's code.

## Files

- **`run_installation.py`** — the full pipeline: welcome screen, beauty-standard
  selection, background capture, pose+hand tracking, triangular mesh
  generation and ARAP deformation, pinch/drag/brush gesture sculpting,
  finish/brush/drag buttons, and a timed session (3 minutes by default,
  `SESSION_DURATION_SECONDS`). This is what gets demoed. Determined to be
  the true final version (of several `fixed_mesh_*` candidates in the old
  `New_code/` folder) via git history and a function-by-function diff — see
  "What was consolidated" below.
- **`helpers/`** — the shared building blocks `run_installation.py` is built
  on:
  - `triangle_mesh.py` — Delaunay triangulation, mesh warping/deformation
    (including the Gaussian and hip/abdomen deformation helpers), mesh
    drawing/debug utilities.
  - `pose_tracking.py` — hand-gesture detection (open palm, peace sign,
    pinch, with hold-duration trackers for each), segmentation, and
    pose/body bounding-box helpers.
  - `interaction_loop.py` — the main per-frame interaction loops (brush/drag
    sculpting, with and without ARAP, rotated and non-rotated variants).
  - `display.py` — output-frame rotation and fullscreen second-monitor
    window setup.
- **`assets/`** — image assets `run_installation.py` loads at runtime:
  `beauty_standard_images/` (the selectable body-standard overlays),
  `UI_gestures/` (gesture icons + the initialization-pose/welcome-text
  overlays), `buttons/` (finish/brush/drag button states), and
  `background_captures/` (where a captured background plate gets saved to
  and reloaded from at runtime — this one is written to, not just read).
- **`proof_of_concept/`** — the earlier hand-gesture-gain-control pilot,
  built on the clinical mirror rather than the mesh pipeline. See its own
  README.
- **`tests/manual_visual_checks.py`** — manual/visual sanity checks for the
  mesh helpers (not an automated test suite — run and inspect visually).

## Running run_installation.py

From the repo root, with the virtual environment active:

```
source .venv/bin/activate
python installation_mirror/run_installation.py
```

## What was consolidated

`New_code/` used to hold 7 candidate "final pipeline" files
(`fixed_mesh_full_code.py`, `_backup.py`, `_backup2.py`, `_sigpaper.py`,
`_fps_count.py`, `_pixel_remap.py`, and `code_hand_refine (1).py`) plus the
6-file `GLSL_HT*` gesture-control prototype family. Using `git log --follow`
commit history plus a function-by-function diff:

- **`fixed_mesh_full_code_backup.py`** was the true final version (98
  functions, the deepest and most recent commit chain, 6 functions found
  nowhere else) — this became `run_installation.py`.
- **`fixed_mesh_full_code_backup2.py`** (92 functions) was a strict subset
  of `backup.py` with zero unique functionality — removed.
- **`fixed_mesh_full_code.py`** (77 functions, despite the plain name) was
  actually one of the *earliest* snapshots — removed.
- **`fixed_mesh_full_code_sigpaper.py`** was the version used for an earlier
  SIGGRAPH submission; confirmed to have zero functions absent from
  `backup.py`, so it was removed (tagged `siggraph-paper-version` in git
  first, so that specific snapshot stays reachable by name).
- **`fixed_mesh_fps_count.py`** and **`fixed_mesh_pixel_remap.py`** — earlier
  experimental variants, superseded — removed.
- **`code_hand_refine (1).py`** — had 5 arm-specific mesh-refinement functions
  not present anywhere else (`refine_arms_only_once`,
  `refine_mesh_conforming_once`, `select_arm_refine_triangles`,
  `_sorted_edge`, `_triangle_split_pattern`). Reviewed and confirmed not
  needed — removed.

`New_code/` is now empty and no longer exists in the repo — everything that
was in it has either moved here (with import paths and asset paths updated
accordingly) or was removed as confirmed-redundant (all removals are
recoverable from git history; nothing was force-deleted).

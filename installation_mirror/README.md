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

- **`run_installation.py`** — now just the entry point: `main()` (the outer
  restart loop — welcome/home/selection screens, background capture,
  per-session initialization, then the interaction loop) and
  `PoseInferenceThread` (the background pose/hand/segmentation inference
  thread). Everything else that used to live in this 3469-line file has
  been sorted into `helpers/` below, grouped by what it actually does,
  with every call site updated to reference the right module. Determined
  to be the true final version (of several `fixed_mesh_*` candidates in the
  old `New_code/` folder) via git history and a function-by-function diff —
  see "What was consolidated" below.
- **`helpers/`** — the shared building blocks `run_installation.py` is built
  on:
  - `triangle_mesh.py` — Delaunay triangulation, mesh warping/deformation/
    reconstruction, the render-group layering + yaw-based render-order
    system, and the segment/render-group name tables and layering config
    that go with it.
  - `pose_tracking.py` — hand-gesture detection (open palm, peace sign,
    pinch, with hold-duration trackers for each), segmentation, pose/body
    bounding-box helpers, and the skeleton-frame/binding math the mesh is
    tracked to (torso frames, mask sampling, body-yaw estimation).
  - `screens.py` — the welcome, home, beauty-standard-selection and
    countdown screens, plus the drawing utilities they share (overlay
    compositing, timeout/countdown text, pinch-hint icons, the animated
    gradient background) and the beauty-standard state
    (`current_beauty_standard`) they own.
  - `interaction_loop.py` — the main per-frame interaction loops (brush/drag
    sculpting, with and without ARAP, rotated and non-rotated variants),
    including the skeleton-mesh version's full gesture/button/session-timer
    logic and its config (brush radius, button geometry, session length).
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

## `run_installation.py` → `helpers/` migration

The ~99 top-level functions and 74 module-level constants that used to sit
directly in `run_installation.py` have been sorted into the four `helpers/`
modules above (a new `screens.py` was added for the ~20 screen-drawing
functions, since they didn't fit any existing module). A few things worth
knowing if you're reading the diff:

- Cross-module calls are qualified (`TMh.`, `PTh.`, `Sc.`, `Lh.`, `Dh.`)
  exactly like the calls into `helpers/` already were from
  `run_installation.py` — nothing bare crosses a module boundary except
  through those aliases.
- `current_beauty_standard` is real shared mutable state (read and
  reassigned from both the selection screen and the interaction loop), so
  `screens.py` is now its sole owner and `interaction_loop.py` reads/writes
  it as `Sc.current_beauty_standard` rather than keeping its own copy.
- Path constants derived from `__file__` (`_ASSETS_DIR` and the paths built
  from it) are recomputed independently in each module that needs them,
  with an extra `".."` for anything under `helpers/`, rather than passed
  around — this avoids depending on import order.
- `OUTPUT_W`/`OUTPUT_H` are still owned by `run_installation.py`, which
  injects them into `helpers/screens.py` and `helpers/interaction_loop.py`
  right after importing (`Sc.OUTPUT_W = OUTPUT_W`, etc.).
- Every cross-module import (here and inside `helpers/*.py`'s own
  imports of each other) is `from helpers import triangle_mesh as TMh`
  rather than a bare `import triangle_mesh as TMh` — `helpers/` is a real
  package now (`helpers/__init__.py`), and `sys.path` gets the package's
  parent (`installation_mirror/`) rather than `helpers/` itself. Doing
  this everywhere at once (not just in `run_installation.py`) matters:
  mixing the two styles would load two separate copies of a module in
  the same process and silently break the shared state described above.
- The file previously had its own `rotate_frame()`, a byte-for-byte
  duplicate of `display.py`'s `rotate_frame_for_output()` but with a
  *different* default rotation (270°/ccw vs. 90°/ccw). It's been removed;
  every call site now calls `Dh.rotate_frame_for_output(..., deg=ROTATE_DEG,
  direction=ROTATE_DIR)` explicitly so the actual demo rotation didn't
  change.
- The old entry-point function (a bare call at the bottom of the file, no
  `if __name__ == "__main__":` guard) has been renamed to `main()` with a
  proper guard added.
- While tracing dependencies, found a pre-existing bug: `triangle_mesh.py`'s
  `compute_render_order_with_arm_override`/`_leg_override` reference
  `ARM_OVERLAP_TRIGGER_FRAC`/`ARM_FRONT_SCORE_DEADBAND`/
  `LEG_OVERLAP_TRIGGER_FRAC`/`LEG_FRONT_SCORE_DEADBAND` as globals, but
  those were only ever assigned as local variables inside the old
  `run_hand_brush_drag_arap_loop_skeleton` — never at module level. Calling
  either function would have raised a `NameError` even before this move,
  any time `SHOW_LAYERED_RENDER` was on. Added them as module constants in
  `triangle_mesh.py` with the same values (0.020/6.0/0.015/8.0) so the
  functions are callable; worth double-checking those are the values you
  actually want.

## Cleanup sweep

A follow-up pass looking for stale references, dead code, and missing
documentation across the whole repo. In `installation_mirror/`:

- Added a module-level docstring to `run_installation.py` and each
  pre-existing `helpers/*.py` file (`triangle_mesh.py`, `pose_tracking.py`,
  `interaction_loop.py`, `display.py`) summarizing what it owns —
  `screens.py` already had one from when it was created.
- Removed dead code found while reading through everything: an unused
  module-level `state = {...}` dict in `triangle_mesh.py` (every real
  caller already passes its own `interaction_state` dict as a parameter —
  this one was never read), a debug-only frame counter + `[BTN] ...` print
  and a bare per-frame `print("dx dy:", dx, dy)` in
  `interaction_loop.py`'s main loop, and a large (~320-line) unreachable
  block in `tests/manual_visual_checks.py` — an earlier `test_hand_brush_
  drag_live` definition (plus two helper functions only it used) that was
  silently shadowed by a second, later definition of the same name and so
  could never actually run; also cleaned up a stale commented-out
  `GLSL_HT_UI` reference and a duplicate `mp_selfie_segmentation`
  assignment in that same test file.
- `proof_of_concept/`'s two files each computed an index-finger-position
  value (`get_index_y_from_pose` → `index_y_raw`/`prev_indexY`) every
  frame that was never actually used anywhere downstream — removed in
  both files. Also corrected `proof_of_concept/README.md`'s description of
  the optimized file: it displays at 1080x1920/1920x1080, not just "a
  lower internal resolution" — a real (half-linear-resolution) difference
  from the primary file's 4K output, not only an internal processing
  detail.

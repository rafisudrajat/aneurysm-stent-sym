# Refactoring Roadmap — Double-Stent Virtual Stenting

This document is a concrete, phased plan to refactor the codebase so that it is
(a) cross-platform (Windows **and** Linux, identical behaviour), (b) maintainable
and testable, and (c) ready to extend for the double-stent workflow.

The roadmap is ordered so that each phase is independently shippable. Phase 1
makes the code run on Linux at all; later phases improve structure and quality.
You do not have to do everything at once.

---

## 1. Goals & non-goals

**Goals**
- One command runs the full single- or double-stent pipeline on Windows and Linux.
- No Windows-only assumptions (path separators, `.cmd`/`.bat`, `del /s`).
- A real Python package with a clean public API, instead of flat scripts that
  `from Utils import *`.
- Deterministic, reproducible runs driven by a validated config file.
- A test suite that protects the geometry/physics core during future changes.
- Reproducible environment that installs the same way on both OSes.

**Non-goals (for this refactor)**
- Changing the FVS physics/algorithm or numerical results. Refactor must be
  **behaviour-preserving** for the core simulation (validated by golden tests).
- Adding the CFD step itself — only making the framework ready to feed it.

---

## 2. Current-state assessment

### 2.1 Cross-platform blockers (must fix first)
| Issue | Location | Why it breaks Linux |
|---|---|---|
| Experiment number parsed with `dir_path.split('\\')[1].split()[1]` | `constructAneuGeom.py:34`, `constructInitFD.py:61`, `deployStent.py:28`, `mergeMesh.py:5` | Hardcoded Windows backslash. On Linux paths use `/`, so this returns the wrong value or throws — the **single biggest blocker**. |
| Orchestration is `runSym.cmd` + `script/*.bat` | repo root, `script/` | Batch files don't run on Linux/macOS. |
| Cleanup via `del /s` | `cleanSym.cmd` | Windows-only command. |
| Path building by string `+` (`dir_path+'/appSettings.json'`) | all 4 drivers | Works by luck; should use `pathlib`. |

### 2.2 Code-quality / correctness issues
- **Real bug:** `PyStenting.py:433` — `faces = np.append(3*np.ones((faces.shape[0],1), faces, axis=1).ravel())`
  has misplaced parentheses (compare the correct sibling at `:471`). The node
  branch of `render_strut` is broken as written.
- **Performance bug:** in `VirtualStenting.deploy` (`PyStenting.py:759-760`) the whole
  position arrays are copied (`p_prev = p.copy(); p = p_new.copy()`) **inside the
  per-node inner loop** → O(N²) copying per iteration.
- **Performance:** `connected_nodes` (`:326`) scans every line for every node →
  O(N²) adjacency build. Build the adjacency list in one pass over `lines`.
- **Performance:** `np.append` is used to grow arrays inside loops throughout
  `cylinder_mesh`, `pattern_wrap`, and every generator in `Utils.py` → repeated
  reallocation. Preallocate or collect-then-`np.concatenate`.
- **Duplication:** `cylinder_bound`, `conical_boundary`, `bent_tube`, `s_curve`,
  `rugged_cylinder` in `Utils.py` repeat the same circle/stack/faces/centerline
  boilerplate. Extract shared helpers.
- **Bug:** `selectPattern` returns `ps.semienterprise` (the function object, not a
  call) — `constructInitFD.py:32`.
- **Leftover debug:** `cylinder_bound` prints wrong variables when
  `get_inlet_outlet=True` (`Utils.py:98-100`).
- `from Utils import *` and `from PyStenting import rotate_layer` create circular,
  opaque coupling (`Utils.py:4`).
- **Fragile serialization:** the case is `pickle`d to `.obj` (`constructInitFD.py`,
  `deployStent.py`). Pickle is version/Python-specific and unsafe to load.
- One module mixes geometry, physics, stent patterns, strut rendering, and GIF
  output (`PyStenting.py`); module-level global `plotter` (`:888`).
- Pinned, ancient deps (`numpy==1.19.5`, `pyvista==0.33.2`); API drift already
  visible (`cell_arrays`→`cell_data`, commented at `:486`).
- No tests, no packaging metadata, no linting/formatting config.

---

## 3. Target architecture

```
aneurysm-stent-sym/
├── pyproject.toml            # packaging, deps, tool config (ruff/black/pytest)
├── README.md                 # cross-platform install + run
├── src/
│   └── stenting/
│       ├── __init__.py       # public API exports
│       ├── geometry/
│       │   ├── transforms.py     # rotate_layer + rotation helpers
│       │   ├── cylinder.py       # shared circle/stack/faces builders
│       │   ├── boundaries.py     # cylinder, conical, bent, s_curve, rugged
│       │   └── aneurysm.py       # aneu_geom
│       ├── stent/
│       │   ├── patterns.py       # helical/enterprise/honeycomb -> Pattern
│       │   ├── flow_diverter.py  # FlowDiverter (mesh build, adjacency)
│       │   └── render.py         # render_strut (strut inflation)
│       ├── centerline.py         # VascCenterline
│       ├── simulation.py         # VirtualStenting.deploy (FVS algorithm)
│       ├── io.py                 # mesh/case load+save, path helpers
│       ├── config.py             # typed config schema + validation
│       ├── pipeline.py           # orchestrates the full single/double-stent run
│       └── cli.py                # `stenting` command (argparse/typer)
├── experiments/
│   └── experiment_0/
│       ├── config.json           # was appSettings.json
│       ├── README.md
│       └── results/              # generated, git-ignored
├── tests/
│   ├── test_transforms.py
│   ├── test_patterns.py
│   ├── test_boundaries.py
│   ├── test_pipeline_smoke.py
│   └── data/golden/              # small golden meshes
└── scripts/
    ├── run.sh                    # thin Linux/macOS wrapper
    └── run.ps1                   # thin Windows wrapper
```

Key idea: **the pipeline becomes a Python module with a CLI**, not a chain of OS
batch files. The `.sh`/`.ps1` wrappers are one-liners that just call the CLI, so
there is a single source of truth.

---

## 4. Phased roadmap

### Phase 0 — Safety net & environment (prereq, ~0.5 day)
- [ ] Create a feature branch; do not refactor on `main`.
- [ ] Add `pyproject.toml` with current deps; **unpin to compatible ranges**
      (e.g. `numpy>=1.24,<2.0`, `pyvista>=0.43`) and verify the code still imports.
      Fix the `cell_arrays`→`cell_data` drift permanently.
- [ ] Add `ruff` + `black` config and a `pytest` skeleton.
- [ ] Capture **golden outputs**: run the existing pipeline once on a small config,
      save the resulting vessel/stent meshes under `tests/data/golden/`. These are
      the reference for "behaviour-preserving" in later phases.

### Phase 1 — Make it run on Linux (highest priority, ~1 day) ✅
Goal: the existing pipeline runs unchanged in behaviour on Linux via one command.
- [x] **Kill the path-parsing bug.** Added `"experiment_id"` to all
      `appSettings.json` files; all four drivers now read it via `_parse_config`.
      All four `split('\\')` sites removed.
- [x] Replace every manual path concat with `pathlib.Path`; `results/` dir
      created with `Path.mkdir(parents=True, exist_ok=True)` in each driver.
- [x] Added `run.py` — cross-platform orchestrator that calls all four
      `main()` functions in the correct order.
- [x] Added `scripts/run.sh` (Linux/macOS) and `scripts/run.ps1` (Windows),
      replacing `runSym.cmd` and `script/*.bat`. `--clean` flag in `run.py`
      replaces `cleanSym.cmd` using `shutil.rmtree`.
- [ ] **Gate:** golden test reproduces Phase 0 outputs on Linux.

### Phase 2 — Package the core (no behaviour change, ~2-3 days) ✅
Goal: turn flat scripts into the `src/stenting/` package in section 3.
- [x] Moved `rotate_layer` → `src/stenting/geometry/transforms.py`. Removed
      the `Utils → PyStenting` back-import.
- [x] Split `PyStenting.py`: `FlowDiverter` → `stent/flow_diverter.py`;
      `render_strut` → `stent/render.py` (**`:433` bug fixed** — corrected
      misplaced parenthesis in `np.ones` call); patterns → `stent/patterns.py`;
      `VascCenterline` → `centerline.py`; `VirtualStenting` → `simulation.py`;
      GIF helper → `io.py` (lazy `_plotter`, no module-level init).
- [x] Extracted shared ring/face builders from `Utils.py` into
      `geometry/cylinder.py` (`_make_ring`, `_build_faces`); rewrote the 5
      boundary generators in `geometry/boundaries.py` on top of those helpers.
      Removed debug prints from `cylinder_bound`.
- [x] Replaced all `from x import *` with explicit imports in driver scripts;
      defined `__all__` in every new module.  `PyStenting.py` and `Utils.py`
      kept as thin re-export shims for backward compatibility.
- [x] `pyproject.toml` updated to `where = ["src"]`; `import stenting` works.
- [ ] **Gate:** golden test reproduces Phase 0 outputs on Linux.

### Phase 3 — Config + CLI + pipeline (~2 days)
- [ ] Define a typed config schema in `config.py` (dataclasses or `pydantic`),
      with validation and clear error messages. Rename `appSettings.json` →
      `config.json`. Collapse the near-identical `inner`/`outer` duplication using
      a shared default + per-stent overrides.
- [ ] Implement `pipeline.py` with explicit steps:
      `build_geometry → build_stent(outer) → deploy(outer) → merge →
      build_stent(inner) → deploy(inner)`. Single- vs double-stent selected by config.
- [ ] Implement `cli.py` exposing one command, e.g.
      `stenting run --experiment experiments/experiment_0` and subcommands
      (`geometry`, `deploy`, `clean`) for partial runs. Register via
      `[project.scripts]` in `pyproject.toml`.
- [ ] Replace `pickle`/`.obj` case files with a portable format (save the meshes
      as `.vtp`/`.stl` and the scalar parameters as JSON; rebuild the case object
      from those). Removes the unsafe, version-locked pickle dependency.
- [ ] **Gate:** `stenting run` reproduces golden outputs on both OSes.

### Phase 4 — Performance & correctness (~2 days)
- [ ] Rewrite adjacency build as a single pass over `lines` (dict of sets).
- [ ] Move the array copies out of the inner node loop in `deploy`; vectorize the
      spring-force update over nodes where possible (operate on arrays, not Python
      loops over `connected[i]`).
- [ ] Replace `np.append`-in-loop with preallocation / `np.concatenate`.
- [ ] Add a fast contact-distance path (the `KDTree.query` per node per iteration
      is a hotspot — batch the queries).
- [ ] **Gate:** golden outputs unchanged within tolerance; record before/after
      timings in the PR.

### Phase 5 — Tests, docs, CI (~1-2 days)
- [ ] Unit tests: `rotate_layer` (rotation invariants), each pattern (node/line
      counts), each boundary generator (closed, watertight, expected bbox).
- [ ] Smoke test: tiny end-to-end double-stent run completes and writes all files.
- [ ] Golden regression test wired into CI.
- [ ] GitHub Actions matrix: `{ubuntu-latest, windows-latest} × {py3.10, 3.11}`
      running install + lint + tests. This is the real guarantee of cross-platform
      health.
- [ ] Rewrite `README.md`: OS-agnostic install (venv + `pip install -e .` or
      conda), single run command, config reference, output description.

---

## 5. Cross-platform checklist (apply throughout)
- Use `pathlib.Path`, never string concatenation or hardcoded `\` / `/`.
- Never parse meaning out of OS paths — pass IDs/values explicitly via config.
- No shelling out to OS-specific commands (`del`, `copy`); use `shutil`/`os`.
- File writes: pick formats portable across PyVista/VTK versions (`.vtp`, `.stl`,
  `.vtk`); avoid pickle.
- Keep one orchestration source of truth (the CLI); shell wrappers stay trivial.
- CI must run on `windows-latest` and `ubuntu-latest` from Phase 5 on.

## 6. Definition of done
- `pip install -e .` + `stenting run --experiment experiments/experiment_0`
  produces identical-within-tolerance results on Windows and Linux.
- Green CI on both OSes; golden regression tests pass.
- No `*.bat`/`*.cmd` required to run; `from x import *` and the pickle case files
  are gone; the four path-parsing bug sites are removed.

## 7. Suggested first PRs (smallest valuable slices)
1. **PR-1 (Phase 0+1):** pyproject + golden capture + fix path bug + `run.sh`/`run.ps1`.
   → Project runs on Linux. Biggest immediate win.
2. **PR-2 (Phase 2):** package split, kill wildcard imports, fix `:433`.
3. **PR-3 (Phase 3):** config schema + CLI + drop pickle.
4. **PR-4 (Phase 4+5):** perf pass + tests + CI matrix.

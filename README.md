<a name="readme-top"></a>
[![CI](https://github.com/rafisudrajat/aneurysm-stent-sym/actions/workflows/ci.yml/badge.svg)](https://github.com/rafisudrajat/aneurysm-stent-sym/actions/workflows/ci.yml)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555)](https://www.linkedin.com/in/muhammad-rafi-sudrajat/)

<br />
<div align="center">
  <a href="https://medik.tf.itb.ac.id/profil/">
    <img src="images/lab-logo.png" alt="Logo" width="80" height="80">
  </a>
  <h3 align="center">Double-Stent Virtual Stenting Simulation</h3>
  <p align="center">
    Fast Virtual Stenting (FVS) deployment simulation for single and double flow-diverter stent therapy of intracranial aneurysms.
    <br />
    <a href="https://ieeexplore.ieee.org/document/9624474">Published paper</a>
    ·
    <a href="https://github.com/rafisudrajat/aneurysm-stent-sym/issues">Report Bug</a>
    ·
    <a href="https://github.com/rafisudrajat/aneurysm-stent-sym/issues">Request Feature</a>
  </p>
</div>

---

## About

This project was initiated by [Narendra Kurnia Putra Ph.D.](https://sites.google.com/view/narendkurnia/home) and [Bonfilio Nainggolan](https://www.linkedin.com/in/bonfilio-nainggolan-12508415a/) for Bonfilio's undergraduate thesis (2021). The original work analysed single-stent flow-diverter therapy; the codebase is now being extended to double-stent CFD analysis.

The simulation uses the **Fast Virtual Stenting (FVS)** spring-relaxation algorithm: each stent node is attracted toward its fully-expanded target position by linear springs, while a KDTree proximity check prevents wall penetration.

---

## Getting started

### Prerequisites

- Python 3.10 or 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) **or** pip + venv

### Install with uv (recommended)

```bash
git clone https://github.com/rafisudrajat/aneurysm-stent-sym.git
cd aneurysm-stent-sym
uv sync                 # creates .venv and installs all dependencies
uv sync --extra dev     # also installs pytest, ruff, black
```

### Install with pip

```bash
git clone https://github.com/rafisudrajat/aneurysm-stent-sym.git
cd aneurysm-stent-sym
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

---

## How the code is organised

```
your terminal
    │
    ▼
python run.py "experiment/experiment 0"           ← entry point (run.py)
    │
    ▼
stenting/pipeline.py                              ← orchestrates the steps
    ├── build_geometry()   → geometry/boundaries.py, geometry/aneurysm.py
    ├── build_stent()      → stent/patterns.py, stent/flow_diverter.py
    ├── deploy_stent()     → simulation.py  (FVS spring-relaxation solver)
    └── merge_meshes()     → io.py
```

[`run.py`](run.py) at the repo root is the single entry point.
Everything else — pipeline steps, geometry builders, the FVS solver — is
called from there. You never need to touch the files inside `src/`.

---

## Running a simulation

Activate the virtual environment once, then use `python run.py`:

```bash
# Activate (Linux/macOS)
source .venv/bin/activate
# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Run the full double-stent pipeline (most common)
python run.py "experiment/experiment 0"

# Run only the outer stent (skip merge + inner)
python run.py "experiment/experiment 0" --single-stent

# Individual steps
python run.py "experiment/experiment 0" --geometry        # Step 1: vessel mesh only
python run.py "experiment/experiment 0" --deploy          # Steps 2-3+4+5-6: stents
python run.py "experiment/experiment 0" --deploy --pos outer  # outer stent only

# Clean generated results
python run.py "experiment/experiment 0" --clean

# Help
python run.py --help
```

The experiment directory must contain a `config.yaml`; see
[`experiment/experiment 0/config.yaml`](experiment/experiment%200/config.yaml)
for the fully annotated reference.

---

## Configuration

Each experiment lives in its own directory and is driven by `config.yaml`
(YAML is preferred over JSON because it supports inline comments):

```yaml
experiment_id: "0"

constructAneuGeom:
  aneu_geom_param:
    r: 1.5        # vessel radius (mm)
    h: 30         # vessel height (mm)
    hstent: 15    # stent landing zone height (mm)
    # ... other geometry params

# Fields in "defaults" are merged into both "inner" and "outer".
# Only the fields that differ between layers need to appear per-layer.
constructInitFD:
  defaults:
    stent:
      radius: 1.2   # crimped radius; expanded by FVS solver
      tcopy: 21
      hcopy: 28
  outer:
    stent: {offset_angle: 0}
  inner:
    stent: {offset_angle: 0.5}   # rotate inner stent to interleave struts

deployStent:
  defaults:
    deploy_param: {tol: 5.0e-5, max_iter: 700, OC: true}
  outer:
    deploy_param: {add_tol: 3.0e-3}
  inner:
    deploy_param: {add_tol: 5.5e-3}
```

See [`experiment/experiment 0/config.yaml`](experiment/experiment%200/config.yaml)
for the fully annotated reference. `config.json` is also accepted as a fallback.

---

## Running tests

```bash
# All tests (52 total: unit + golden regression)
PYTHONPATH="" uv run pytest

# With coverage
PYTHONPATH="" uv run pytest --cov=src/stenting

# Lint
uv run ruff check src/ tests/
```

> **Note:** `PYTHONPATH=""` clears any injected paths (e.g. from ROS) that can
> load broken pytest plugins.  Make it an alias: `alias pt='PYTHONPATH="" uv run pytest'`

---

## Project structure

```
src/stenting/
├── geometry/
│   ├── transforms.py   # rotate_layer
│   ├── cylinder.py     # shared ring/face builders
│   ├── boundaries.py   # cylinder, conical, bent_tube, s_curve, rugged_cylinder
│   └── aneurysm.py     # aneu_geom (vessel + sac)
├── stent/
│   ├── patterns.py     # helical, enterprise, honeycomb, semienterprise
│   ├── flow_diverter.py# FlowDiverter (wireframe mesh + adjacency)
│   └── render.py       # render_strut (strut inflation)
├── centerline.py       # VascCenterline
├── simulation.py       # VirtualStenting.deploy (FVS algorithm)
├── config.py           # typed config schema + load_config()
├── pipeline.py         # build_geometry, deploy_stent, merge_meshes, run
└── io.py               # frame() GIF helper
```

---

## CI

GitHub Actions runs `lint + test` on every push and pull request, across:

| OS | Python |
|---|---|
| ubuntu-latest | 3.10, 3.11 |
| windows-latest | 3.10, 3.11 |

---

## References

- B. Nainggolan et al., *"Flow Diverter Stent Simulation on Patient-Specific Intracranial Aneurysm"*, ICITEE 2021. [IEEE link](https://ieeexplore.ieee.org/document/9624474)
- M. Appanaboyina et al., *"Computational modelling of blood flow in side arterial branches after stenting"*, Int. J. Comput. Fluid Dyn., 2008.

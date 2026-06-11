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

## Running a simulation

```bash
# Run the full pipeline for an experiment
uv run stenting run --experiment experiments/experiment_0

# Individual steps
uv run stenting geometry --experiment experiments/experiment_0
uv run stenting deploy   --experiment experiments/experiment_0

# Clean generated results
uv run stenting clean    --experiment experiments/experiment_0

# Help
uv run stenting --help
```

The experiment directory must contain a `config.json`; see
`experiments/experiment_0/config.json` for an annotated example.

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
for the fully annotated reference. `config.json` and `appSettings.json` are also
accepted for backward compatibility.

---

## Running tests

```bash
# All tests (51 unit + 1 golden regression)
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
├── cli.py              # `stenting` CLI entry point
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

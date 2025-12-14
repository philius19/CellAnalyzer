# CellAnalyzer

Python package for quantitative 3D morphology analysis of cell surface reconstructions from Lattice Light Sheet Microscopy. Processes MATLAB-generated meshes (u-shape3D) to compute volume, surface area, and curvature statistics for immune cell morphological fingerprinting.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Data Flow](#data-flow)
- [Directory Structure](#directory-structure)
- [Usage Examples](#usage-examples)
- [Motion Data](#motion-data)
- [API Reference](#api-reference)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Description

CellAnalyzer enables quantitative morphological fingerprinting of cells from 3D fluorescence microscopy. Designed for Lattice Light Sheet Microscopy (LLSM) of murine primary myeloid immune cells (neutrophils, macrophages, dendritic cells), it bridges MATLAB's u-shape3D mesh generation pipeline with Python's scientific ecosystem.

**Key Capabilities:**
- **Morphology metrics**: Volume (µm³), surface area (µm²), curvature statistics
- **Time-lapse analysis**: Track morphological changes across timepoints
- **Quality control**: Automated mesh validation, topology checks
- **Visualization**: Publication-quality curvature maps and distributions

**Scientific Context**: Establishes cell type-specific morphological signatures by quantifying 3D surface geometry. Handles isotropic meshes (MATLAB resamples Z-dimension before marching cubes reconstruction).

---

## Installation

### Requirements

- Python 3.10+
- Core: `numpy`, `scipy`, `matplotlib`
- Mesh handling: `vedo`
- MATLAB I/O: `mat73` (v7.3 HDF5 support)

### Setup

```bash
cd /path/to/Pipeline/Scripts
pip install -e .
```

The `-e` flag installs in editable mode for development.

**Verify installation:**
```bash
python -c "from CellAnalyzer import load_cell; print('Installation successful')"
```

---

## Quick Start

```python
from CellAnalyzer import load_cell

# Load all frames from cell directory
cell = load_cell('/path/to/cell_directory', channel=1)

print(f"Loaded {cell.n_frames} frames")
print(f"Pixel size: {cell.pixel_size_xy_um:.3f} µm (XY)")

# Access single frame
frame = cell[1]
print(f"T1: {frame.n_vertices:,} vertices, {frame.n_faces:,} faces")

# Calculate statistics
stats = cell.calculate_statistics(1)
print(f"Volume: {stats.mesh_stats.volume_um3:.2f} µm³")
print(f"Surface Area: {stats.mesh_stats.surface_area_um2:.2f} µm²")
print(f"Mean Curvature: {stats.curvature_stats.mean:.4f} ± {stats.curvature_stats.std:.4f}")

# Visualize
fig, ax = cell.plot_frame(1, title="Frame 1 Curvature")
```

---

## Architecture

CellAnalyzer follows strict separation of concerns with specialized modules:

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                         │
│              from CellAnalyzer import load_cell             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   CellAnalyzer    │  ◄── Orchestrator
                 │   (analyzer.py)   │      (thin delegation)
                 └───┬───────┬───────┘
                     │       │
       ┌─────────────┼───────┼─────────────┐
       │             │       │             │
       ▼             ▼       ▼             ▼
 ┌──────────┐  ┌──────────┐  ┌───────────────┐
 │  io.py   │  │ analysis │  │ visualization │
 │          │  │   .py    │  │     .py       │
 │ Loading  │  │Statistics│  │   Plotting    │
 └────┬─────┘  └────┬─────┘  └───────┬───────┘
      │             │                │
      └─────────────┴────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ datatypes.py  │  ◄── Data structures
            │               │      (MeshFrame, CellData,
            │               │       AnalysisResults)
            └───────────────┘
```

**Module Responsibilities:**

| Module | Purpose |
|--------|---------|
| `analyzer.py` | CellAnalyzer orchestrator (delegates, no computation) |
| `io.py` | MATLAB file loading, metadata extraction |
| `analysis.py` | Pure statistical functions (volume, area, curvature) |
| `visualization.py` | Matplotlib plotting |
| `datatypes.py` | Dataclasses (MeshFrame, AnalysisResults, etc.) |
| `_validation.py` | Internal data validation |

---

## Data Flow

```
MATLAB (u-shape3D)                      Python (CellAnalyzer)
──────────────────                      ──────────────────────

Deconvolved TIFF Stack
        ↓
Three-Level Segmentation
 (inside / cell / surface)
        ↓
Z-Resampling (isotropic voxels)
        ↓
Marching Cubes (isovalue=0.999)
        ↓
    ┌─────────────────────────┐
    │ surface_1_*.mat         │
    │ meanCurvature_1_*.mat   │─────────►  load_cell(path)
    │ *_Decon.mat (metadata)  │                  ↓
    └─────────────────────────┘            CellAnalyzer
                                                 ↓
                                          ┌───────────┐
                                          │ MeshFrame │
                                          └─────┬─────┘
                                                │
                                    ┌───────────┼───────────┐
                                    │           │           │
                                    ▼           ▼           ▼
                              Statistics   Visualization  Quality
                              (analysis)   (visualization) Checks
```

**Key Points:**
- MATLAB performs Z-resampling → mesh coordinates are isotropic
- Volume calculation uses `pixel_size_xy³` (correct for isotropic mesh)
- All frames loaded upfront into memory (eager loading)

---

## Directory Structure

CellAnalyzer expects u-shape3D output with this structure:

```
cell_directory/
├── Cell01_Decon.mat                      # MovieData (metadata source)
├── metadata_export.mat                   # Python-friendly metadata (optional)
└── Morphology/
    └── Analysis/
        └── Mesh/
            └── ch1/                       # Channel 1 meshes
                ├── surface_1_1.mat        # T1: vertices + faces
                ├── surface_1_2.mat        # T2: vertices + faces
                ├── ...
                ├── meanCurvature_1_1.mat  # T1: curvature per face
                ├── meanCurvature_1_2.mat  # T2: curvature per face
                ├── ...
                ├── faceNormals_1_*.mat              # Optional
                ├── gaussCurvatureUnsmoothed_1_*.mat # Optional
                └── imageSurface_1_*.mat             # Optional
```

### File Contents

**surface_*.mat** (MATLAB v7.3 or legacy):
```
surface (struct)
├── vertices (N × 3 double)   % XYZ coordinates (isotropic pixels)
└── faces (M × 3 double)      % Triangle indices (1-indexed)
```

**meanCurvature_*.mat**:
```
meanCurvature (M × 1 double)  % Smoothed mean curvature per face
```

**MovieData file (*_Decon.mat)**:
- Pixel sizes (XY and Z in nanometers)
- Time interval
- Original image path
- Processing parameters

---

## Usage Examples

### 1. Single Frame Analysis

```python
from CellAnalyzer import load_cell

# Load cell data
cell = load_cell('/data/Cell01', channel=1)

# Access first frame
frame = cell[1]
print(f"Mesh: {frame.n_vertices} vertices, {frame.n_faces} faces")

# Calculate statistics
stats = cell.calculate_statistics(1)

# Geometry
print(f"Volume: {stats.mesh_stats.volume_um3:.2f} µm³")
print(f"Surface Area: {stats.mesh_stats.surface_area_um2:.2f} µm²")
print(f"Watertight: {stats.mesh_stats.is_watertight}")
print(f"Euler χ: {stats.mesh_stats.euler_number}")

# Curvature
curv = stats.curvature_stats
print(f"Mean curvature: {curv.mean:.4f} ± {curv.std:.4f}")
print(f"Median: {curv.median:.4f}")
print(f"Range: [{curv.min:.4f}, {curv.max:.4f}]")
```

### 2. Time-Lapse Iteration

```python
from CellAnalyzer import load_cell

cell = load_cell('/data/Cell01')

# Iterate through all frames
for time_idx, frame in cell:
    print(f"T{time_idx:02d}: {frame.n_vertices:,} vertices")

# Check if specific frame exists
if 10 in cell:
    frame_10 = cell[10]

# Get all time indices
print(f"Available frames: {cell.time_indices}")
```

### 3. Batch Statistics Calculation

```python
from CellAnalyzer import load_cell

cell = load_cell('/data/Cell01')

# Calculate statistics for all frames
all_stats = cell.calculate_all_statistics()

# Extract time series
volumes = [all_stats[t].mesh_stats.volume_um3 for t in cell.time_indices]
mean_curvatures = [all_stats[t].curvature_stats.mean for t in cell.time_indices]

print(f"Volume range: {min(volumes):.1f} - {max(volumes):.1f} µm³")
```

### 4. Visualization

```python
from CellAnalyzer import load_cell, basic_spatial_plot, publication_style
import matplotlib.pyplot as plt

cell = load_cell('/data/Cell01')

# Method 1: Via CellAnalyzer
fig, ax = cell.plot_frame(1, title="T1 Curvature Map")
plt.savefig('frame1_curvature.pdf', dpi=300, bbox_inches='tight')

# Method 2: Direct function call with publication style
with publication_style():
    frame = cell[1]
    fig, ax = basic_spatial_plot(frame, title="Membrane Curvature")
    plt.show()
```

### 5. Batch Processing Multiple Cells

```python
from pathlib import Path
from CellAnalyzer import load_cell
import pandas as pd

results = []
cell_dirs = Path('/data/experiment01/').glob('Cell*')

for cell_dir in sorted(cell_dirs):
    try:
        cell = load_cell(cell_dir, channel=1)
        stats = cell.calculate_statistics(1)

        results.append({
            'cell_id': cell_dir.name,
            'n_frames': cell.n_frames,
            'volume_um3': stats.mesh_stats.volume_um3,
            'surface_area_um2': stats.mesh_stats.surface_area_um2,
            'mean_curvature': stats.curvature_stats.mean,
            'curvature_std': stats.curvature_stats.std
        })
        print(f"✓ {cell_dir.name}: {cell.n_frames} frames")

    except Exception as e:
        print(f"✗ {cell_dir.name}: {e}")

df = pd.DataFrame(results)
df.to_csv('batch_morphology.csv', index=False)
print(f"\nProcessed {len(results)} cells")
```

### 6. Access Underlying Data with Motion

```python
from CellAnalyzer import load_cell

cell = load_cell('/data/Cell01', load_motion=True)

for t, frame in cell:
    if frame.motion and frame.motion.has_backward:
        print(f"T{t}: motion available")
```

### 7. Access Underlying Data

```python
from CellAnalyzer import load_cell

cell = load_cell('/data/Cell01')
frame = cell[1]

# Raw mesh data
vertices = frame.vertices       # (N, 3) ndarray
faces = frame.faces            # (M, 3) ndarray
curvature = frame.curvature    # (M,) ndarray

# vedo Mesh object (for advanced operations)
mesh = frame.mesh
print(f"Mesh volume: {mesh.volume():.2f} pixels³")
print(f"Mesh area: {mesh.area():.2f} pixels²")

# Metadata
metadata = cell.metadata
print(f"Pixel size XY: {metadata.pixel_size_xy_nm} nm")
print(f"Pixel size Z: {metadata.pixel_size_z_nm} nm")
print(f"Time interval: {metadata.time_interval_sec} s")
```

---

## Motion Data

CellAnalyzer supports surface motion analysis, quantifying membrane displacement between consecutive frames. Motion data measures how far each surface element (face) moves relative to the previous or next timepoint, enabling dynamic morphological characterization.

**Scientific Context**: Surface motion captures protrusion (positive values) and retraction (negative values) dynamics. For immune cells, this reveals migratory behavior, membrane ruffling, and pseudopod formation rates.

### Loading Motion Data

Motion data is integrated into the standard loading workflow via a single boolean parameter:

```python
from CellAnalyzer import load_cell

# Load cell with motion data
cell = load_cell('/path/to/cell', channel=1, load_motion=True)

# Access motion from any frame
frame = cell[10]
if frame.motion and frame.motion.has_backward:
    print(f"Motion range: [{frame.motion.backward.min():.2f}, {frame.motion.backward.max():.2f}] voxels")
```

**Requirements**: Motion data must be generated by u-shape3D's `meshMotion` process (`p.control.meshMotion = 1`).

### Motion Direction

| Direction | Description | Availability |
|-----------|-------------|--------------|
| `backward` | Displacement from previous frame | Frames 2–N |
| `forward` | Displacement to next frame | Frames 1–(N-1) |

- **Frame 1**: Forward motion only (no previous frame for comparison)
- **Frame N**: Backward motion only (no next frame for comparison)
- **Frames 2–(N-1)**: Both directions available

### Motion Values

Per-face displacement values in **voxels** (multiply by pixel size for physical units):

| Value | Interpretation |
|-------|----------------|
| Positive | Outward motion (protrusion) |
| Negative | Inward motion (retraction) |
| NaN | No comparison available |

### Motion Examples

#### Basic Motion Access

```python
from CellAnalyzer import load_cell
import numpy as np

cell = load_cell('/data/Cell01', load_motion=True)

for t, frame in cell:
    if frame.motion and frame.motion.has_backward:
        valid = ~np.isnan(frame.motion.backward)
        mean_motion = np.mean(np.abs(frame.motion.backward[valid]))
        print(f"T{t:02d}: mean |motion| = {mean_motion:.3f} voxels")
```

#### Motion Statistics

Motion statistics are automatically included when calculating frame statistics:

```python
from CellAnalyzer import load_cell

cell = load_cell('/data/Cell01', load_motion=True)
stats = cell.calculate_statistics(10)

if stats.motion_stats:
    print(f"Mean motion: {stats.motion_stats.mean:.3f} ± {stats.motion_stats.std:.3f} voxels")
    print(f"Active fraction: {stats.motion_stats.active_fraction:.1%}")
    print(f"Mean magnitude: {stats.motion_stats.mean_magnitude:.3f} voxels")
```

#### Time-Series Motion Analysis

```python
from CellAnalyzer import load_cell
import numpy as np

cell = load_cell('/data/Cell01', load_motion=True)

motion_magnitudes = []
for t, frame in cell:
    if frame.motion and frame.motion.has_backward:
        valid = ~np.isnan(frame.motion.backward)
        magnitude = np.mean(np.abs(frame.motion.backward[valid]))
        motion_magnitudes.append({'time': t, 'magnitude': magnitude})

print(f"Average motion magnitude: {np.mean([m['magnitude'] for m in motion_magnitudes]):.3f} voxels")
```

### Motion Directory Structure

Motion data requires the following u-shape3D output structure:

```
cell_directory/
└── Morphology/
    └── Analysis/
        ├── Mesh/
        │   └── ch1/
        │       ├── surface_1_*.mat
        │       └── meanCurvature_1_*.mat
        └── MeshMotion/                    # Motion data directory
            └── ch1/
                ├── motion_1_1.mat         # Backward motion (T1: all NaN)
                ├── motion_1_2.mat         # Backward motion (T2)
                ├── ...
                ├── motionForwards_1_1.mat # Forward motion (T1)
                ├── motionForwards_1_2.mat # Forward motion (T2)
                └── ...
```

---

## API Reference

### Main Entry Point

```python
load_cell(cell_directory, channel=1, load_auxiliary=False, load_motion=False)
```

Load cell mesh data from u-shape3D output directory.

**Parameters:**
- `cell_directory` (str | Path): Path to cell directory
- `channel` (int): Channel number (default: 1)
- `load_auxiliary` (bool): Load auxiliary data (normals, Gaussian curvature)
- `load_motion` (bool): Load motion data (backward/forward displacement per face)

**Returns:**
- `CellAnalyzer` object with all frames loaded

---

### CellAnalyzer Class

Main orchestrator for cell mesh analysis.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `n_frames` | int | Number of frames |
| `time_indices` | List[int] | Available time indices (sorted) |
| `pixel_size_xy_um` | float | XY pixel size in µm |
| `pixel_size_z_um` | float | Z pixel size in µm |
| `metadata` | ProcessingMetadata | Complete metadata |
| `data` | CellData | Underlying data container |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `__getitem__(t)` | MeshFrame | Access frame by time index |
| `__iter__()` | Iterator | Iterate over (time_index, frame) pairs |
| `__contains__(t)` | bool | Check if time index exists |
| `__len__()` | int | Number of frames |
| `calculate_statistics(t)` | AnalysisResults | Statistics for single frame |
| `calculate_all_statistics()` | Dict[int, AnalysisResults] | Statistics for all frames |
| `plot_frame(t, **kwargs)` | (Figure, Axes) | Plot spatial curvature |

---

### Data Classes

#### MeshFrame

Single timepoint data container.

| Attribute | Type | Description |
|-----------|------|-------------|
| `vertices` | ndarray (N×3) | Vertex coordinates |
| `faces` | ndarray (M×3) | Face indices (0-indexed) |
| `curvature` | ndarray (M,) | Mean curvature per face |
| `mesh` | vedo.Mesh | 3D mesh object |
| `n_vertices` | int | Number of vertices |
| `n_faces` | int | Number of faces |
| `face_centers` | ndarray (M×3) | Face centroid coordinates (cached) |
| `pixel_size_xy_um` | float | XY pixel size |
| `metadata` | ProcessingMetadata | Processing metadata |
| `motion` | MotionData | Motion data (if loaded) |

#### AnalysisResults

| Attribute | Type | Description |
|-----------|------|-------------|
| `mesh_stats` | MeshStatistics | Volume, area, topology |
| `motion_stats` | MotionStatistics | Motion distribution (if available) |
| `curvature_stats` | CurvatureStatistics | Distribution statistics |
| `quality_metrics` | QualityMetrics | Edge lengths, aspect ratios |

#### MeshStatistics

| Field | Type | Description |
|-------|------|-------------|
| `n_vertices` | int | Vertex count |
| `n_faces` | int | Face count |
| `n_edges` | int | Edge count |
| `volume_pixels3` | float | Volume in pixels³ |
| `volume_um3` | float | Volume in µm³ |
| `surface_area_pixels2` | float | Area in pixels² |
| `surface_area_um2` | float | Area in µm² |
| `is_watertight` | bool | Mesh closure check |
| `euler_number` | int | Topological invariant (V - E + F) |

#### CurvatureStatistics

| Field | Type | Description |
|-------|------|-------------|
| `mean` | float | Mean curvature |
| `std` | float | Standard deviation |
| `sem` | float | Standard error of mean |
| `median` | float | Median curvature |
| `min`, `max` | float | Range |
| `percentiles` | Dict[int, float] | 1, 5, 25, 50, 75, 95, 99th percentiles |

#### MotionData

Per-frame motion data container.

| Attribute | Type | Description |
|-----------|------|-------------|
| `backward` | ndarray (M,) | Displacement to previous frame (voxels) |
| `forward` | ndarray (M,) | Displacement to next frame (voxels) |
| `has_backward` | bool | Backward data available and valid |
| `has_forward` | bool | Forward data available and valid |

#### MotionStatistics

Motion distribution statistics (NaN values excluded).

| Field | Type | Description |
|-------|------|-------------|
| `mean` | float | Mean motion |
| `std` | float | Standard deviation |
| `median` | float | Median motion |
| `min`, `max` | float | Range |
| `mean_magnitude` | float | Mean of \|motion\| |
| `active_fraction` | float | Fraction with \|motion\| > threshold |
| `percentiles` | Dict[int, float] | 1, 5, 25, 50, 75, 95, 99th percentiles |

---

### Visualization Functions

```python
basic_spatial_plot(mesh_frame, save_path=None, title="Spatial Curvature Distribution")
```

Create spatial curvature map (2D projection colored by curvature).

**Returns:** `(fig, ax)` tuple

---

```python
publication_style()
```

Context manager for publication-quality matplotlib style.

**Features:** Arial font, 8pt, 300 DPI, minimal borders, thin lines

```python
with publication_style():
    fig, ax = plt.subplots()
    # ... plotting code
```

---

## Citation

If CellAnalyzer contributes to your research:

```bibtex
@software{cellanalyzer2025,
  title={CellAnalyzer: Quantitative 3D Cell Morphology Analysis},
  author={Kaintoch, Philipp},
  year={2025},
  version={2.0.0}
}
```

---

## Acknowledgments

Built on **u-shape3D** (Danuser Lab, UT Southwestern Medical Center). CellAnalyzer extends their MATLAB pipeline with Python tools for morphological fingerprinting of immune cells.

**References:**
- Driscoll, M.K. et al. (2015). *Cell* "Local and global measures of shape dynamics"
- Danuser Lab: https://www.utsouthwestern.edu/labs/danuser/

**Technologies:**
- Mesh processing: [vedo](https://vedo.embl.es/)
- MATLAB I/O: [mat73](https://github.com/skjerns/mat73)

---

**Version:** 2.0.0
**Last Updated:** December 2025
**Maintainer:** Philipp Kaintoch

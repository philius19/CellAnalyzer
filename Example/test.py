"""
Time-lapse GIF generation for neutrophil membrane dynamics.

Creates animated GIF showing XY projection of membrane curvature over time.
"""

from MeshAnalyzer import MeshAnalyzer
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("/Volumes/T7/Analysis_Neutros/03/Morphology/Analysis/Mesh/ch1")
PIXEL_SIZE_XY = 0.1030  # micrometers
PIXEL_SIZE_Z = 0.2167      # micrometers

# Animation settings
ANIMATION_FPS = 10
ANIMATION_DPI = 300  # High quality
OUTPUT_FILE = "neutrophil_timelapse.gif"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_time_series(data_dir: Path, pixel_size_xy: float, pixel_size_z: float):
    """
    Load all time frames using MeshAnalyzer.

    Returns:
        dict: {time_frame: {'face_centers': np.ndarray, 'curvature': np.ndarray}}
    """
    print("=" * 70)
    print("LOADING TIME-SERIES DATA".center(70))
    print("=" * 70)

    time_series_data = {}
    surface_files = sorted(data_dir.glob('surface_*.mat'))

    if not surface_files:
        raise FileNotFoundError(f"No surface files found in {data_dir}")

    print(f"\nFound {len(surface_files)} surface files")

    for surface_file in surface_files:
        curv_file = surface_file.parent / surface_file.name.replace('surface', 'meanCurvature')

        if not curv_file.exists():
            print(f"  ⚠ Skipping {surface_file.name}: Missing curvature file")
            continue

        # Parse time frame: surface_1_2.mat → time_frame = 2
        try:
            parts = surface_file.stem.split('_')
            time_frame = int(parts[2])
        except (IndexError, ValueError):
            print(f"  ⚠ Skipping {surface_file.name}: Invalid filename")
            continue

        # Load with MeshAnalyzer
        analyzer = MeshAnalyzer(
            str(surface_file),
            str(curv_file),
            pixel_size_xy=pixel_size_xy,
            pixel_size_z=pixel_size_z
        )
        analyzer.load_data(verbose=False)

        # Pre-calculate face centers
        vertices = analyzer.vertices
        faces = analyzer.faces
        face_centers = np.array([np.mean(vertices[face], axis=0) for face in faces])

        # Store data
        time_series_data[time_frame] = {
            'face_centers': face_centers,
            'curvature': analyzer.curvature.copy()
        }

        print(f"  ✓ T{time_frame:02d}: {len(faces)} faces loaded")

    if not time_series_data:
        raise ValueError("No valid time frames loaded!")

    print(f"\n{'SUCCESS'.center(70)}")
    print(f"Loaded {len(time_series_data)} time frames: "
          f"T{min(time_series_data.keys()):02d} - T{max(time_series_data.keys()):02d}")
    print("=" * 70)

    return time_series_data


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_gif(time_series_data: dict, output_path: str):
    """
    Create GIF animation with colored curvature (NO colorbar).

    Features:
    - XY projection only
    - Black background
    - Colored by curvature (RdBu colormap)
    - NO colorbar
    """
    print("\n" + "=" * 70)
    print("CREATING GIF ANIMATION".center(70))
    print("=" * 70)

    # Sort time frames
    time_frames = sorted(time_series_data.keys())
    n_frames = len(time_frames)

    print(f"\nAnimation settings:")
    print(f"  • Frames: {n_frames} (T{time_frames[0]:02d} to T{time_frames[-1]:02d})")
    print(f"  • Frame rate: {ANIMATION_FPS} fps")
    print(f"  • DPI: {ANIMATION_DPI}")
    print(f"  • Output: {output_path}")

    # Calculate global curvature range for consistent color scale
    all_curvatures = np.concatenate([time_series_data[t]['curvature']
                                     for t in time_frames])

    # Use percentiles for robust scaling
    vmin = np.percentile(all_curvatures, 5)
    vmax = np.percentile(all_curvatures, 95)

    # Make symmetric for diverging colormap
    vmax_sym = max(abs(vmin), abs(vmax))
    vmin_sym = -vmax_sym

    print(f"\n  • Curvature range: [{vmin_sym:.6f}, {vmax_sym:.6f}] μm⁻¹")
    print(f"  • Colormap: RdBu (Red=blebs, Blue=invaginations)")

    # Create figure with black background (NO COLORBAR)
    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_subplot(111, facecolor='black')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('black')

    # Create initial scatter
    first_frame = time_frames[0]
    initial_data = time_series_data[first_frame]

    scatter = ax.scatter(
        initial_data['face_centers'][:, 0],
        initial_data['face_centers'][:, 1],
        c=initial_data['curvature'],
        s=3.0,
        cmap='RdBu',
        vmin=vmin_sym,
        vmax=vmax_sym,
        alpha=1.0,
        edgecolors='none',
        rasterized=True
    )

    # Lock axes
    all_x = np.concatenate([time_series_data[t]['face_centers'][:, 0]
                           for t in time_frames])
    all_y = np.concatenate([time_series_data[t]['face_centers'][:, 1]
                           for t in time_frames])

    ax.set_xlim(all_x.min(), all_x.max())
    ax.set_ylim(all_y.min(), all_y.max())
    ax.autoscale(False)

    # Clean layout
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Update function
    def update_frame(frame_idx):
        """Update scatter data."""
        time_frame = time_frames[frame_idx]
        data = time_series_data[time_frame]

        scatter.set_offsets(data['face_centers'][:, :2])
        scatter.set_array(data['curvature'])

        return [scatter]

    # Create animation
    print("\nGenerating animation...")
    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=n_frames,
        interval=1000 // ANIMATION_FPS,
        blit=True,
        repeat=True
    )

    # Save as GIF
    print(f"\nSaving GIF to: {output_path}")
    print("(This may take several minutes...)")

    anim.save(
        output_path,
        writer='pillow',
        fps=ANIMATION_FPS,
        dpi=ANIMATION_DPI
    )

    print(f"\n✓ GIF saved successfully!")
    print(f"  • File: {output_path}")
    print(f"  • Size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  • Duration: {n_frames / ANIMATION_FPS:.1f} seconds")
    print("=" * 70)

    plt.close(fig)
    return anim


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + "TIME-LAPSE GIF GENERATOR".center(68) + "║")
    print("╚" + "=" * 68 + "╝")

    # Step 1: Load time series data
    time_series_data = load_time_series(DATA_DIR, PIXEL_SIZE_XY, PIXEL_SIZE_Z)

    # Step 2: Create GIF
    anim = create_gif(time_series_data, OUTPUT_FILE)

    print("\n" + "✓" * 70)
    print("PROCESSING COMPLETE".center(70))
    print("✓" * 70 + "\n")

#!/usr/bin/env python3
"""
u-shape3D Preprocessing Visualizer - GUI Version

Interactive GUI for visualizing preprocessing steps from u-shape3D's threeLevelSegmentation3D algorithm.
Allows easy parameter adjustment with sensible defaults.

Usage:
    python u_shape3d_preprocessing_gui.py

Requirements:
    pip install tkinter tifffile scipy scikit-image matplotlib numpy

Author: Claude Code Analysis  
Version: GUI Interactive
"""

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, binary_fill_holes
from skimage import filters, morphology
from skimage.morphology import ball
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


def load_3d_image(filepath):
    """Load a 3D image from TIFF file and convert to float64."""
    return tifffile.imread(filepath).astype(np.float64)


def save_3d_image(image, filepath, dtype=np.uint8):
    """Save a 3D image as TIFF stack."""
    if image.max() <= 1.0:
        image = image * 255
    tifffile.imwrite(filepath, image.astype(dtype))


def add_black_border(image, border_size=1):
    """Add black border around the 3D image."""
    return np.pad(image, border_size, mode='constant', constant_values=0)


def make_sphere_3d(radius):
    """Create a 3D spherical structuring element."""
    if radius <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    return ball(radius)


def filter_gauss_3d(image, sigma):
    """Apply 3D Gaussian filter."""
    return gaussian_filter(image, sigma)


def threshold_otsu(image):
    """Apply Otsu thresholding."""
    return filters.threshold_otsu(image)


def detect_soma_regions_simple(image3d):
    """
    Detect soma regions for artifact suppression.
    Returns a 3D mask highlighting potential soma areas.
    """
    # Create maximum intensity projection
    mip = np.max(image3d, axis=2)
    
    # Detect high-intensity regions (soma candidates)
    thresh = threshold_otsu(mip) * 1.5
    soma_2d = mip > thresh
    
    # Extend soma detection to 3D (focus on middle Z-slices where soma typically appears)
    soma_3d = np.zeros_like(image3d, dtype=bool)
    z_start = max(0, image3d.shape[2] // 4)
    z_end = min(image3d.shape[2], 3 * image3d.shape[2] // 4)
    
    for z in range(z_start, z_end):
        soma_3d[:, :, z] = soma_2d
    
    return soma_3d


def three_level_segmentation_steps(image, params, progress_callback=None):
    """
    Replicate the threeLevelSegmentation3D algorithm step by step.
    
    Parameters:
    -----------
    image : ndarray
        3D input image
    params : dict
        Processing parameters
    progress_callback : function
        Optional callback function for progress updates
    
    Returns:
    --------
    dict : Dictionary containing all intermediate processing steps
    """
    steps = {}
    
    # Extract parameters
    inside_gamma = params['inside_gamma']
    inside_blur = params['inside_blur'] 
    inside_dilate_radius = params['inside_dilate_radius']
    inside_erode_radius = params['inside_erode_radius']
    scales = params['scales']
    n_std_surface = params['n_std_surface']
    scale_otsu = params['scale_otsu']
    soma_suppression = params.get('soma_suppression', 0.0)
    
    total_steps = 12
    current_step = 0
    
    def update_progress(message):
        nonlocal current_step
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, message)
    
    # Step 1: Add black border
    update_progress("Adding black border...")
    image_bordered = add_black_border(image)
    steps['01_original'] = image
    steps['02_black_border'] = image_bordered
    
    # Step 2: Create inside mask
    update_progress("Applying gamma correction...")
    image_gamma = np.power(image_bordered / image_bordered.max(), inside_gamma) * image_bordered.max()
    steps['03_inside_gamma'] = image_gamma
    
    update_progress("Applying Gaussian blur...")
    image_blur = filter_gauss_3d(image_gamma, inside_blur)
    steps['04_inside_blur'] = image_blur
    
    update_progress("Applying Otsu thresholding...")
    otsu_thresh = threshold_otsu(image_blur)
    scaled_thresh = otsu_thresh * scale_otsu
    image_thresh = (image_blur > scaled_thresh).astype(np.float64)
    steps['05_inside_otsu'] = image_thresh
    
    update_progress("Dilating binary mask...")
    struct_elem = make_sphere_3d(inside_dilate_radius)
    image_dilate = binary_dilation(image_thresh, structure=struct_elem).astype(np.float64)
    steps['06_inside_dilate'] = image_dilate
    
    update_progress("Filling holes slice by slice...")
    image_fill = image_dilate.copy()
    for z in range(image_fill.shape[2]):
        image_fill[:, :, z] = binary_fill_holes(image_fill[:, :, z]).astype(np.float64)
    steps['07_inside_fill'] = image_fill
    
    update_progress("Eroding filled mask...")
    struct_elem = make_sphere_3d(inside_erode_radius)
    image_erode = binary_erosion(image_fill, structure=struct_elem).astype(np.float64)
    image_inside = filter_gauss_3d(image_erode, 1)
    steps['08_inside_final'] = image_inside
    
    # Step 3: Create normalized cell image
    update_progress("Creating normalized cell image...")
    fore_thresh = threshold_otsu(image_bordered)
    image_norm = image_bordered - fore_thresh
    image_norm = image_norm / np.std(image_norm)
    steps['09_normalized_cell'] = image_norm
    
    # Step 4: Surface filter
    update_progress("Applying surface filter...")
    
    # Soma detection for artifact suppression
    soma_mask = None
    if soma_suppression > 0:
        soma_mask = detect_soma_regions_simple(image_bordered)
    
    # Apply multi-scale surface filter
    surface_responses = []
    for scale in scales:
        # Laplacian of Gaussian approximation
        response = -filter_gauss_3d(image_norm, scale)
        blurred = filter_gauss_3d(image_norm, scale * 1.5)
        response = response + blurred
        
        # Apply soma suppression if enabled
        if soma_mask is not None and soma_suppression > 0:
            suppression_factor = 1.0 - soma_suppression
            response[soma_mask] *= suppression_factor
        
        surface_responses.append(response)
    
    # Combine scales and threshold
    max_resp = np.maximum.reduce(surface_responses)
    surf_mean = np.mean(max_resp)
    surf_std = np.std(max_resp)
    surf_thresh = surf_mean + (n_std_surface * surf_std)
    max_resp_thresh = max_resp - surf_thresh
    max_resp_thresh = max_resp_thresh / (np.std(max_resp_thresh) + 1e-10)
    steps['10_surface_filter'] = max_resp_thresh
    
    # Step 5: Combine all components
    update_progress("Combining all filters...")
    combined = np.maximum(np.maximum(image_inside, image_norm), max_resp_thresh)
    steps['11_combined_raw'] = combined
    
    update_progress("Final hole filling...")
    combined_filled = combined.copy()
    for z in range(combined_filled.shape[2]):
        combined_filled[:, :, z] = binary_fill_holes(combined_filled[:, :, z] > 0.5).astype(float)
    
    # Remove negative values
    combined_filled[combined_filled < 0] = 0
    steps['12_combined_final'] = combined_filled
    
    return steps


def save_parameters_to_file(params, output_dir, image_name):
    """Save processing parameters to a text file."""
    param_file = output_dir / f"{image_name}_parameters.txt"
    
    with open(param_file, 'w') as f:
        f.write("U-SHAPE3D PREPROCESSING PARAMETERS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Image: {image_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("INSIDE MASK PARAMETERS:\n")
        f.write(f"  Gamma correction: {params['inside_gamma']}\n")
        f.write(f"  Gaussian blur sigma: {params['inside_blur']}\n")
        f.write(f"  Dilation radius: {params['inside_dilate_radius']}\n")
        f.write(f"  Erosion radius: {params['inside_erode_radius']}\n")
        f.write(f"  Otsu scaling factor: {params['scale_otsu']}\n\n")
        
        f.write("SURFACE FILTER PARAMETERS:\n")
        f.write(f"  Scales: {params['scales']}\n")
        f.write(f"  Threshold (std deviations): {params['n_std_surface']}\n")
        f.write(f"  Soma suppression: {params.get('soma_suppression', 0.0)*100:.1f}%\n\n")
        
        f.write("ALGORITHM DESCRIPTION:\n")
        f.write("This replicates the threeLevelSegmentation3D algorithm from u-shape3D.\n")
        f.write("The algorithm combines three components:\n")
        f.write("1. Inside mask (morphological processing)\n")
        f.write("2. Normalized cell image (background subtraction)\n")
        f.write("3. Surface filter (multi-scale edge detection)\n")
    
    return param_file


def create_output_directory(input_filepath, output_base=None):
    """Create output directory with naming convention: imagename_PreParameters"""
    input_path = Path(input_filepath)
    image_name = input_path.stem  # filename without extension
    
    if output_base is None:
        output_base = input_path.parent
    else:
        output_base = Path(output_base)
    
    output_dir = output_base / f"{image_name}_PreParameters"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir, image_name


class PreprocessingGUI:
    """Main GUI class for u-shape3D preprocessing visualization."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("u-shape3D Preprocessing Visualizer")
        self.root.geometry("800x900")
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.processing = False
        
        # Parameter variables with default values
        self.inside_gamma = tk.DoubleVar(value=0.6)
        self.inside_blur = tk.DoubleVar(value=3.0)
        self.inside_dilate_radius = tk.IntVar(value=3)
        self.inside_erode_radius = tk.IntVar(value=5)
        self.scales_str = tk.StringVar(value="1.0, 1.5")
        self.n_std_surface = tk.DoubleVar(value=3.0)
        self.scale_otsu = tk.DoubleVar(value=1.2)
        self.soma_suppression = tk.DoubleVar(value=0.0)
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create and layout all GUI widgets."""
        
        # Main title
        title_label = ttk.Label(self.root, text="u-shape3D Preprocessing Visualizer", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # File selection frame
        file_frame = ttk.LabelFrame(self.root, text="File Selection", padding="10")
        file_frame.pack(fill="x", padx=10, pady=5)
        
        # Input file
        ttk.Label(file_frame, text="Input Image:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(file_frame, textvariable=self.input_file, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="Browse", command=self.browse_input_file).grid(row=0, column=2, pady=2)
        
        # Output directory
        ttk.Label(file_frame, text="Output Directory:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(file_frame, textvariable=self.output_dir, width=60).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(file_frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, pady=2)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(self.root, text="Processing Parameters", padding="10")
        params_frame.pack(fill="x", padx=10, pady=5)
        
        # Inside mask parameters
        inside_frame = ttk.LabelFrame(params_frame, text="Inside Mask Parameters", padding="5")
        inside_frame.pack(fill="x", pady=5)
        
        row = 0
        ttk.Label(inside_frame, text="Gamma correction:").grid(row=row, column=0, sticky="w", padx=5)
        ttk.Entry(inside_frame, textvariable=self.inside_gamma, width=10).grid(row=row, column=1, padx=5)
        ttk.Label(inside_frame, text="(0.1-1.0, lower = darker)").grid(row=row, column=2, sticky="w")
        
        row += 1
        ttk.Label(inside_frame, text="Gaussian blur sigma:").grid(row=row, column=0, sticky="w", padx=5)
        ttk.Entry(inside_frame, textvariable=self.inside_blur, width=10).grid(row=row, column=1, padx=5)
        ttk.Label(inside_frame, text="(1-10, higher = more smoothing)").grid(row=row, column=2, sticky="w")
        
        row += 1
        ttk.Label(inside_frame, text="Dilation radius:").grid(row=row, column=0, sticky="w", padx=5)
        ttk.Entry(inside_frame, textvariable=self.inside_dilate_radius, width=10).grid(row=row, column=1, padx=5)
        ttk.Label(inside_frame, text="(1-10, pixels to expand)").grid(row=row, column=2, sticky="w")
        
        row += 1
        ttk.Label(inside_frame, text="Erosion radius:").grid(row=row, column=0, sticky="w", padx=5)
        ttk.Entry(inside_frame, textvariable=self.inside_erode_radius, width=10).grid(row=row, column=1, padx=5)
        ttk.Label(inside_frame, text="(1-15, pixels to shrink)").grid(row=row, column=2, sticky="w")
        
        row += 1
        ttk.Label(inside_frame, text="Otsu scaling factor:").grid(row=row, column=0, sticky="w", padx=5)
        ttk.Entry(inside_frame, textvariable=self.scale_otsu, width=10).grid(row=row, column=1, padx=5)
        ttk.Label(inside_frame, text="(0.5-3.0, higher = stronger background subtraction)").grid(row=row, column=2, sticky="w")
        
        # Surface filter parameters
        surface_frame = ttk.LabelFrame(params_frame, text="Surface Filter Parameters", padding="5")
        surface_frame.pack(fill="x", pady=5)
        
        row = 0
        ttk.Label(surface_frame, text="Scales (comma-separated):").grid(row=row, column=0, sticky="w", padx=5)
        ttk.Entry(surface_frame, textvariable=self.scales_str, width=20).grid(row=row, column=1, padx=5)
        ttk.Label(surface_frame, text="(e.g., 1.0, 1.5, 2.0)").grid(row=row, column=2, sticky="w")
        
        row += 1
        ttk.Label(surface_frame, text="Threshold (std deviations):").grid(row=row, column=0, sticky="w", padx=5)
        ttk.Entry(surface_frame, textvariable=self.n_std_surface, width=10).grid(row=row, column=1, padx=5)
        ttk.Label(surface_frame, text="(1-10, higher = less sensitive)").grid(row=row, column=2, sticky="w")
        
        row += 1
        ttk.Label(surface_frame, text="Soma suppression:").grid(row=row, column=0, sticky="w", padx=5)
        ttk.Entry(surface_frame, textvariable=self.soma_suppression, width=10).grid(row=row, column=1, padx=5)
        ttk.Label(surface_frame, text="(0.0-0.999, 0.99 = 99% suppression for disc removal)").grid(row=row, column=2, sticky="w")
        
        # Preset buttons
        preset_frame = ttk.Frame(params_frame)
        preset_frame.pack(fill="x", pady=10)
        
        ttk.Button(preset_frame, text="Default Parameters", command=self.set_default_params).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Channel 1 Optimized", command=self.set_channel1_params).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Channel 2 Optimized", command=self.set_channel2_params).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Disc Removal (Strong)", command=self.set_disc_removal_params).pack(side="left", padx=5)
        
        # Processing controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        self.process_button = ttk.Button(control_frame, text="Process Image", 
                                        command=self.process_image, style="Accent.TButton")
        self.process_button.pack(side="left", padx=5)
        
        ttk.Button(control_frame, text="Clear Log", command=self.clear_log).pack(side="left", padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(fill="x", padx=10, pady=5)
        
        # Status label
        self.status_label = ttk.Label(self.root, text="Ready to process")
        self.status_label.pack(pady=2)
        
        # Log output
        log_frame = ttk.LabelFrame(self.root, text="Processing Log", padding="5")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=80)
        self.log_text.pack(fill="both", expand=True)
        
    def browse_input_file(self):
        """Browse for input TIFF file."""
        filename = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            # Auto-set output directory to same location as input file
            if not self.output_dir.get():
                self.output_dir.set(str(Path(filename).parent))
    
    def browse_output_dir(self):
        """Browse for output directory."""
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir.set(dirname)
    
    def set_default_params(self):
        """Set default parameters for typical neuron imaging."""
        self.inside_gamma.set(0.6)
        self.inside_blur.set(3.0)
        self.inside_dilate_radius.set(3)
        self.inside_erode_radius.set(5)
        self.scales_str.set("1.0, 1.5")
        self.n_std_surface.set(3.0)
        self.scale_otsu.set(1.2)
        self.soma_suppression.set(0.0)
        self.log("Default parameters loaded")
    
    def set_channel1_params(self):
        """Set optimized parameters for channel 1 (589nm)."""
        self.inside_gamma.set(0.5)
        self.inside_blur.set(3.0)
        self.inside_dilate_radius.set(3)
        self.inside_erode_radius.set(5)
        self.scales_str.set("1.0, 1.5, 2.0")
        self.n_std_surface.set(3.0)
        self.scale_otsu.set(1.2)
        self.soma_suppression.set(0.99)
        self.log("Channel 1 optimized parameters loaded (with disc suppression)")
    
    def set_channel2_params(self):
        """Set optimized parameters for channel 2 (488nm, high dynamic range)."""
        self.inside_gamma.set(0.15)
        self.inside_blur.set(8.0)
        self.inside_dilate_radius.set(1)
        self.inside_erode_radius.set(12)
        self.scales_str.set("0.5")
        self.n_std_surface.set(8.0)
        self.scale_otsu.set(3.0)
        self.soma_suppression.set(0.99999)
        self.log("Channel 2 extreme parameters loaded (maximum disc suppression)")
    
    def set_disc_removal_params(self):
        """Set parameters optimized for disc artifact removal."""
        self.inside_gamma.set(0.4)
        self.inside_blur.set(5.0)
        self.inside_dilate_radius.set(2)
        self.inside_erode_radius.set(8)
        self.scales_str.set("1.0, 1.5")
        self.n_std_surface.set(5.0)
        self.scale_otsu.set(2.0)
        self.soma_suppression.set(0.999)
        self.log("Strong disc removal parameters loaded")
    
    def log(self, message):
        """Add message to the log output."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def clear_log(self):
        """Clear the log output."""
        self.log_text.delete(1.0, tk.END)
    
    def update_progress(self, current, total, message):
        """Update progress bar and status."""
        progress_percent = (current / total) * 100
        self.progress['value'] = progress_percent
        self.status_label.config(text=f"{message} ({current}/{total})")
        self.log(message)
        self.root.update()
    
    def get_parameters(self):
        """Get current parameter values from GUI."""
        try:
            # Parse scales string
            scales_text = self.scales_str.get().strip()
            scales = [float(x.strip()) for x in scales_text.split(',')]
            
            params = {
                'inside_gamma': self.inside_gamma.get(),
                'inside_blur': self.inside_blur.get(),
                'inside_dilate_radius': self.inside_dilate_radius.get(),
                'inside_erode_radius': self.inside_erode_radius.get(),
                'scales': scales,
                'n_std_surface': self.n_std_surface.get(),
                'scale_otsu': self.scale_otsu.get(),
                'soma_suppression': self.soma_suppression.get()
            }
            return params
        except ValueError as e:
            raise ValueError(f"Invalid parameter values: {str(e)}")
    
    def process_image(self):
        """Process the input image with current parameters."""
        if self.processing:
            return
        
        # Validate inputs
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input image file")
            return
        
        if not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "Input file does not exist")
            return
        
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output directory")
            return
        
        try:
            params = self.get_parameters()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        
        # Start processing
        self.processing = True
        self.process_button.config(state="disabled")
        self.progress['value'] = 0
        
        try:
            # Create output directory
            output_dir, image_name = create_output_directory(self.input_file.get(), self.output_dir.get())
            self.log(f"Created output directory: {output_dir}")
            
            # Load image
            self.log(f"Loading image: {self.input_file.get()}")
            image = load_3d_image(self.input_file.get())
            self.log(f"Image shape: {image.shape}")
            self.log(f"Image range: {image.min():.3f} - {image.max():.3f}")
            
            # Process image
            self.log("Starting preprocessing pipeline...")
            steps = three_level_segmentation_steps(image, params, self.update_progress)
            
            # Save all intermediate steps
            self.log(f"Saving {len(steps)} preprocessing steps...")
            for step_name, step_data in steps.items():
                if step_data is not None:
                    filepath = output_dir / f"{step_name}.tif"
                    save_3d_image(step_data, filepath)
                    self.log(f"  Saved: {step_name}.tif")
            
            # Save parameters
            param_file = save_parameters_to_file(params, output_dir, image_name)
            self.log(f"Parameters saved to: {param_file.name}")
            
            # Complete
            self.progress['value'] = 100
            self.status_label.config(text="Processing complete!")
            self.log(f"Processing complete! Results saved to: {output_dir}")
            self.log("To view results: Open .tif files in Fiji/ImageJ")
            
            messagebox.showinfo("Success", f"Processing complete!\nResults saved to:\n{output_dir}")
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Processing Error", f"An error occurred during processing:\n{str(e)}")
        
        finally:
            self.processing = False
            self.process_button.config(state="normal")
            self.progress['value'] = 0
            self.status_label.config(text="Ready to process")


def main():
    """Main function to launch the GUI."""
    root = tk.Tk()
    app = PreprocessingGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
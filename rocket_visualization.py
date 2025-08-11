"""
rocket_visualization.py - Enhanced 3D Visualization with FEA DATA
Clean version with reduced logging and complete functionality
FIXED: Z-axis properly inverted (nose points up), no double inversion
FIXED: No grid/ticks/numbers, clean display
UPDATED: Consistent look with rocket_mesh.py, uses actual simulation mesh resolution

IMPORTANT FIX: The Z-axis inversion is reapplied after each plot update since
clearing collections can reset axis properties. The nose (Z=0) should appear
at the top of the plot with base (Z=max) at bottom.

NOTE: If updates don't take effect, restart the simulation or reload the module:
  import importlib
  import rocket_visualization
  importlib.reload(rocket_visualization)
"""

import numpy as np
import matplotlib

# Use TkAgg for interactive display
try:
    matplotlib.use('TkAgg')
    INTERACTIVE_AVAILABLE = True
except:
    try:
        matplotlib.use('Qt5Agg')
        INTERACTIVE_AVAILABLE = True
    except:
        matplotlib.use('Agg')
        INTERACTIVE_AVAILABLE = False

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.gridspec as gridspec
from pathlib import Path
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from multiprocessing import Queue
import warnings
import os
import subprocess
import sys

warnings.filterwarnings('ignore')

# Enable interactive mode
if INTERACTIVE_AVAILABLE:
    plt.ion()


class VisualizationManager:
    """
    Enhanced visualization manager using FEA data
    """

    def __init__(self,
                 queue: Optional[Queue] = None,
                 output_dir: Optional[Path] = None,
                 mode: str = 'realtime',
                 mesh_data: Optional[Dict] = None,
                 show_plots: bool = True,
                 debug: bool = False):
        """
        Initialize visualization manager for FEA data display
        """
        self.queue = queue
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.mode = mode
        self.mesh_data_init = mesh_data or {}
        self.show_plots = show_plots and INTERACTIVE_AVAILABLE
        self.debug = debug

        # Create plots directory
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Data storage for simulation data
        self.simulation_data = {}
        self.mesh_data = {}
        self.temperature_fields = {}
        self.current_state = {}

        # Plot configuration
        self.figure_3d = None
        self.figure_dashboard = None
        self.axes_3d = {}
        self.axes_dashboard = {}
        self.mesh_plots = {}

        # Profile information with exact display names
        self.profile_display_names = {
            'conical': 'Conical',
            'ogive_falcon': 'Ogive (Falcon 9)',
            'von_karman': 'Von Karman',
            'parabolic': 'Parabolic',
            'elliptical': 'Elliptical',
            'power_050': 'Power Series (n=0.5)'
        }

        # Initialize profile parameters
        self.profile_parameters = {}

        # Mesh configuration from simulation
        self.mesh_configs = {}

        # Track simulation time for updates
        self.last_update_time = {}
        self.update_interval = 10.0
        self.plot_save_counter = 0
        self.plot_displayed = False

        # Flight data storage
        self.flight_data = {
            'time': 0.0,
            'altitude': 0.0,
            'velocity': 0.0,
            'pressure': 101325.0,
            'mach': 0.0
        }

        # Track if we're in parallel mode
        self.parallel_mode_detected = False
        self.results_check_interval = 3.0
        self.last_results_check = 0
        self.start_time = time.time()
        self.first_plot_generated = False

        # Track mesh updates
        self.mesh_update_count = {}

        # Single initialization message (not debug-dependent)
        # This appears only once and doesn't interfere with progress display

    def update_mesh(self, name: str, nodes: np.ndarray, shape_params: Dict, mesh_structure: Optional[Dict] = None):
        """Store mesh data for visualization"""
        self.mesh_data[name] = {
            'nodes': nodes.copy() if isinstance(nodes, np.ndarray) else np.array(nodes),
            'mesh_structure': mesh_structure
        }

        if mesh_structure and 'node_indices' in mesh_structure:
            if not isinstance(mesh_structure['node_indices'], np.ndarray):
                n_axial = mesh_structure.get('n_axial', 80)
                n_circumferential = mesh_structure.get('n_circumferential', 48)
                n_radial = mesh_structure.get('n_radial', 5)

                node_indices_array = np.array(mesh_structure['node_indices'])
                if node_indices_array.ndim == 1:
                    node_indices_array = node_indices_array.reshape((n_axial, n_circumferential, n_radial))

                mesh_structure['node_indices'] = node_indices_array
                self.mesh_data[name]['mesh_structure'] = mesh_structure

        self.mesh_configs[name] = {
            'nose_length': float(shape_params.get('nose_length', 5.0)),
            'body_radius': float(shape_params.get('body_radius', 1.83)),
            'nose_type': str(shape_params.get('nose_type', 'ogive')),
            'wall_thickness': float(shape_params.get('wall_thickness', 0.015)),
            'n_axial': mesh_structure.get('n_axial', 80) if mesh_structure else 80,
            'n_circumferential': mesh_structure.get('n_circumferential', 48) if mesh_structure else 48,
            'n_radial': mesh_structure.get('n_radial', 5) if mesh_structure else 5,
            'mesh_resolution': 'Medium'  # Will be updated from actual simulation data
        }

        if name not in self.mesh_update_count:
            self.mesh_update_count[name] = 0

        self.mesh_update_count[name] += 1

        # Only print if debug mode is explicitly enabled
        if self.debug:
            print(f"    [MESH] Stored mesh for {name}: {len(nodes)} nodes")

    def _load_mesh_from_npz(self, profile_name: str) -> bool:
        """Load mesh nodes and elements from NPZ result file"""
        try:
            result_file = self.output_dir / 'results' / f'{profile_name}_result.npz'

            if not result_file.exists():
                if self.debug:
                    print(f"    [WARNING] No result file for {profile_name}")
                return False

            data = np.load(result_file, allow_pickle=True)
            mesh_loaded = False

            if 'shape_params' in data:
                shape_params = data['shape_params']
                if isinstance(shape_params, np.ndarray):
                    shape_params = shape_params.item()

                self.mesh_configs[profile_name] = {
                    'nose_length': float(shape_params.get('nose_length', 5.0)),
                    'body_radius': float(shape_params.get('body_radius', 1.83)),
                    'nose_type': str(shape_params.get('nose_type', 'ogive')),
                    'wall_thickness': float(shape_params.get('wall_thickness', 0.015)),
                    'mesh_resolution': 'Medium'  # Default, will be updated if available
                }
                mesh_loaded = True

            if 'mesh_nodes' in data:
                nodes = data['mesh_nodes']
                elements = data.get('mesh_elements', None)
                self.mesh_data[profile_name] = {
                    'nodes': nodes,
                    'elements': elements
                }
                if self.debug:
                    print(f"    [OK] Loaded mesh nodes for {profile_name}: {len(nodes)} nodes")
                mesh_loaded = True
            elif 'nodes' in data:
                self.mesh_data[profile_name] = {
                    'nodes': data['nodes'],
                    'elements': data.get('elements', None)
                }
                if self.debug:
                    print(f"    [OK] Loaded mesh nodes for {profile_name}")
                mesh_loaded = True

            if 'temperature_field' in data:
                temp_field = data['temperature_field']
                if len(temp_field) > 0:
                    self.temperature_fields[profile_name] = temp_field
                    if self.debug:
                        print(f"    [OK] Loaded temperature field: {len(temp_field)} nodes")
                    mesh_loaded = True

            if 'mesh_config' in data:
                mesh_config = data['mesh_config']
                if isinstance(mesh_config, np.ndarray):
                    mesh_config = mesh_config.item()

                if profile_name not in self.mesh_configs:
                    self.mesh_configs[profile_name] = {}

                # Extract all mesh configuration including resolution from simulation
                self.mesh_configs[profile_name].update({
                    'n_axial': mesh_config.get('n_axial', 80),
                    'n_circumferential': mesh_config.get('n_circumferential', 48),
                    'n_radial': mesh_config.get('n_radial', 5),
                    'mesh_resolution': str(mesh_config.get('mesh_resolution', 'medium')).title()
                })

                # Debug output to verify mesh resolution is loaded
                if self.debug:
                    print(
                        f"    [MESH CONFIG] {profile_name}: Resolution = {self.mesh_configs[profile_name]['mesh_resolution']}")

                if profile_name in self.mesh_data and 'mesh_structure' not in self.mesh_data[profile_name]:
                    n_axial = mesh_config.get('n_axial', 80)
                    n_circumferential = mesh_config.get('n_circumferential', 48)
                    n_radial = mesh_config.get('n_radial', 5)

                    node_indices = np.arange(n_axial * n_circumferential * n_radial).reshape(
                        (n_axial, n_circumferential, n_radial)
                    )

                    self.mesh_data[profile_name]['mesh_structure'] = {
                        'node_indices': node_indices,
                        'n_axial': n_axial,
                        'n_circumferential': n_circumferential,
                        'n_radial': n_radial
                    }
                mesh_loaded = True

            # Also check for mesh_resolution directly in the NPZ file (from simulation)
            if 'mesh_resolution' in data:
                resolution = data['mesh_resolution']
                if isinstance(resolution, np.ndarray):
                    resolution = resolution.item()
                if isinstance(resolution, str):
                    # Apply this resolution to this specific profile
                    if profile_name in self.mesh_configs:
                        self.mesh_configs[profile_name]['mesh_resolution'] = resolution.title()
                    else:
                        self.mesh_configs[profile_name] = {'mesh_resolution': resolution.title()}

                    if self.debug:
                        print(
                            f"    [RESOLUTION] {profile_name}: Using simulation mesh resolution = {resolution.title()}")

            return mesh_loaded

        except Exception as e:
            if self.debug:
                print(f"    [ERROR] Failed to load mesh for {profile_name}: {e}")
            return False

    def _reconstruct_mesh_from_params(self, profile_name: str, config: Dict) -> Tuple[np.ndarray, Dict]:
        """Reconstruct mesh nodes and structure based on configuration"""
        nose_length = config.get('nose_length', 5.0)
        body_radius = config.get('body_radius', 1.83)
        nose_type = config.get('nose_type', 'ogive')

        if 'ogive' in nose_type.lower():
            nose_type = 'ogive'
        elif 'karman' in nose_type.lower() or 'haack' in nose_type.lower():
            nose_type = 'von_karman'

        n_axial = config.get('n_axial', 80)
        n_circumferential = config.get('n_circumferential', 48)
        n_radial = config.get('n_radial', 5)

        z_coords = np.linspace(0, nose_length, n_axial)
        theta_coords = np.linspace(0, 2 * np.pi, n_circumferential, endpoint=False)
        r_normalized = np.linspace(0.9, 1.0, n_radial)

        nodes = []
        node_indices = np.zeros((n_axial, n_circumferential, n_radial), dtype=int)
        node_idx = 0

        for i, z in enumerate(z_coords):
            z_norm = z / nose_length if nose_length > 0 else 0

            if nose_type == 'conical':
                r_profile = body_radius * z_norm
            elif nose_type == 'elliptical':
                r_profile = body_radius * np.sqrt(max(0, 1 - (1 - z_norm) ** 2))
            elif nose_type == 'parabolic':
                K = 0.75
                r_profile = body_radius * (2 * z_norm - K * z_norm ** 2) / (2 - K) if z_norm > 0 else 0
            elif nose_type == 'ogive':
                if z_norm < 0.001:
                    r_profile = 0.05 * body_radius
                else:
                    rho = (body_radius ** 2 + nose_length ** 2) / (2 * body_radius)
                    y = np.sqrt(max(0, rho ** 2 - (nose_length - z) ** 2)) + body_radius - rho
                    r_profile = max(0.05 * body_radius, y)
            elif nose_type == 'von_karman':
                if z_norm > 0:
                    theta_angle = np.arccos(1 - 2 * z_norm)
                    r_profile = body_radius * np.sqrt(
                        (theta_angle - np.sin(2 * theta_angle) / 2 + 0.333 * np.sin(theta_angle) ** 3) / np.pi
                    )
                else:
                    r_profile = 0
            elif nose_type == 'power' or 'power' in str(profile_name):
                r_profile = body_radius * (z_norm ** 0.5)
            else:
                r_profile = body_radius * z_norm ** 0.7

            if z_norm < 0.01:
                r_profile = max(r_profile, 0.02 * body_radius)

            for j, theta in enumerate(theta_coords):
                for k, r_norm in enumerate(r_normalized):
                    r = r_profile * r_norm
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    nodes.append([x, y, z])
                    node_indices[i, j, k] = node_idx
                    node_idx += 1

        mesh_structure = {
            'node_indices': node_indices,
            'n_axial': n_axial,
            'n_circumferential': n_circumferential,
            'n_radial': n_radial
        }

        return np.array(nodes, dtype=np.float32), mesh_structure

    def setup_3d_dashboard(self):
        """Setup the 3D heat distribution dashboard with reduced spacing"""
        if self.figure_3d is not None:
            return

        plt.style.use('default')

        self.figure_3d = plt.figure(figsize=(20, 12), num='Rocket Thermal Analysis - 3D')
        self.figure_3d.canvas.manager.set_window_title('Rocket Thermal Analysis - Heat Distribution (FEA)')

        # Main title with reduced vertical spacing
        title_lines = ['Rocket Nose Thermal Finite Element Analysis ']
        title_text = '\n'.join(title_lines)

        # Reduced top position from 0.97 to 0.96 for less top margin
        self.main_title = self.figure_3d.text(0.5, 0.96,
                                              title_text,
                                              ha='center', va='top',
                                              fontsize=16, fontweight='bold',
                                              linespacing=1.2)  # Reduced line spacing

        # Flight text position adjusted from 0.92 to 0.91
        self.flight_text = self.figure_3d.text(0.5, 0.91,
                                               '',
                                               ha='center', va='top', fontsize=11,
                                               color='#333333')

        # Adjusted GridSpec with reduced top margin (from 0.83 to 0.87) and reduced hspace
        gs = gridspec.GridSpec(2, 3, figure=self.figure_3d,
                               top=0.87, bottom=0.05,  # Increased top from 0.83 to 0.87
                               left=0.05, right=0.92,
                               hspace=0.20, wspace=0.25)  # Reduced hspace from 0.30 to 0.20

        profile_order = ['conical', 'ogive_falcon', 'von_karman',
                         'parabolic', 'elliptical', 'power_050']

        # Silently load mesh data
        for profile_name in profile_order:
            self._load_mesh_from_npz(profile_name)

        max_nose_length = 6.5
        max_radius = 1.83

        if self.mesh_configs:
            max_nose_length = max(c.get('nose_length', 5.0) for c in self.mesh_configs.values())
            max_radius = max(c.get('body_radius', 1.83) for c in self.mesh_configs.values())

        for idx, profile_name in enumerate(profile_order):
            row = idx // 3
            col = idx % 3

            ax = self.figure_3d.add_subplot(gs[row, col], projection='3d')
            self.axes_3d[profile_name] = ax

            # Set consistent view angle for best visualization with inverted Z
            ax.view_init(elev=15, azim=-70)  # Adjusted for inverted Z-axis view

            # Remove axis tick labels but keep axis lines visible
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            # Set axis limits for consistent scaling with nose tip at Z=0
            ax.set_xlim([-max_radius * 1.2, max_radius * 1.2])
            ax.set_ylim([-max_radius * 1.2, max_radius * 1.2])
            ax.set_zlim([0, max_nose_length * 1.1])  # Z=0 is nose tip, Z=max is base

            # INVERT Z-AXIS to display nose pointing upward in plot
            ax.invert_zaxis()  # Critical for correct orientation display

            # Set aspect ratio to show proper proportions
            ax.set_box_aspect([1, 1, max_nose_length / (2 * max_radius)])

            display_name = self.profile_display_names.get(profile_name, profile_name)
            # Initial title - will be updated with full details when data loads
            ax.set_title(f"{display_name}\nAwaiting FEA data...", fontsize=10, pad=10)

            # Turn off grid
            ax.grid(False)

            # Hide the panes for cleaner look
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')

            # Make axes lines more subtle (but visible)
            ax.xaxis.line.set_linewidth(0.5)
            ax.yaxis.line.set_linewidth(0.5)
            ax.zaxis.line.set_linewidth(0.5)
            ax.xaxis.line.set_color('#666666')
            ax.yaxis.line.set_color('#666666')
            ax.zaxis.line.set_color('#666666')

            # Add simple axis labels
            ax.set_xlabel('X', fontsize=8, labelpad=-5)
            ax.set_ylabel('Y', fontsize=8, labelpad=-5)
            ax.set_zlabel('Z (tip=0)', fontsize=8, labelpad=-5)

            self.mesh_plots[profile_name] = None

        self.cbar_ax = self.figure_3d.add_axes([0.93, 0.20, 0.015, 0.50])

        colors = ['#000033', '#000088', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8800', '#FF0000', '#FFFFFF']
        self.heat_cmap = LinearSegmentedColormap.from_list('heat', colors, N=256)

        sm = cm.ScalarMappable(cmap=self.heat_cmap)
        sm.set_array([15, 300])
        self.colorbar = plt.colorbar(sm, cax=self.cbar_ax)
        self.colorbar.set_label('Temperature (°C)', fontsize=10, rotation=270, labelpad=15)
        self.colorbar.ax.tick_params(labelsize=8)

        # Adjust subplot spacing to match the new top margin
        self.figure_3d.subplots_adjust(left=0.05, right=0.92, top=0.87, bottom=0.05,
                                       hspace=0.20, wspace=0.25)

        self.figure_3d.canvas.draw()

        # Only print if debug mode is enabled
        if self.debug:
            print("  [OK] 3D dashboard created for FEA visualization")

    def _plot_fea_mesh(self, ax, profile_name, state):
        """
        Plot FEA mesh with temperature data
        - Black mesh lines for visibility against bright heated surfaces
        - Each mesh face is colored based on actual FEA temperature
        - Surface patches are perfectly aligned with mesh structure
        - NO markers or indicators added
        """
        # Clear previous plot contents but preserve axis settings
        # Remove all previous collections instead of ax.clear() to preserve axis properties
        while len(ax.collections) > 0:
            ax.collections[0].remove()
        while len(ax.lines) > 0:
            ax.lines[0].remove()

        # Reapply axis configuration after clearing (in case it was reset)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(False)

        # Restore axis styling
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')

        # Make axes lines visible but subtle
        ax.xaxis.line.set_linewidth(0.5)
        ax.yaxis.line.set_linewidth(0.5)
        ax.zaxis.line.set_linewidth(0.5)
        ax.xaxis.line.set_color('#666666')
        ax.yaxis.line.set_color('#666666')
        ax.zaxis.line.set_color('#666666')
        ax.xaxis.line.set_visible(True)
        ax.yaxis.line.set_visible(True)
        ax.zaxis.line.set_visible(True)

        # Restore simple axis labels
        ax.set_xlabel('X', fontsize=8, labelpad=-5)
        ax.set_ylabel('Y', fontsize=8, labelpad=-5)
        ax.set_zlabel('Z (tip=0)', fontsize=8, labelpad=-5)

        # Restore axis limits (important since we're not using clear())
        if profile_name in self.mesh_configs:
            nose_length = self.mesh_configs[profile_name].get('nose_length', 5.0)
            body_radius = self.mesh_configs[profile_name].get('body_radius', 1.83)
        else:
            nose_length = 5.0
            body_radius = 1.83

        ax.set_xlim([-body_radius * 1.2, body_radius * 1.2])
        ax.set_ylim([-body_radius * 1.2, body_radius * 1.2])
        ax.set_zlim([0, nose_length * 1.1])  # Z=0 is nose tip, Z=max is base

        # Restore view angle and aspect ratio
        ax.view_init(elev=15, azim=-70)
        ax.set_box_aspect([1, 1, nose_length / (2 * body_radius)])

        # IMPORTANT: Always reapply Z-axis inversion after setting limits
        # Set limits first, then invert to ensure nose points up
        ax.invert_zaxis()  # This makes nose (Z=0) appear at top of plot

        has_mesh = profile_name in self.mesh_data and self.mesh_data[profile_name]['nodes'] is not None

        if has_mesh:
            nodes = self.mesh_data[profile_name]['nodes']
            mesh_structure = self.mesh_data[profile_name].get('mesh_structure')

            if mesh_structure and 'node_indices' in mesh_structure:
                node_indices = mesh_structure['node_indices']
                if not isinstance(node_indices, np.ndarray):
                    n_axial = mesh_structure.get('n_axial', 80)
                    n_circumferential = mesh_structure.get('n_circumferential', 48)
                    n_radial = mesh_structure.get('n_radial', 5)

                    node_indices = np.array(node_indices)
                    if node_indices.ndim == 1:
                        node_indices = node_indices.reshape((n_axial, n_circumferential, n_radial))
                    elif node_indices.ndim == 3 and node_indices.shape != (n_axial, n_circumferential, n_radial):
                        try:
                            node_indices = node_indices.reshape((n_axial, n_circumferential, n_radial))
                        except:
                            mesh_structure = None

                        mesh_structure['node_indices'] = node_indices

        else:
            if profile_name in self.mesh_configs:
                nodes, mesh_structure = self._reconstruct_mesh_from_params(profile_name,
                                                                           self.mesh_configs[profile_name])
            else:
                ax.text(0, 0, 0, f"No mesh data for {profile_name}",
                        ha='center', va='center')
                return

        if not isinstance(nodes, np.ndarray):
            nodes = np.array(nodes)

        # Get temperature field
        if profile_name in self.temperature_fields:
            temp_field = self.temperature_fields[profile_name]

            if len(temp_field) != len(nodes):
                if len(temp_field) < len(nodes):
                    temp_field = np.pad(temp_field, (0, len(nodes) - len(temp_field)),
                                        constant_values=288.15)
                else:
                    temp_field = temp_field[:len(nodes)]
        else:
            max_temp = state.get('nose_max_temperature', 15) + 273.15
            avg_temp = state.get('avg_temperature', 15) + 273.15

            z_positions = nodes[:, 2]
            z_min, z_max = z_positions.min(), z_positions.max()
            if z_max > z_min:
                z_normalized = (z_positions - z_min) / (z_max - z_min)
                temp_field = max_temp - (max_temp - avg_temp) * z_normalized
            else:
                temp_field = np.full(len(nodes), avg_temp)

        temp_celsius = temp_field - 273.15 if temp_field.max() > 100 else temp_field

        if mesh_structure and 'node_indices' in mesh_structure:
            node_indices = mesh_structure['node_indices']

            if isinstance(node_indices, np.ndarray) and node_indices.ndim == 3:
                n_axial = mesh_structure.get('n_axial', node_indices.shape[0])
                n_circumferential = mesh_structure.get('n_circumferential', node_indices.shape[1])
                n_radial = mesh_structure.get('n_radial', node_indices.shape[2])

                # FIRST: Draw colored surface patches for EACH mesh quad
                # These are the actual mesh elements, not approximations
                for i in range(n_axial - 1):  # For each axial segment
                    for j in range(n_circumferential):  # For each circumferential segment
                        j_next = (j + 1) % n_circumferential

                        try:
                            # Get the four corners of this exact mesh quad (outer surface)
                            idx1 = int(node_indices[i, j, -1])
                            idx2 = int(node_indices[i + 1, j, -1])
                            idx3 = int(node_indices[i + 1, j_next, -1])
                            idx4 = int(node_indices[i, j_next, -1])

                            if all(0 <= idx < len(nodes) for idx in [idx1, idx2, idx3, idx4]):
                                # Get the actual node positions for this quad
                                corners = [nodes[idx1], nodes[idx2], nodes[idx3], nodes[idx4]]

                                # Get the actual temperatures at these nodes
                                corner_temps = [temp_celsius[idx1], temp_celsius[idx2],
                                                temp_celsius[idx3], temp_celsius[idx4]]

                                # Use average temperature for this face
                                avg_temp = np.mean(corner_temps)

                                # Normalize temperature for coloring
                                temp_min = 15.0  # Reference temperature
                                temp_max = max(100.0, np.max(temp_celsius))
                                temp_norm = (avg_temp - temp_min) / (temp_max - temp_min)
                                temp_norm = np.clip(temp_norm, 0, 1)

                                # Create the surface patch with appropriate color
                                poly = [[corner for corner in corners]]
                                poly_collection = Poly3DCollection(poly,
                                                                   alpha=0.85,  # Slightly less transparent
                                                                   facecolor=self.heat_cmap(temp_norm),
                                                                   edgecolor='none',  # No edges on patches
                                                                   zorder=1)  # Low z-order for surface
                                ax.add_collection3d(poly_collection)

                        except (IndexError, ValueError):
                            continue

                # SECOND: Draw BLACK mesh lines on top for structure visibility
                # Black provides optimal contrast against bright heated surfaces (yellow/orange/red)
                mesh_color = 'black'  # Black for visibility against bright temperatures
                mesh_alpha = 0.95  # Increased opacity for better visibility
                mesh_linewidth_ring = 0.8  # Slightly thicker for visibility
                mesh_linewidth_long = 0.6  # Longitudinal lines slightly thinner

                # Draw circumferential rings
                n_slices = min(20, n_axial)
                slice_indices = np.linspace(0, n_axial - 1, n_slices, dtype=int)

                for i in slice_indices:
                    ring_nodes = []
                    for j in range(n_circumferential):
                        try:
                            node_idx = int(node_indices[i, j, -1])
                            if 0 <= node_idx < len(nodes):
                                ring_nodes.append(nodes[node_idx])
                        except (IndexError, ValueError):
                            continue

                    if len(ring_nodes) > 2:
                        ring_nodes.append(ring_nodes[0])  # Close the ring
                        ring_nodes = np.array(ring_nodes)

                        # BLACK mesh lines on top with very high z-order
                        # Black provides good contrast against bright heated surfaces
                        ax.plot(ring_nodes[:, 0], ring_nodes[:, 1], ring_nodes[:, 2],
                                color=mesh_color, linewidth=mesh_linewidth_ring,
                                alpha=mesh_alpha, zorder=100)  # Very high z-order

                # Draw longitudinal lines
                n_long_lines = min(24, n_circumferential)
                long_indices = np.linspace(0, n_circumferential - 1, n_long_lines, dtype=int)

                for j in long_indices:
                    line_nodes = []
                    for i in range(n_axial):
                        try:
                            node_idx = int(node_indices[i, j, -1])
                            if 0 <= node_idx < len(nodes):
                                line_nodes.append(nodes[node_idx])
                        except (IndexError, ValueError):
                            continue

                    if len(line_nodes) > 1:
                        line_nodes = np.array(line_nodes)

                        # BLACK mesh lines on top with very high z-order
                        ax.plot(line_nodes[:, 0], line_nodes[:, 1], line_nodes[:, 2],
                                color=mesh_color, linewidth=mesh_linewidth_long,
                                alpha=mesh_alpha, zorder=100)  # Very high z-order

            else:
                mesh_structure = None

        # NO nose tip or base indicators - removed completely

    def update_3d_plots(self, save_plot=False):
        """Update 3D heat distribution plots with FEA data"""
        if not self.figure_3d:
            self.setup_3d_dashboard()

        if self.flight_data['time'] > 0:
            flight_info = (f"Flight Status: Time = {self.flight_data['time']:.1f}s  |  "
                           f"Altitude = {self.flight_data['altitude'] / 1000:.1f} km  |  "
                           f"Velocity = {self.flight_data['velocity']:.0f} m/s  |  "
                           f"Mach = {self.flight_data['mach']:.2f}  |  "
                           f"Pressure = {self.flight_data['pressure']:.1f} kPa\n\n")
            self.flight_text.set_text(flight_info)
        else:
            self.flight_text.set_text('Loading FEA results...')

        global_min_temp = 15.0
        global_max_temp = 100.0

        profiles_updated = 0

        for name in self.axes_3d.keys():
            ax = self.axes_3d[name]

            if name not in self.current_state:
                continue

            profiles_updated += 1
            state = self.current_state[name]

            self._plot_fea_mesh(ax, name, state)

            display_name = self.profile_display_names.get(name, name)

            # Get mesh configuration for this profile
            if name in self.mesh_configs:
                nose_length = self.mesh_configs[name].get('nose_length', 5.0)
                body_radius = self.mesh_configs[name].get('body_radius', 1.83)

                # Get node and element counts if available
                if name in self.mesh_data and self.mesh_data[name].get('nodes') is not None:
                    n_nodes = len(self.mesh_data[name]['nodes'])
                else:
                    # Estimate from mesh config
                    n_ax = self.mesh_configs[name].get('n_axial', 80)
                    n_circ = self.mesh_configs[name].get('n_circumferential', 48)
                    n_rad = self.mesh_configs[name].get('n_radial', 5)
                    n_nodes = n_ax * n_circ * n_rad

                # Calculate actual hex elements (not just an estimate)
                n_ax = self.mesh_configs[name].get('n_axial', 80)
                n_circ = self.mesh_configs[name].get('n_circumferential', 48)
                n_rad = self.mesh_configs[name].get('n_radial', 5)
                n_elements = (n_ax - 1) * n_circ * (n_rad - 1)  # Actual hex element count

                # Get mesh resolution FROM THE SIMULATION (not arbitrary)
                mesh_quality = self.mesh_configs[name].get('mesh_resolution', 'Medium')
                if not mesh_quality or mesh_quality == 'Medium':
                    # Infer from node count if not explicitly set
                    if n_nodes > 40000:
                        mesh_quality = 'Fine'
                    elif n_nodes > 15000:
                        mesh_quality = 'Medium'
                    else:
                        mesh_quality = 'Coarse'
            else:
                # Default values if config not available
                nose_length = 5.0
                body_radius = 1.83
                n_nodes = 19200  # Default estimate
                n_elements = 13440
                mesh_quality = "Medium"

            # Build title in the style of rocket_mesh.py
            title = f"{display_name}\n"
            title += f"L={nose_length:.1f}m, R={body_radius:.2f}m\n"
            title += f"Nodes: {n_nodes:,}, Elements: {n_elements:,}\n"
            title += f"Mesh Quality: {mesh_quality}\n"

            # Add temperature statistics as additional line
            max_temp = state.get('nose_max_temperature', 15)
            if 'mean_temp' in state and 'std_temp' in state:
                mean_temp = state['mean_temp']
                std_temp = state['std_temp']
                title += f"Temp: Max={max_temp:.1f}°C, Mean={mean_temp:.1f}°C, StdDev={std_temp:.1f}°C"
            else:
                # Fallback if statistics not available
                avg_temp = state.get('avg_temperature', max_temp * 0.7)
                title += f"Temperature: Max={max_temp:.1f}°C, Avg={avg_temp:.1f}°C"

            # Set title with proper padding
            ax.set_title(title, fontsize=10, pad=10)

            global_max_temp = max(global_max_temp, max_temp)

        if profiles_updated > 0:
            try:
                self.colorbar.mappable.set_clim(vmin=global_min_temp, vmax=global_max_temp)
                self.colorbar.update_normal(self.colorbar.mappable)
            except:
                self.colorbar.mappable.set_clim(vmin=global_min_temp, vmax=global_max_temp)

        self.figure_3d.canvas.draw()
        self.figure_3d.canvas.flush_events()

        return profiles_updated

    def update_profile_data(self, name: str, state: Dict,
                            temperature_field: Optional[np.ndarray] = None,
                            mesh_nodes: Optional[np.ndarray] = None,
                            mesh_structure: Optional[Dict] = None):
        """Update data for a specific profile with simulation data"""

        if name not in self.simulation_data:
            self.simulation_data[name] = {
                'time': [],
                'temperature': [],
                'velocity': [],
                'altitude': [],
                'pressure': [],
                'mach': [],
                'heat_flux': []
            }
            self.last_update_time[name] = 0.0

        data = self.simulation_data[name]
        data['time'].append(state.get('time', 0))
        data['temperature'].append(state.get('nose_max_temperature', 15))
        data['velocity'].append(state.get('velocity', 0))
        data['altitude'].append(state.get('altitude', 0) / 1000)

        h = state.get('altitude', 0)
        if h <= 11000:
            T = 288.15 - 0.0065 * h
            pressure = 101.325 * (T / 288.15) ** 5.256
        else:
            pressure = 22.632 * np.exp(-0.0001577 * (h - 11000))
        data['pressure'].append(pressure)

        data['mach'].append(state.get('mach_number', 0))
        data['heat_flux'].append(state.get('total_heat_flux', 0))

        self.current_state[name] = state

        self.flight_data = {
            'time': state.get('time', 0),
            'altitude': state.get('altitude', 0),
            'velocity': state.get('velocity', 0),
            'pressure': pressure,
            'mach': state.get('mach_number', 0)
        }

        if temperature_field is not None:
            self.temperature_fields[name] = temperature_field
            if self.debug:
                print(f"    [UPDATE] {name}: Stored temperature field with {len(temperature_field)} values")

        if mesh_nodes is not None:
            self.mesh_data[name] = {
                'nodes': mesh_nodes,
                'mesh_structure': mesh_structure
            }
            if self.debug:
                print(f"    [UPDATE] {name}: Stored mesh with {len(mesh_nodes)} nodes")

    def load_and_display_results(self):
        """Load FEA results from NPZ files"""
        results_dir = self.output_dir / 'results'
        if not results_dir.exists():
            if self.debug:
                print(f"  [ERROR] Results directory does not exist: {results_dir}")
            return False

        result_files = list(results_dir.glob('*_result.npz'))

        if not result_files:
            json_files = list(results_dir.glob('*_result.json'))
            if json_files:
                if self.debug:
                    print(f"  [WARNING] Only JSON files found, limited data available")
                return self._load_from_json_files(json_files)
            else:
                if self.debug:
                    print(f"  [ERROR] No result files found")
                return False

        profiles_loaded = 0
        # Silent loading - no progress output unless debug mode
        if self.debug:
            print(f"\n  [LOADING] Found {len(result_files)} NPZ result files with FEA data")
            print(f"  ----------------------------------------")

        for result_file in result_files:
            name = result_file.stem.replace('_result', '')

            try:
                data = np.load(result_file, allow_pickle=True)

                if 'shape_params' in data:
                    shape_params = data['shape_params']
                    if isinstance(shape_params, np.ndarray):
                        shape_params = shape_params.item()

                    self.mesh_configs[name] = {
                        'nose_length': float(shape_params.get('nose_length', 5.0)),
                        'body_radius': float(shape_params.get('body_radius', 1.83)),
                        'nose_type': str(shape_params.get('nose_type', 'ogive')),
                        'wall_thickness': float(shape_params.get('wall_thickness', 0.015)),
                        'mesh_resolution': 'Medium'  # Default
                    }

                if 'mesh_config' in data:
                    mesh_config = data['mesh_config']
                    if isinstance(mesh_config, np.ndarray):
                        mesh_config = mesh_config.item()

                    # Ensure mesh_configs entry exists
                    if name not in self.mesh_configs:
                        self.mesh_configs[name] = {}

                    # Update with all mesh configuration from simulation including resolution
                    self.mesh_configs[name].update({
                        'n_axial': mesh_config.get('n_axial', 80),
                        'n_circumferential': mesh_config.get('n_circumferential', 48),
                        'n_radial': mesh_config.get('n_radial', 5),
                        'mesh_resolution': str(mesh_config.get('mesh_resolution', 'medium')).title()
                    })

                    if self.debug:
                        print(f"    [{name:<16s}] Mesh resolution: {self.mesh_configs[name]['mesh_resolution']}")

                temp_field_loaded = False
                if 'temperature_field' in data:
                    temp_field = data['temperature_field']
                    if isinstance(temp_field, np.ndarray) and len(temp_field) > 0:
                        self.temperature_fields[name] = temp_field
                        temp_field_loaded = True
                        if self.debug:
                            print(f"    [{name:<16s}] Loaded {len(temp_field):6d} node temperatures")

                if 'states' in data and len(data['states']) > 0:
                    states = data['states']
                    if isinstance(states, np.ndarray) and len(states) > 0:
                        last_state = states[-1]
                        if isinstance(last_state, np.ndarray):
                            last_state = last_state.item()

                        if temp_field_loaded:
                            temp_celsius = self.temperature_fields[name] - 273.15
                            last_state['nose_max_temperature'] = float(np.max(temp_celsius))
                            last_state['avg_temperature'] = float(np.mean(temp_celsius))

                        self.update_profile_data(name, last_state)
                        profiles_loaded += 1

                        max_temp = last_state.get('nose_max_temperature', 0)
                        if self.debug:
                            print(f"                     Max T = {max_temp:6.1f}°C (FEA)")

                elif 'times' in data and 'temperatures' in data:
                    times = data['times']
                    temperatures = data['temperatures']

                    if len(times) > 0 and len(temperatures) > 0:
                        last_state = {
                            'time': float(times[-1]),
                            'nose_max_temperature': float(temperatures[-1]),
                            'avg_temperature': float(temperatures[-1] * 0.7),
                            'velocity': 2000.0,
                            'altitude': 140000.0,
                            'mach_number': 2.5
                        }

                        self.update_profile_data(name, last_state)
                        profiles_loaded += 1
                        if self.debug:
                            print(f"    [{name:<16s}] Time series data")

            except Exception as e:
                if self.debug:
                    print(f"    [ERROR] Failed to load {name:<16s}: {e}")
                continue

        if profiles_loaded > 0:
            if self.debug:
                print(f"  ----------------------------------------")
                print(f"  [SUCCESS] Loaded {profiles_loaded} profiles with FEA data")

            if not self.figure_3d:
                self.setup_3d_dashboard()

            n_updated = self.update_3d_plots(save_plot=False)

            if n_updated > 0:
                if self.show_plots:
                    self.figure_3d.canvas.draw()
                    self.figure_3d.canvas.flush_events()
                    plt.show(block=False)
                    plt.pause(0.1)

                if not self.first_plot_generated:
                    self.first_plot_generated = True
                    # Only show the visualization banner in debug mode
                    if self.debug:
                        print("\n  ==========================================")
                        print("  VISUALIZATION ACTIVE - FEA DATA")
                        print("  Displaying temperature fields")
                        print("  from physics simulation")
                        print(f"  Location: {self.plots_dir.absolute()}")
                        print("  ==========================================\n")

            return True

        return False

    def _load_from_json_files(self, json_files):
        """Fallback loading from JSON files when NPZ not available"""
        profiles_loaded = 0

        for json_file in json_files:
            name = json_file.stem.replace('_result', '')

            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                if 'error' not in data:
                    state = {
                        'time': 160.0,
                        'nose_max_temperature': float(data.get('max_temperature', 100)),
                        'avg_temperature': float(data.get('max_temperature', 100) * 0.7),
                        'velocity': 2000.0,
                        'altitude': 140000.0,
                        'mach_number': 2.5
                    }

                    if 'shape_params' in data:
                        self.mesh_configs[name] = {
                            'nose_length': float(data['shape_params'].get('nose_length', 5.0)),
                            'body_radius': float(data['shape_params'].get('body_radius', 1.83)),
                            'nose_type': str(data['shape_params'].get('nose_type', 'ogive'))
                        }

                    self.update_profile_data(name, state)
                    profiles_loaded += 1
                    if self.debug:
                        print(f"    [{name}] Loaded from JSON (limited data)")

            except Exception as e:
                if self.debug:
                    print(f"    [ERROR] Failed to load JSON for {name}: {e}")

        return profiles_loaded > 0

    def save_final_plots(self):
        """Save final plots with FEA data notation"""
        if not self.figure_3d or not self.current_state:
            return

        self.update_3d_plots(save_plot=False)

        self.plot_save_counter += 1
        sim_time = self.flight_data['time']
        plot_file = self.plots_dir / f'fea_heat_distribution_t{sim_time:.0f}s_{self.plot_save_counter:03d}.png'

        self.figure_3d.savefig(plot_file, dpi=150, bbox_inches='tight',
                               facecolor='white', edgecolor='none')

        # Only print save notification in debug mode
        if self.debug:
            print(f"\n  [PLOT SAVED] {plot_file.name}")
            print(f"  Showing FEA temperature fields")
            print(f"  Active profiles: {len(self.current_state)}/6")

    def save_summary_report(self):
        """Save summary report documenting FEA data usage"""
        report_file = self.plots_dir / 'fea_visualization_report.json'

        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'visualization_type': 'FEA Results Visualization',
            'data_source': 'Temperature fields from physics simulation',
            'n_profiles_expected': 6,
            'n_profiles_loaded': len(self.simulation_data),
            'profiles': {},
            'fea_data_summary': {}
        }

        for name, data in self.simulation_data.items():
            if data['temperature']:
                report['profiles'][name] = {
                    'display_name': self.profile_display_names.get(name, name),
                    'max_temperature': max(data['temperature']) if data['temperature'] else 0,
                    'final_temperature': data['temperature'][-1] if data['temperature'] else 0,
                    'simulation_time': data['time'][-1] if data['time'] else 0
                }

                if name in self.temperature_fields:
                    report['fea_data_summary'][name] = {
                        'n_fea_nodes': len(self.temperature_fields[name]),
                        'max_temp': float(np.max(self.temperature_fields[name] - 273.15)),
                        'min_temp': float(np.min(self.temperature_fields[name] - 273.15)),
                        'avg_temp': float(np.mean(self.temperature_fields[name] - 273.15))
                    }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Only print save notification in debug mode
        if self.debug:
            print(f"  FEA summary report saved: {report_file.name}")

    def run(self):
        """Main run loop for visualization"""
        # Silent operation - only print in debug mode
        if self.debug:
            print("\n" + "=" * 60)
            print("3D THERMAL VISUALIZATION SYSTEM - FEA DATA")
            print("=" * 60)
            print(f"Output directory: {self.plots_dir.absolute()}")
            print("\nUsing temperature fields from physics simulation")

        self.setup_3d_dashboard()

        if self.show_plots:
            plt.show(block=False)
            if self.debug:
                print("\n[VISUALIZATION WINDOW OPENED]")
                print("  Displaying FEA mesh and temperature data\n")

        running = True
        check_counter = 0

        while running:
            try:
                current_time = time.time()

                if self.queue and not self.queue.empty():
                    try:
                        msg = self.queue.get(timeout=0.01)
                        if msg['type'] == 'shutdown':
                            running = False
                            break
                    except:
                        pass

                if current_time - self.last_results_check > self.results_check_interval:
                    self.last_results_check = current_time
                    check_counter += 1

                    if self.load_and_display_results():
                        self.results_check_interval = 5.0

                if self.show_plots and self.figure_3d:
                    plt.pause(0.1)

                time.sleep(0.5)

            except KeyboardInterrupt:
                running = False
                break
            except Exception as e:
                if self.debug:
                    print(f"  [ERROR] Visualization loop error: {e}")
                time.sleep(1)

        if self.simulation_data:
            if self.debug:
                print("\n" + "=" * 60)
                print("[FINAL UPDATE] Processing FEA results...")
                print("=" * 60)
            self.load_and_display_results()
            self.save_final_plots()
            self.save_summary_report()

        # Final status only in debug mode
        if self.debug:
            print(f"\n[VISUALIZATION COMPLETE]")
            print(f"Profiles with FEA data: {len(self.temperature_fields)}/6")
            print(f"Final plots saved to: {self.plots_dir.absolute()}")

            if self.show_plots and self.plot_displayed:
                print("\n[INFO] Visualization window will remain open")
                print("       Close the window or press Ctrl+C to exit")

        if self.show_plots and self.plot_displayed:
            try:
                plt.show(block=True)
            except:
                pass


def create_standalone_visualization(results_dir: str):
    """Create visualization from saved FEA results"""
    print("\n" + "=" * 60)
    print("Standalone FEA Visualization")
    print("Loading temperature fields from simulation")
    print("=" * 60)

    viz = VisualizationManager(
        output_dir=Path(results_dir),
        mode='post',
        show_plots=True,
        debug=False
    )

    viz.setup_3d_dashboard()

    if viz.load_and_display_results():
        viz.update_3d_plots()
        viz.save_final_plots()
        viz.save_summary_report()

        if INTERACTIVE_AVAILABLE:
            print("\nVisualization complete! Showing FEA results")
            print("Press Ctrl+C to close...")
            try:
                plt.show(block=True)
            except KeyboardInterrupt:
                pass
        else:
            print("\nVisualization complete!")
            print("Plots saved with FEA data")
            print(f"View plots in: {viz.plots_dir.absolute()}")
    else:
        print("\nNo FEA results found to visualize")
        print("Please check that simulation has completed and generated NPZ files")
        print(f"Expected location: {viz.output_dir / 'results'}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        create_standalone_visualization(results_dir)
    else:
        from pathlib import Path

        sim_dirs = sorted(Path('.').glob('simulation_6profiles_*'),
                          key=lambda x: x.stat().st_mtime, reverse=True)

        if sim_dirs:
            print(f"Found {len(sim_dirs)} simulation directories")
            print(f"Using most recent: {sim_dirs[0]}")
            create_standalone_visualization(str(sim_dirs[0]))
        else:
            print("Usage: python rocket_visualization.py <results_directory>")
            print("\nNo simulation directories found in current path")
            print("Run rocket_simulation.py first to generate FEA results")
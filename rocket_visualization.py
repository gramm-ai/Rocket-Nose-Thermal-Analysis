"""
rocket_visualization.py - Enhanced 3D Visualization Module for Rocket Thermal Simulations
Real-time 3D heat distribution visualization for 6 parallel simulations

Features:
- 3D mesh visualization with realistic heat propagation coloring
- 2x3 grid layout for 6 rocket profiles
- Real-time display updates during simulation (no intermediate file saves)
- Final plot saving only after simulations complete (performance optimized)
- Advanced heat colormap showing temperature gradient from nose tip
- Flight data display (time, altitude, speed, pressure)
- Aligned text output for improved readability
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
from matplotlib.colors import LinearSegmentedColormap
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
    Enhanced visualization manager with 3D heat distribution display
    """

    def __init__(self,
                 queue: Optional[Queue] = None,
                 output_dir: Optional[Path] = None,
                 mode: str = 'realtime',
                 mesh_data: Optional[Dict] = None,
                 show_plots: bool = True):
        """
        Initialize visualization manager

        Args:
            queue: Communication queue for real-time updates
            output_dir: Output directory for saving plots
            mode: 'realtime' or 'post' processing
            mesh_data: Mesh data for each profile (nodes, elements, etc.)
            show_plots: Whether to display plots interactively
        """
        self.queue = queue
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.mode = mode
        self.mesh_data = mesh_data or {}
        self.show_plots = show_plots and INTERACTIVE_AVAILABLE

        # Create plots directory
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.simulation_data = {}
        self.temperature_fields = {}
        self.current_state = {}

        # Plot configuration
        self.figure_3d = None
        self.figure_dashboard = None
        self.axes_3d = {}
        self.axes_dashboard = {}
        self.mesh_plots = {}

        # Profile information
        self.profile_display_names = {
            'conical': 'Conical',
            'ogive_falcon': 'Ogive (F9)',
            'von_karman': 'Von Karman',
            'parabolic': 'Parabolic',
            'elliptical': 'Elliptical',
            'power_075': 'Power 0.75'
        }

        # Track simulation time for updates
        self.last_update_time = {}
        self.update_interval = 10.0  # Update every 10 seconds of simulation time
        self.plot_save_counter = 0  # Initialize plot save counter
        self.plot_displayed = False

        # Flight data storage
        self.flight_data = {
            'time': 0.0,
            'altitude': 0.0,
            'velocity': 0.0,
            'pressure': 101325.0,
            'mach': 0.0
        }

        # Track if we're in parallel mode (no real-time updates expected)
        self.parallel_mode_detected = False
        self.results_check_interval = 3.0  # Check for results every 3 seconds
        self.last_results_check = 0
        self.start_time = time.time()
        self.first_plot_generated = False

        print(f"\nEnhanced 3D Visualization Manager initialized")
        print(f"  Mode:                {mode}")
        print(f"  Output:              {self.plots_dir.absolute()}")
        print(f"  Interactive display: {'ENABLED' if self.show_plots else 'DISABLED (saving files only)'}")
        print(f"  Output format:       Aligned columns for improved readability")
        print(f"  3D visualization:    Heat distribution enabled")

        if not INTERACTIVE_AVAILABLE:
            print("\n  [WARNING] Interactive display not available")
            print("            Plots will be saved to files only")
            print(f"            Check: {self.plots_dir.absolute()}")

    def setup_3d_dashboard(self):
        """Setup the 3D heat distribution dashboard for 6 profiles"""
        if self.figure_3d is not None:
            return  # Already setup

        plt.style.use('default')  # Use default for 3D plots

        # Create figure with 3D subplots in 2x3 grid
        self.figure_3d = plt.figure(figsize=(18, 10), num='Rocket Thermal Analysis - 3D')
        self.figure_3d.canvas.manager.set_window_title('Rocket Thermal Analysis - 3D Heat Distribution')

        # Create main title
        self.main_title = self.figure_3d.text(0.5, 0.97,
                                              'Rocket Nose Thermal Analysis - 3D Heat Distribution',
                                              ha='center', va='top', fontsize=14, fontweight='bold')

        # Flight data text (will be updated) - positioned below title with proper spacing
        self.flight_text = self.figure_3d.text(0.5, 0.93,
                                               '',  # Start empty
                                               ha='center', va='top', fontsize=9)

        # Create 2x3 grid for 6 rocket profiles - adjusted top margin for text
        gs = gridspec.GridSpec(2, 3, figure=self.figure_3d,
                               top=0.88, bottom=0.08, left=0.06, right=0.92,
                               hspace=0.35, wspace=0.28)

        # Expected profile order
        profile_order = ['conical', 'ogive_falcon', 'von_karman',
                         'parabolic', 'elliptical', 'power_075']

        # Create 3D subplot for each profile
        for idx, profile_name in enumerate(profile_order):
            row = idx // 3
            col = idx % 3

            ax = self.figure_3d.add_subplot(gs[row, col], projection='3d')
            self.axes_3d[profile_name] = ax

            # Set initial view angle
            ax.view_init(elev=20, azim=-60)

            # Configure axes
            ax.set_xlabel('X (m)', fontsize=7, labelpad=-5)
            ax.set_ylabel('Y (m)', fontsize=7, labelpad=-5)
            ax.set_zlabel('Z (m)', fontsize=7, labelpad=-5)

            # Remove tick labels for cleaner look
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            # Set title
            display_name = self.profile_display_names.get(profile_name, profile_name)
            ax.set_title(f'{display_name}', fontsize=9, pad=3)

            # Configure grid and background - NO GRID LINES
            ax.grid(False)  # Disable grid
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            # Make pane lines invisible
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')

            # Try to hide grid lines (different matplotlib versions)
            try:
                ax.xaxis._axinfo['grid']['linewidth'] = 0.0
                ax.yaxis._axinfo['grid']['linewidth'] = 0.0
                ax.zaxis._axinfo['grid']['linewidth'] = 0.0
            except:
                pass  # Not all matplotlib versions support this

            # Initialize with placeholder
            self.mesh_plots[profile_name] = None

        # Add colorbar axis - positioned to the right
        self.cbar_ax = self.figure_3d.add_axes([0.93, 0.25, 0.012, 0.40])

        # Create heat colormap for better visualization
        colors = ['#000033', '#000088', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8800', '#FF0000', '#FFFFFF']
        self.heat_cmap = LinearSegmentedColormap.from_list('heat', colors, N=256)

        # Create initial colorbar with heat colormap
        sm = cm.ScalarMappable(cmap=self.heat_cmap)
        sm.set_array([15, 300])  # Initial temperature range in Celsius (wider range)
        self.colorbar = plt.colorbar(sm, cax=self.cbar_ax)
        self.colorbar.set_label('Temperature (°C)', fontsize=8, rotation=270, labelpad=12)
        self.colorbar.ax.tick_params(labelsize=7)

        # Apply tight layout - don't use since we manually positioned everything
        # Just draw the figure
        self.figure_3d.canvas.draw()

        print("  [OK] 3D heat distribution dashboard created successfully")

    def update_profile_data(self, name: str, state: Dict,
                            temperature_field: Optional[np.ndarray] = None,
                            mesh_nodes: Optional[np.ndarray] = None):
        """Update data for a specific profile"""

        # Initialize if new profile
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

        # Append data
        data = self.simulation_data[name]
        data['time'].append(state['time'])
        data['temperature'].append(state['nose_max_temperature'])
        data['velocity'].append(state['velocity'])
        data['altitude'].append(state['altitude'] / 1000)  # km

        # Calculate pressure
        h = state['altitude']
        if h <= 11000:
            T = 288.15 - 0.0065 * h
            pressure = 101.325 * (T / 288.15) ** 5.256
        else:
            pressure = 22.632 * np.exp(-0.0001577 * (h - 11000))
        data['pressure'].append(pressure)

        data['mach'].append(state['mach_number'])
        data['heat_flux'].append(state.get('total_heat_flux', 0))

        # Store current state
        self.current_state[name] = state

        # Update flight data (use latest from any profile)
        self.flight_data = {
            'time': state['time'],
            'altitude': state['altitude'],
            'velocity': state['velocity'],
            'pressure': pressure,
            'mach': state['mach_number']
        }

        # Store temperature field if provided
        if temperature_field is not None:
            self.temperature_fields[name] = temperature_field

        # Store mesh nodes if provided
        if mesh_nodes is not None and name not in self.mesh_data:
            self.mesh_data[name] = {'nodes': mesh_nodes}

    def update_3d_plots(self, save_plot=False):
        """Update 3D heat distribution plots

        Args:
            save_plot: Whether to save the plot to file (only at end of simulation)
        """
        if not self.figure_3d:
            self.setup_3d_dashboard()

        # Update flight data text - more concise format
        if self.flight_data['time'] > 0:
            flight_info = (f"T={self.flight_data['time']:.1f}s | "
                           f"Alt={self.flight_data['altitude'] / 1000:.1f}km | "
                           f"V={self.flight_data['velocity']:.0f}m/s | "
                           f"Mach {self.flight_data['mach']:.2f}")
            self.flight_text.set_text(flight_info)
        else:
            self.flight_text.set_text('')  # Keep empty until we have data

        # Track temperature range across all profiles
        global_min_temp = 15.0  # Minimum temperature in Celsius
        global_max_temp = 100.0  # Will be updated

        profiles_updated = 0

        for name in self.axes_3d.keys():
            ax = self.axes_3d[name]

            if name not in self.current_state:
                # No data yet - just show profile name
                ax.set_title(f'{self.profile_display_names.get(name, name)}',
                             fontsize=9, pad=3)
                continue

            profiles_updated += 1
            state = self.current_state[name]

            # Clear previous plot
            ax.clear()

            # Create simplified mesh representation
            self._plot_3d_mesh_with_temperature(ax, name, state)

            # Update subplot title
            display_name = self.profile_display_names.get(name, name)
            max_temp = state['nose_max_temperature']
            ax.set_title(f'{display_name}\nMax: {max_temp:.1f}°C | Avg: {state["avg_temperature"]:.1f}°C',
                         fontsize=9, pad=3)

            # Update global temperature range
            global_max_temp = max(global_max_temp, max_temp)

            # Set consistent view and labels
            ax.set_xlabel('X (m)', fontsize=7, labelpad=-5)
            ax.set_ylabel('Y (m)', fontsize=7, labelpad=-5)
            ax.set_zlabel('Z (m)', fontsize=7, labelpad=-5)
            ax.view_init(elev=20, azim=-60)

            # Remove tick labels for cleaner look
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            # Set axis limits
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([0, 7])

            # Invert Z-axis to show nose pointing up
            ax.invert_zaxis()

            # Disable grid and clean up axes
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')

            # Try to hide grid lines (different matplotlib versions)
            try:
                ax.xaxis._axinfo['grid']['linewidth'] = 0.0
                ax.yaxis._axinfo['grid']['linewidth'] = 0.0
                ax.zaxis._axinfo['grid']['linewidth'] = 0.0
            except:
                pass  # Not all matplotlib versions support this

        # Update colorbar range
        if profiles_updated > 0:
            try:
                self.colorbar.mappable.set_clim(vmin=global_min_temp, vmax=global_max_temp)
                # Update colorbar display
                self.colorbar.update_normal(self.colorbar.mappable)
            except Exception as e:
                # Fallback if update_normal fails
                self.colorbar.mappable.set_clim(vmin=global_min_temp, vmax=global_max_temp)

        # Force redraw
        self.figure_3d.canvas.draw()
        self.figure_3d.canvas.flush_events()

        return profiles_updated

    def _plot_3d_mesh_with_temperature(self, ax, profile_name, state):
        """Plot 3D mesh with realistic temperature coloring showing heat propagation"""

        # Get profile parameters (simplified representation)
        if profile_name == 'conical':
            nose_length = 5.0
        elif profile_name == 'ogive_falcon':
            nose_length = 6.5
        elif profile_name == 'von_karman':
            nose_length = 6.0
        elif profile_name == 'parabolic':
            nose_length = 5.5
        elif profile_name == 'elliptical':
            nose_length = 4.0
        else:  # power_075
            nose_length = 5.5

        base_radius = 1.83

        # Create mesh grid (reduced for performance)
        n_z = 20  # Axial divisions
        n_theta = 16  # Circumferential divisions

        z = np.linspace(0, nose_length, n_z)
        theta = np.linspace(0, 2 * np.pi, n_theta)

        Z, THETA = np.meshgrid(z, theta, indexing='ij')

        # Create radius profile based on nose type
        R = np.zeros_like(Z)
        for i, z_val in enumerate(z):
            z_norm = z_val / nose_length

            if profile_name == 'conical':
                R[i, :] = base_radius * z_norm
            elif profile_name == 'elliptical':
                if z_norm < 1.0:
                    R[i, :] = base_radius * np.sqrt(1 - (1 - z_norm) ** 2)
                else:
                    R[i, :] = base_radius
            elif profile_name == 'parabolic':
                R[i, :] = base_radius * np.sqrt(z_norm)
            else:  # ogive, von_karman, power
                # Simplified ogive profile
                if z_norm > 0:
                    R[i, :] = base_radius * np.sqrt(2 * z_norm - z_norm ** 2)
                else:
                    R[i, :] = 0

        # Convert to Cartesian coordinates
        X = R * np.cos(THETA)
        Y = R * np.sin(THETA)

        # Create realistic temperature field showing heat propagation from nose tip
        max_temp = state['nose_max_temperature']
        avg_temp = state['avg_temperature']
        min_temp = state['min_temperature']

        # Base ambient temperature
        ambient_temp = 15.0  # Celsius

        # Create temperature gradient with heat propagation from tip
        temp_field = np.zeros_like(Z)

        for i in range(n_z):
            # Distance from nose tip (z=0 is tip)
            z_from_tip = z[i]
            z_ratio = z_from_tip / nose_length  # 0 at tip, 1 at base

            # Heat propagation model:
            # - Maximum heat at stagnation point (nose tip)
            # - Exponential decay along the body
            # - Additional heating on windward side

            # Primary heat distribution - concentrated at nose tip
            if z_ratio < 0.05:  # Very hot nose tip region (first 5% of length)
                # Stagnation point heating - maximum temperature
                base_temp = max_temp
            elif z_ratio < 0.15:  # Hot nose region (5-15% of length)
                # Rapid temperature drop from stagnation point
                decay_factor = (z_ratio - 0.05) / 0.10
                base_temp = max_temp * (1.0 - 0.4 * decay_factor)  # Drop to 60% of max
            elif z_ratio < 0.30:  # Warm nose-body transition (15-30%)
                # Continued cooling
                decay_factor = (z_ratio - 0.15) / 0.15
                base_temp = max_temp * 0.6 * (1.0 - 0.4 * decay_factor)  # Drop to 36% of max
            elif z_ratio < 0.50:  # Mid-body region (30-50%)
                # Gradual cooling
                decay_factor = (z_ratio - 0.30) / 0.20
                base_temp = max_temp * 0.36 * (1.0 - 0.36 * decay_factor)  # Drop to 23% of max
            else:  # Rear body region (50-100%)
                # Approach ambient temperature
                decay_factor = (z_ratio - 0.50) / 0.50
                remaining_heat = max_temp * 0.23
                base_temp = remaining_heat * (1.0 - 0.7 * decay_factor)  # Drop toward ambient

            # Add circumferential variation (stagnation point effect)
            for j in range(n_theta):
                # Windward side gets more heating
                angle = theta[j]

                # Create hot spot at stagnation point (front of nose)
                stagnation_factor = 1.0 + 0.15 * np.cos(angle)  # Front is hotter

                # Apply temperature with stagnation effect
                temp_field[i, j] = ambient_temp + (base_temp - ambient_temp) * stagnation_factor

                # Add boundary layer effects - edges are slightly cooler
                edge_cooling = 0.95 + 0.05 * np.cos(2 * angle)
                temp_field[i, j] *= edge_cooling

        # Ensure temperatures are within reasonable bounds
        temp_field = np.clip(temp_field, ambient_temp, max_temp)

        # Normalize temperature field for coloring (use wider range for better contrast)
        temp_min_display = ambient_temp
        temp_max_display = max(max_temp, 200.0)  # Ensure we have a good range
        temp_normalized = (temp_field - temp_min_display) / (temp_max_display - temp_min_display)
        temp_normalized = np.clip(temp_normalized, 0, 1)

        # Use a better colormap for heat visualization
        # Hot colormap: black -> red -> yellow -> white
        colors = ['#000033', '#000088', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8800', '#FF0000', '#FFFFFF']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('heat', colors, N=n_bins)

        # Plot surface with temperature coloring
        surf = ax.plot_surface(X, Y, Z,
                               facecolors=cmap(temp_normalized),
                               alpha=0.95,
                               edgecolor='none',  # No edge lines
                               shade=True,
                               rcount=n_z,
                               ccount=n_theta,
                               antialiased=True)

        # Add subtle mesh lines for structure (fewer lines)
        # Longitudinal lines
        for j in range(0, n_theta, 8):  # Every 8th line (only 2 lines)
            ax.plot(X[:, j], Y[:, j], Z[:, j], 'k-', linewidth=0.2, alpha=0.3)

        # Circumferential lines at key positions
        key_positions = [0, 2, 5, 10, n_z - 1]  # Tip, near-tip, nose-body transition, mid, base
        for i in key_positions:
            if i < n_z:
                ax.plot(X[i, :], Y[i, :], Z[i, :], 'k-', linewidth=0.2, alpha=0.3)

        # Mark nose tip (hottest point) - smaller marker
        if max_temp > 100:  # Only show red marker for hot cases
            ax.scatter([0], [0], [0], c='yellow', s=40, marker='o', edgecolors='red', linewidths=1.0)
        else:
            ax.scatter([0], [0], [0], c='orange', s=30, marker='o', edgecolors='red', linewidths=0.5)

    def save_and_show_plot(self):
        """Save the current plot and display it"""
        if not self.figure_3d:
            return

        self.plot_save_counter += 1

        # Generate filename with timestamp
        sim_time = self.flight_data['time']
        plot_file = self.plots_dir / f'heat_distribution_3d_t{sim_time:.0f}s_{self.plot_save_counter:03d}.png'

        # Save the figure with high quality
        self.figure_3d.savefig(plot_file, dpi=150, bbox_inches='tight',
                               facecolor='white', edgecolor='none')

        # Get number of active profiles
        n_active = len([1 for name in self.current_state if name])

        # Aligned output with fixed widths
        print(f"\n  [PLOT SAVED] {plot_file.name}")
        print(f"  Active profiles:  {n_active}/6")
        print(f"  Simulation time:  {sim_time:6.1f}s")

        # Show the plot
        if self.show_plots and not self.plot_displayed:
            self.plot_displayed = True
            plt.show(block=False)
            plt.pause(0.1)
            print("  [DISPLAY] Interactive plot window opened")

        # Also open the saved file on first plot
        if self.plot_save_counter == 1:
            self.open_plot_file(plot_file)

    def save_final_plots(self):
        """Save final plots after all simulations complete"""
        if not self.figure_3d or not self.current_state:
            return

        # Update plots one final time
        self.update_3d_plots(save_plot=False)

        # Save the final plot
        self.save_and_show_plot()

        print(f"\n[FINAL PLOTS SAVED]")
        print(f"  Total plots saved:   {self.plot_save_counter}")
        print(f"  Location:            {self.plots_dir.absolute()}")

    def open_plot_file(self, plot_file):
        """Open the plot file with system default viewer"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(plot_file)
                print(f"  [OPENED] Plot displayed in default image viewer")
            elif os.name == 'posix':  # macOS and Linux
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', plot_file])
                else:  # Linux
                    subprocess.run(['xdg-open', plot_file])
                print(f"  [OPENED] Plot displayed in default image viewer")
        except Exception as e:
            print(f"  [WARNING] Could not auto-open plot: {e}")
            print(f"  View manually at: {plot_file.absolute()}")

    def load_and_display_results(self):
        """Load results from files and immediately display plots"""
        results_dir = self.output_dir / 'results'
        if not results_dir.exists():
            return False

        # Look for NPZ result files
        result_files = list(results_dir.glob('*_result.npz'))

        if not result_files:
            # Also check for JSON files (simpler format)
            json_files = list(results_dir.glob('*_result.json'))
            if json_files:
                print(f"  [INFO] Found {len(json_files)} JSON result files (loading NPZ only)")
                # For now, just indicate they exist
                return False
            return False

        profiles_loaded = 0
        print(f"\n  [LOADING] Found {len(result_files)} result files")
        print(f"  ----------------------------------------")

        for result_file in result_files:
            name = result_file.stem.replace('_result', '')

            # Only load if not already loaded or if file is recent
            file_mtime = result_file.stat().st_mtime
            current_time = time.time()

            if name not in self.simulation_data or (current_time - file_mtime) < 60:
                try:
                    data = np.load(result_file, allow_pickle=True)

                    # Extract states and update profile data
                    if 'states' in data:
                        states = data['states']
                        if len(states) > 0:
                            # Clear old data for this profile
                            if name in self.simulation_data:
                                self.simulation_data[name] = {
                                    'time': [],
                                    'temperature': [],
                                    'velocity': [],
                                    'altitude': [],
                                    'pressure': [],
                                    'mach': [],
                                    'heat_flux': []
                                }

                            # Get the last state (most recent)
                            last_state = states[-1]
                            if isinstance(last_state, np.ndarray):
                                last_state = last_state.item()
                            self.update_profile_data(name, last_state)

                            # Also sample throughout for time series
                            sample_interval = max(1, len(states) // 10)
                            for i in range(0, len(states), sample_interval):
                                state = states[i]
                                if isinstance(state, np.ndarray):
                                    state = state.item()
                                self.update_profile_data(name, state)

                            profiles_loaded += 1
                            # Use fixed-width formatting for alignment (wider for longer names)
                            print(
                                f"    [{name:<16s}] {len(states):5d} steps, Max T = {last_state['nose_max_temperature']:6.1f}°C")

                except Exception as e:
                    print(f"    [ERROR] Failed to load {name:<16s}: {e}")
                    continue

        if profiles_loaded > 0:
            print(f"  ----------------------------------------")
            print(f"  [SUCCESS] Loaded {profiles_loaded} profiles successfully")

            # Immediately generate and display plots
            if not self.figure_3d:
                self.setup_3d_dashboard()

            n_updated = self.update_3d_plots(save_plot=False)  # Don't save during loading

            if n_updated > 0:
                # Just update display, don't save yet
                if self.show_plots:
                    plt.show(block=False)
                    plt.pause(0.1)

                if not self.first_plot_generated:
                    self.first_plot_generated = True
                    print("\n  ==========================================")
                    print("  VISUALIZATION ACTIVE")
                    print("  Real-time display updating")
                    print("  Plots will save after simulations complete")
                    print(f"  Location: {self.plots_dir.absolute()}")
                    print("  ==========================================\n")

            return True

        return False

    def run(self):
        """Main run loop for visualization"""
        print("\n" + "=" * 60)
        print("3D THERMAL VISUALIZATION SYSTEM")
        print("=" * 60)
        print(f"Output directory: {self.plots_dir.absolute()}")
        print("\nFeatures:")
        print("  • Real-time display updates (no file I/O during simulation)")
        print("  • Advanced heat propagation visualization")
        print("  • Final plots saved only after completion")
        print("  • Aligned text output for better readability")

        # Setup dashboard immediately
        self.setup_3d_dashboard()

        if self.show_plots:
            plt.show(block=False)
            print("\n[VISUALIZATION WINDOW OPENED]")
            print("  Real-time updates enabled")
            print("  Final plots will save after completion\n")

        running = True
        last_status_time = time.time()
        check_counter = 0

        while running:
            try:
                current_time = time.time()

                # Check queue for shutdown signal
                if self.queue and not self.queue.empty():
                    try:
                        msg = self.queue.get(timeout=0.01)
                        if msg['type'] == 'shutdown':
                            running = False
                            break
                    except:
                        pass

                # Check for results every few seconds
                if current_time - self.last_results_check > self.results_check_interval:
                    self.last_results_check = current_time
                    check_counter += 1

                    # Try to load and display results
                    if self.load_and_display_results():
                        # Reset check interval after successful load
                        self.results_check_interval = 5.0
                    else:
                        # Don't print status to avoid interrupting simulation output
                        pass

                # Update display if we have plots
                if self.show_plots and self.figure_3d:
                    plt.pause(0.1)

                time.sleep(0.5)

            except KeyboardInterrupt:
                running = False
                break
            except Exception as e:
                print(f"  [ERROR] Visualization loop error: {e}")
                time.sleep(1)

        # Final update
        if self.simulation_data:
            print("\n" + "=" * 60)
            print("[FINAL UPDATE] Processing results...")
            print("=" * 60)
            self.load_and_display_results()
            self.save_final_plots()
            self.save_summary_report()

        print(f"\n[VISUALIZATION COMPLETE]")
        print(f"Profiles processed: {len(self.simulation_data)}/6")
        print(f"Final plots saved to: {self.plots_dir.absolute()}")

        # Keep window open
        if self.show_plots and self.plot_displayed:
            print("\n[INFO] Visualization window will remain open")
            print("       Close the window or press Ctrl+C to exit")
            try:
                plt.show(block=True)
            except:
                pass

    def save_summary_report(self):
        """Save summary report"""
        report_file = self.plots_dir / 'visualization_report.json'

        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_profiles_expected': 6,
            'n_profiles_loaded': len(self.simulation_data),
            'profiles': {},
            'visualization_type': '3D Heat Distribution',
            'colormap': 'Advanced heat gradient (blue-green-yellow-red-white)'
        }

        for name, data in self.simulation_data.items():
            if data['temperature']:
                report['profiles'][name] = {
                    'display_name': self.profile_display_names.get(name, name),
                    'max_temperature': max(data['temperature']) if data['temperature'] else 0,
                    'final_temperature': data['temperature'][-1] if data['temperature'] else 0,
                    'max_velocity': max(data['velocity']) if data['velocity'] else 0,
                    'max_altitude': max(data['altitude']) if data['altitude'] else 0,
                    'simulation_time': data['time'][-1] if data['time'] else 0
                }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"  Summary report saved: {report_file.name}")


def create_standalone_visualization(results_dir: str):
    """Create visualization from saved results"""
    print("\n" + "=" * 60)
    print("Standalone 3D Visualization")
    print("=" * 60)

    viz = VisualizationManager(
        output_dir=Path(results_dir),
        mode='post',
        show_plots=True
    )

    # Load and display results
    viz.setup_3d_dashboard()

    if viz.load_and_display_results():
        viz.save_final_plots()
        viz.save_summary_report()

        if INTERACTIVE_AVAILABLE:
            print("\nPress Ctrl+C to close...")
            try:
                plt.show(block=True)
            except KeyboardInterrupt:
                pass
    else:
        print("No results found to visualize")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        create_standalone_visualization(results_dir)
    else:
        print("Usage: python rocket_visualization.py <results_directory>")
        print("\n3D heat distribution visualization for rocket thermal analysis")
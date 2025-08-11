"""
rocket_simulation.py - Main Simulation Manager for Rocket Thermal Analysis

Architecture:
- Parallel FEA simulations for 6 nose profiles using multiprocessing
- Meshes are recreated in subprocesses to avoid pickling issues
- Real-time visualization with checkpoint synchronization
- GPU support with automatic device assignment
- Single-line progress tracking for clean output

Performance:
- Medium resolution (19k nodes): 5-10 min for 60s simulation
- Fine resolution (46k nodes): 30-60 min for 60s simulation
- Adaptive timestep for numerical stability

Updated to use combined rocket_mesh.py module
"""

import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Process, Manager, Queue, Barrier
import threading
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import sys
import os
import warnings
import pickle
import shutil
import io
import contextlib

# Set matplotlib backend early
import matplotlib

try:
    matplotlib.use('TkAgg')
    INTERACTIVE_AVAILABLE = True
except:
    matplotlib.use('Agg')
    INTERACTIVE_AVAILABLE = False

import matplotlib.pyplot as plt

# Import from combined rocket_mesh module
from rocket_mesh import (
    HexahedralRocketMesh,
    RocketShapeParameters,
    RocketNoseGenerator,
    ThermalEquivalentProperties
)

# Import physics engine
from rocket_physics import RocketPhysicsEngine, SimulationConfig, MaterialProperties

# Import enhanced visualization if available
try:
    from rocket_visualization import VisualizationManager as EnhancedVisualizationManager
    ENHANCED_VIZ_AVAILABLE = True
except ImportError:
    ENHANCED_VIZ_AVAILABLE = False


class ProgressTracker:
    """Single-line progress tracker for parallel simulations"""

    def __init__(self, n_profiles=6, simulation_time=160.0):
        self.n_profiles = n_profiles
        self.simulation_time = simulation_time
        self.start_time = time.time()
        self.profile_status = {}
        self.profile_start_times = {}
        self.completed_count = 0
        self.terminal_width = shutil.get_terminal_size().columns
        self.current_sim_time = 0.0
        self.last_display_time = 0
        self.display_interval = 0.5  # Update every 0.5 seconds

        # Profile name abbreviations for compact display
        self.profile_abbrev = {
            'conical': 'CON',
            'ogive_falcon': 'OGV',
            'von_karman': 'VKM',
            'parabolic': 'PAR',
            'elliptical': 'ELL',
            'power_050': 'POW'
        }

    def start_profile(self, name):
        """Mark a profile as started"""
        self.profile_status[name] = 'running'
        self.profile_start_times[name] = time.time()

    def update_profile(self, name, progress_percent, sim_time=None):
        """Update progress for a profile"""
        if name in self.profile_status:
            elapsed = time.time() - self.profile_start_times.get(name, time.time())

            # Estimate time remaining
            if progress_percent > 0:
                total_estimated = elapsed / (progress_percent / 100.0)
                remaining = total_estimated - elapsed
                self.profile_status[name] = ('running', progress_percent, elapsed, remaining)
            else:
                self.profile_status[name] = ('running', progress_percent, elapsed, 0)

        if sim_time is not None:
            self.current_sim_time = max(self.current_sim_time, sim_time)

    def complete_profile(self, name, success=True):
        """Mark a profile as completed"""
        elapsed = time.time() - self.profile_start_times.get(name, time.time())
        self.profile_status[name] = ('done' if success else 'failed', 100, elapsed)
        self.completed_count += 1

    def display(self):
        """Display current progress on a single line"""
        current_time = time.time()

        # Rate limit display updates
        if current_time - self.last_display_time < self.display_interval:
            return

        self.last_display_time = current_time

        # Calculate overall metrics
        elapsed = current_time - self.start_time
        progress_pct = (self.completed_count / self.n_profiles) * 100

        # Count running profiles
        running = sum(1 for status in self.profile_status.values()
                      if isinstance(status, tuple) and status[0] == 'running')

        # Build compact status for each profile
        status_parts = []
        for name in ['conical', 'ogive_falcon', 'von_karman', 'parabolic', 'elliptical', 'power_050']:
            abbrev = self.profile_abbrev.get(name, name[:3])
            if name in self.profile_status:
                status = self.profile_status[name]
                if isinstance(status, tuple):
                    if status[0] == 'done':
                        status_parts.append(f"{abbrev}:✓")
                    elif status[0] == 'failed':
                        status_parts.append(f"{abbrev}:✗")
                    else:  # running
                        pct = status[1] if len(status) > 1 else 0
                        status_parts.append(f"{abbrev}:{pct:.0f}%")
            else:
                status_parts.append(f"{abbrev}:--")

        # Create single line status
        status_line = (f"\rSim t={self.current_sim_time:.1f}s | "
                       f"Progress: {progress_pct:.0f}% | "
                       f"Active: {running} | "
                       f"[{' '.join(status_parts)}] | "
                       f"Time: {elapsed:.1f}s")

        # Pad to terminal width to clear old text
        status_line = status_line.ljust(self.terminal_width - 1)

        # Write without newline
        sys.stdout.write(status_line)
        sys.stdout.flush()

    def final_display(self):
        """Display final completion status"""
        print()  # New line after progress
        elapsed = time.time() - self.start_time
        print(f"Completed: {self.completed_count}/{self.n_profiles} profiles in {elapsed:.1f}s")


def run_simulation_subprocess_with_checkpoints(name: str, shape_params: Dict, mesh_config: Dict,
                                               instance_id: int, device_id: int, result_file: str,
                                               output_dir: str, simulation_time: float,
                                               checkpoint_queue: Queue, barrier: Optional[Barrier],
                                               checkpoint_interval: float = 1.0,
                                               use_gpu: bool = True, debug_mode: bool = False):
    """
    Run simulation in separate process with checkpoint synchronization

    Architecture:
    - Mesh is recreated in subprocess to avoid pickling issues
    - Sends periodic checkpoints to main process for visualization
    - Handles GPU device assignment for parallel execution
    """

    try:
        # Initial status
        checkpoint_queue.put({
            'type': 'status',
            'name': name,
            'message': 'Initializing...'
        })

        # Configure compute device
        if use_gpu and torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cpu')

        # Send mesh creation status
        checkpoint_queue.put({
            'type': 'status',
            'name': name,
            'message': 'Creating mesh...'
        })

        # Create shape parameters from dictionary
        shape = RocketShapeParameters(
            nose_type=shape_params['nose_type'],
            nose_length=shape_params['nose_length'],
            nose_sharpness=shape_params.get('nose_sharpness', 0.7),
            nose_power=shape_params.get('nose_power', 0.5),
            nose_haack_c=shape_params.get('nose_haack_c', 0.0),
            body_radius=shape_params['body_radius'],
            wall_thickness=shape_params['wall_thickness'],
            variable_thickness=shape_params.get('variable_thickness', False),
            thickness_profile=shape_params.get('thickness_profile', None)
        )

        # Create mesh with suppressed output
        with contextlib.redirect_stdout(io.StringIO()):
            mesh = HexahedralRocketMesh(
                shape_params=shape,
                n_axial=mesh_config['n_axial'],
                n_circumferential=mesh_config['n_circumferential'],
                n_radial=mesh_config['n_radial'],
                nose_only=True,
                mesh_resolution=mesh_config.get('mesh_resolution', 'medium'),
                enable_parallel=False
            )

        # Send mesh completion status
        checkpoint_queue.put({
            'type': 'status',
            'name': name,
            'message': f'Mesh created ({mesh.quality_metrics.n_nodes:,} nodes)'
        })

        # Send mesh structure to main process for visualization
        mesh_structure = {
            'node_indices': mesh.node_indices.tolist(),
            'n_axial': mesh.n_axial,
            'n_circumferential': mesh.n_circumferential,
            'n_radial': mesh.n_radial
        }

        checkpoint_queue.put({
            'type': 'mesh',
            'name': name,
            'nodes': mesh.nodes.copy(),
            'shape_params': shape_params,
            'mesh_structure': mesh_structure
        })

        # Initialize physics engine
        checkpoint_queue.put({
            'type': 'status',
            'name': name,
            'message': 'Initializing physics...'
        })

        # Create simulation configuration
        config = SimulationConfig(
            time_step=0.01,
            adaptive_timestep=True,
            min_timestep=0.001,
            max_timestep=0.1,
            target_temp_change=1.0,
            fourier_safety_factor=0.45,
            mesh_resolution=mesh_config.get('mesh_resolution', 'medium'),
            enable_profiling=False
        )

        # Create physics engine with suppressed output
        with contextlib.redirect_stdout(io.StringIO()):
            engine = RocketPhysicsEngine(
                mesh=mesh,
                config=config,
                device_id=device_id,
                instance_id=instance_id
            )

        # Send simulation start status
        checkpoint_queue.put({
            'type': 'status',
            'name': name,
            'message': f'Starting simulation (0/{simulation_time}s)...'
        })

        # Main simulation loop
        start_time = time.time()
        checkpoint_time = 0.0
        real_time_checkpoint = time.time()
        states = []
        temperatures = []
        times = []
        min_dt_used = 1.0

        while engine.time < simulation_time:
            # Execute simulation step
            state = engine.step()

            # Track minimum timestep
            if 'dt' in state:
                min_dt_used = min(min_dt_used, state['dt'])

            # Store state data
            states.append(state.copy())
            temperatures.append(state['nose_max_temperature'])
            times.append(state['time'])

            # Send periodic progress updates
            if time.time() - real_time_checkpoint > 10.0:
                real_time_checkpoint = time.time()
                elapsed = time.time() - start_time
                sim_rate = engine.time / elapsed if elapsed > 0 else 0

                # Send lightweight progress update
                checkpoint_queue.put({
                    'type': 'progress',
                    'name': name,
                    'time': engine.time,
                    'progress': (engine.time / simulation_time) * 100,
                    'steps': engine.step_count,
                    'max_temp': state['nose_max_temperature'],
                    'sim_rate': sim_rate,
                    'min_dt': min_dt_used
                })

            # Send checkpoint at simulation time intervals
            if engine.time >= checkpoint_time + checkpoint_interval:
                checkpoint_time = engine.time

                # Get temperature field
                temp_field = engine.get_temperature_field().numpy()

                # Send checkpoint data
                checkpoint_queue.put({
                    'type': 'checkpoint',
                    'name': name,
                    'time': engine.time,
                    'state': state,
                    'temperature_field': temp_field,
                    'mesh_structure': mesh_structure,
                    'progress': (engine.time / simulation_time) * 100
                })

        # Save results
        computation_time = time.time() - start_time

        # Save to NPZ file with mesh data
        npz_file = result_file.replace('.json', '.npz')
        np.savez_compressed(
            npz_file,
            states=np.array(states),
            temperatures=np.array(temperatures),
            times=np.array(times),
            temperature_field=engine.get_temperature_field().numpy(),
            shape_params=shape_params,
            mesh_config=mesh_config,
            computation_time=computation_time,
            mesh_nodes=mesh.nodes,
            mesh_elements=mesh.hex_elements
        )

        # Save JSON summary
        result = {
            'profile': name,
            'max_temperature': float(np.max(temperatures)),
            'final_temperature': float(temperatures[-1]),
            'computation_time': computation_time,
            'n_steps': engine.step_count,
            'shape_params': shape_params,
            'mesh_config': mesh_config
        }

        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        # Send completion message
        checkpoint_queue.put({
            'type': 'complete',
            'name': name,
            'success': True
        })

        # Cleanup
        engine.cleanup()

    except Exception as e:
        checkpoint_queue.put({
            'type': 'error',
            'name': name,
            'error': str(e)
        })


class SimulationManager:
    """
    Central manager for parallel thermal FEA simulations

    Architecture:
    - Manages parallel execution of 6 nose profile simulations
    - Handles mesh generation or loading from existing parameters
    - Coordinates real-time visualization with simulation processes
    - Provides GPU/CPU resource management
    """

    def __init__(self,
                 output_dir: str = None,
                 mesh_resolution: str = 'medium',
                 simulation_time: float = 160.0,
                 visualization: bool = True,
                 checkpoint_interval: float = 1.0,
                 force_new_mesh: bool = False,
                 debug_mode: bool = False,
                 use_gpu: bool = True):
        """
        Initialize simulation manager

        Args:
            output_dir: Output directory (auto-generated if None)
            mesh_resolution: 'coarse', 'medium', or 'fine'
            simulation_time: Total simulation time in seconds
            visualization: Enable real-time visualization
            checkpoint_interval: Update interval in simulation time
            force_new_mesh: Force creation of new meshes
            debug_mode: Enable debug output
            use_gpu: Use GPU acceleration if available
        """
        self.mesh_resolution = mesh_resolution
        self.simulation_time = simulation_time
        self.visualization = visualization
        self.checkpoint_interval = checkpoint_interval
        self.expected_profiles = 6
        self.force_new_mesh = force_new_mesh
        self.debug_mode = debug_mode
        self.use_gpu = use_gpu

        # Setup output directory structure
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"simulation_6profiles_{timestamp}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.mesh_dir = self.output_dir / "meshes"
        self.results_dir = self.output_dir / "results"
        self.logs_dir = self.output_dir / "logs"

        for dir_path in [self.mesh_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.mesh_generator = None
        self.nose_profiles = {}
        self.meshes = {}
        self.engines = {}
        self.results = {}

        # Hardware detection
        self._detect_hardware()

        # Visualization manager
        self.viz_manager = None
        self.manager = None

    def _detect_hardware(self):
        """Detect and configure hardware capabilities"""
        self.n_cpu_cores = mp.cpu_count()
        self.n_physical_cores = os.cpu_count() or self.n_cpu_cores

        # Configure parallel processes based on available cores
        if self.n_physical_cores >= 12:
            self.max_parallel_processes = min(12, self.n_physical_cores - 2)
            self.optimal_parallel = 6
        else:
            self.max_parallel_processes = min(6, self.n_physical_cores - 1)
            self.optimal_parallel = min(6, self.max_parallel_processes)

        self.logger.info(f"CPU: {self.n_cpu_cores} logical cores, {self.n_physical_cores} physical cores")
        self.logger.info(f"Will use up to {self.optimal_parallel} parallel processes")

        # GPU detection and configuration
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.n_gpus = torch.cuda.device_count()
            for i in range(self.n_gpus):
                props = torch.cuda.get_device_properties(i)
                self.logger.info(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1e9:.1f} GB")
                self.gpu_memory_gb = props.total_memory / 1e9
                self.max_gpu_simulations = int(self.gpu_memory_gb / 3.5)
        else:
            self.logger.warning("No GPU detected - running on CPU")
            self.n_gpus = 0
            self.max_gpu_simulations = 0

    def _setup_logging(self):
        """Configure logging with reduced console verbosity"""
        log_file = self.logs_dir / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        self.logger = logging.getLogger('SimulationManager')
        self.logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)

        # File handler - full logging
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        # Console handler - warnings only for clean output
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # Initial log entry
        self.logger.info("=" * 60)
        self.logger.info("Rocket Thermal Simulation Manager")
        self.logger.info("=" * 60)
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Mesh resolution: {self.mesh_resolution.upper()}")
        self.logger.info(f"Simulation time: {self.simulation_time}s")
        self.logger.info(f"Visualization: {'Enabled' if self.visualization else 'Disabled'}")
        self.logger.info(f"GPU mode: {'Enabled' if self.use_gpu else 'Disabled'}")

    def find_latest_mesh_directory(self):
        """Locate the most recent mesh directory from previous runs"""
        patterns = ['rocket_mesh_*', 'mesh_*', 'nose_profiles_*', 'simulation_6profiles_*']
        search_dirs = [Path('.'), Path('..')]

        found_dirs = []
        for search_dir in search_dirs:
            if search_dir.exists():
                for pattern in patterns:
                    found_dirs.extend(search_dir.glob(pattern))

        # Filter to directories only and sort by modification time
        mesh_dirs = [d for d in found_dirs if d.is_dir()]

        if mesh_dirs:
            mesh_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for mesh_dir in mesh_dirs:
                if (mesh_dir / 'mesh_files').exists() or \
                        (mesh_dir / 'meshes').exists() or \
                        list(mesh_dir.glob('nose_profiles_data_*.json')) or \
                        list(mesh_dir.glob('mesh_6profiles.json')):
                    self.logger.info(f"Found existing mesh directory: {mesh_dir}")
                    return mesh_dir

        self.logger.info("No existing mesh directory found")
        return None

    def load_existing_meshes(self):
        """Load existing mesh parameters and recreate meshes for simulation"""
        if self.force_new_mesh:
            self.logger.info("Force new mesh option set - will create new meshes")
            return False

        mesh_source_dir = self.find_latest_mesh_directory()

        if not mesh_source_dir:
            self.logger.info("No existing mesh directory found")
            return False

        try:
            # Look for mesh metadata
            metadata_files = list(mesh_source_dir.glob('nose_profiles_data_*.json')) + \
                             list(mesh_source_dir.glob('mesh_6profiles.json')) + \
                             [mesh_source_dir / 'profile_parameters.json']

            metadata = None
            for mf in metadata_files:
                if mf.exists():
                    with open(mf, 'r') as f:
                        metadata = json.load(f)
                        self.logger.info(f"Loaded mesh metadata from: {mf.name}")
                        break

            if not metadata:
                self.logger.warning("No mesh metadata found in directory")
                return False

            # Extract mesh configuration
            if 'metadata' in metadata:
                mesh_info = metadata['metadata']

                if 'mesh_resolution' in mesh_info:
                    actual_resolution = mesh_info['mesh_resolution'].get('level', self.mesh_resolution)
                    if isinstance(actual_resolution, str):
                        found_resolution = actual_resolution.lower()

                        if found_resolution != self.mesh_resolution:
                            self.logger.warning(f"Found {found_resolution} meshes but requested {self.mesh_resolution}")
                            if not self.force_new_mesh:
                                response = input(f"Use existing {found_resolution} meshes? (y/n): ")
                                if response.lower() != 'y':
                                    return False
                                self.mesh_resolution = found_resolution

                    # Get grid parameters
                    self.n_axial = mesh_info['mesh_resolution'].get('n_axial', 120)
                    self.n_circumferential = mesh_info['mesh_resolution'].get('n_circumferential', 64)
                    self.n_radial = mesh_info['mesh_resolution'].get('n_radial', 6)
                else:
                    # Default parameters for fine mesh
                    self.n_axial = 120
                    self.n_circumferential = 64
                    self.n_radial = 6

            # Load profile parameters
            expected_profiles = ['conical', 'ogive_falcon', 'von_karman',
                                 'parabolic', 'elliptical', 'power_050']

            self.nose_profiles = {}

            self.logger.info("Recreating mesh geometries from parameters...")

            if 'profiles' in metadata:
                for name, profile_data in metadata['profiles'].items():
                    if name in expected_profiles:
                        # Create shape parameters
                        shape = RocketShapeParameters(
                            nose_type=profile_data.get('parameters', {}).get('nose_type', name.split('_')[0]),
                            nose_length=profile_data.get('nose_length_m', 5.0),
                            body_radius=profile_data.get('body_radius_m', 1.83),
                            wall_thickness=profile_data.get('wall_thickness_m', 0.015),
                            nose_sharpness=profile_data.get('parameters', {}).get('nose_sharpness', 0.7),
                            nose_power=profile_data.get('parameters', {}).get('nose_power', 0.75),
                            nose_haack_c=profile_data.get('parameters', {}).get('nose_haack_c', 0.0)
                        )

                        # Recreate mesh with suppressed output
                        with contextlib.redirect_stdout(io.StringIO()):
                            mesh = HexahedralRocketMesh(
                                shape_params=shape,
                                n_axial=self.n_axial,
                                n_circumferential=self.n_circumferential,
                                n_radial=self.n_radial,
                                nose_only=True,
                                mesh_resolution=self.mesh_resolution,
                                enable_parallel=False
                            )

                        self.nose_profiles[name] = {
                            'mesh': mesh,
                            'shape': shape,
                            'description': profile_data.get('description', ''),
                            'nodes': profile_data.get('n_nodes', 0),
                            'elements': profile_data.get('n_elements', 0)
                        }

                        self.meshes[name] = mesh

                        self.logger.debug(f"  Recreated {name}: {mesh.quality_metrics.n_nodes:,} nodes")

            if len(self.meshes) == 6:
                self.logger.info(f"Successfully loaded 6 mesh parameters from {mesh_source_dir.name}")
                self.logger.info(f"Mesh resolution: {self.mesh_resolution.upper()}")
                self.logger.info(f"Grid: {self.n_axial}×{self.n_circumferential}×{self.n_radial}")

                self._save_profile_parameters()
                return True
            else:
                self.logger.warning(f"Only loaded {len(self.meshes)}/6 profiles")
                return False

        except Exception as e:
            self.logger.error(f"Error loading existing meshes: {e}")
            if self.debug_mode:
                import traceback
                self.logger.debug(traceback.format_exc())
            return False

    def create_meshes(self):
        """Create or load meshes for 6 nose profiles"""
        if self.load_existing_meshes():
            print("\n[MESH] Using existing mesh parameters")
            return

        print(f"\n[MESH] Creating new meshes with {self.mesh_resolution.upper()} resolution...")

        # Create mesh generator
        self.mesh_generator = RocketNoseGenerator(
            base_radius=1.83,
            thermal_equivalent=True,
            mesh_resolution=self.mesh_resolution,
            output_dir=str(self.mesh_dir)
        )

        # Generate the 6 standard nose profiles
        self.nose_profiles = self.mesh_generator.create_nose_profiles()

        for name, profile_data in self.nose_profiles.items():
            self.meshes[name] = profile_data['mesh']
            self.logger.info(f"  Created {name}: {profile_data['nodes']:,} nodes")

        self._save_profile_parameters()
        self._save_mesh_metadata()

    def _save_mesh_metadata(self):
        """Save mesh metadata for future use"""
        metadata = {
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'mesh_resolution': {
                    'level': self.mesh_resolution,
                    'n_axial': self.mesh_generator.n_axial if self.mesh_generator else 120,
                    'n_circumferential': self.mesh_generator.n_circumferential if self.mesh_generator else 64,
                    'n_radial': self.mesh_generator.n_radial if self.mesh_generator else 6
                },
                'thermal_equivalent': True,
                'n_profiles': 6
            },
            'profiles': {}
        }

        for name, profile_data in self.nose_profiles.items():
            mesh = profile_data['mesh']
            metadata['profiles'][name] = {
                'nose_length_m': mesh.shape.nose_length,
                'body_radius_m': mesh.shape.body_radius,
                'wall_thickness_m': mesh.shape.wall_thickness,
                'n_nodes': profile_data.get('nodes', 0),
                'n_elements': profile_data.get('elements', 0),
                'description': profile_data.get('description', ''),
                'parameters': {
                    'nose_type': mesh.shape.nose_type,
                    'nose_sharpness': getattr(mesh.shape, 'nose_sharpness', None),
                    'nose_power': getattr(mesh.shape, 'nose_power', None),
                    'nose_haack_c': getattr(mesh.shape, 'nose_haack_c', None)
                }
            }

        metadata_file = self.output_dir / 'mesh_6profiles.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved mesh metadata to {metadata_file.name}")

    def _save_profile_parameters(self):
        """Save profile parameters for visualization"""
        profile_params = {}

        for name, profile_data in self.nose_profiles.items():
            mesh = profile_data['mesh']
            profile_params[name] = {
                'nose_length': mesh.shape.nose_length,
                'base_radius': mesh.shape.body_radius,
                'nose_sharpness': getattr(mesh.shape, 'nose_sharpness', 0.7),
                'nose_power': getattr(mesh.shape, 'nose_power', 0.5),
                'nose_haack_c': getattr(mesh.shape, 'nose_haack_c', 0.0),
                'wall_thickness': mesh.shape.wall_thickness,
                'nodes': profile_data.get('nodes', 0),
                'elements': profile_data.get('elements', 0)
            }

        params_file = self.output_dir / 'profile_parameters.json'
        with open(params_file, 'w') as f:
            json.dump(profile_params, f, indent=2)

    def run_simulations(self):
        """Execute parallel simulations with real-time visualization"""
        print("\n" + "=" * 60)
        print("Running 6 Parallel FEA Simulations")
        if self.visualization:
            print("Real-Time Visualization: ENABLED")
        print("=" * 60 + "\n")

        # Initialize progress tracker
        progress = ProgressTracker(n_profiles=len(self.meshes),
                                   simulation_time=self.simulation_time)

        # Setup visualization if enabled
        if self.visualization:
            if ENHANCED_VIZ_AVAILABLE:
                self.viz_manager = EnhancedVisualizationManager(
                    output_dir=self.output_dir,
                    mode='realtime',
                    show_plots=True,
                    debug=False  # Disable verbose output
                )
                self.viz_manager.setup_3d_dashboard()
                self.logger.info("Enhanced 3D visualization enabled")
            else:
                self.logger.warning("Enhanced visualization not available")
                self.viz_manager = None

        # Create multiprocessing components
        self.manager = Manager()
        checkpoint_queue = self.manager.Queue()

        # Calculate optimal process count
        n_simulations = len(self.meshes)
        if self.gpu_available and self.use_gpu:
            n_processes = min(n_simulations, self.max_gpu_simulations, self.optimal_parallel)
        else:
            n_processes = min(n_simulations, self.optimal_parallel)

        print(f"Starting {n_processes} parallel processes...")

        # Start simulation processes
        processes = []
        for idx, (name, mesh) in enumerate(self.meshes.items()):
            # Assign GPU device
            device_id = idx % self.n_gpus if self.gpu_available and self.use_gpu else 0

            # Extract parameters for subprocess
            shape_params = {
                'nose_type': str(mesh.shape.nose_type),
                'nose_length': float(mesh.shape.nose_length),
                'nose_sharpness': float(getattr(mesh.shape, 'nose_sharpness', 0.7)),
                'nose_power': float(getattr(mesh.shape, 'nose_power', 0.5)),
                'nose_haack_c': float(getattr(mesh.shape, 'nose_haack_c', 0.0)),
                'body_radius': float(mesh.shape.body_radius),
                'wall_thickness': float(mesh.shape.wall_thickness),
                'variable_thickness': bool(mesh.shape.variable_thickness),
                'thickness_profile': list(mesh.shape.thickness_profile) if mesh.shape.thickness_profile else None
            }

            mesh_config = {
                'n_axial': int(mesh.n_axial),
                'n_circumferential': int(mesh.n_circumferential),
                'n_radial': int(mesh.n_radial),
                'mesh_resolution': str(self.mesh_resolution)
            }

            result_file = str(self.results_dir / f"{name}_result.json")

            # Start process
            p = Process(
                target=run_simulation_subprocess_with_checkpoints,
                args=(name, shape_params, mesh_config, idx, device_id,
                      result_file, str(self.output_dir), self.simulation_time,
                      checkpoint_queue, None, self.checkpoint_interval,
                      self.use_gpu, self.debug_mode)
            )
            p.start()
            processes.append((p, name))
            progress.start_profile(name)

        # Monitor simulation progress
        checkpoint_data = {}
        meshes_received = set()
        completed = set()
        profile_progress = {name: 0.0 for name in self.meshes.keys()}

        # Show initial progress
        progress.display()

        while len(completed) < n_simulations:
            # Process checkpoint queue
            while not checkpoint_queue.empty():
                try:
                    data = checkpoint_queue.get(timeout=0.1)

                    if data['type'] == 'mesh' and self.viz_manager:
                        # Pass mesh to visualization
                        self.viz_manager.update_mesh(
                            data['name'],
                            data['nodes'],
                            data['shape_params'],
                            data.get('mesh_structure')
                        )
                        meshes_received.add(data['name'])

                    elif data['type'] == 'progress':
                        profile_progress[data['name']] = data['progress']
                        progress.update_profile(data['name'], data['progress'], data['time'])

                    elif data['type'] == 'checkpoint':
                        # Store checkpoint data
                        checkpoint_data[data['name']] = data
                        profile_progress[data['name']] = data['progress']
                        progress.update_profile(data['name'], data['progress'], data['time'])

                        # Update visualization
                        if self.viz_manager and 'temperature_field' in data:
                            self.viz_manager.update_profile_data(
                                data['name'],
                                data['state'],
                                temperature_field=data['temperature_field'],
                                mesh_nodes=None,
                                mesh_structure=data.get('mesh_structure')
                            )

                    elif data['type'] == 'complete':
                        completed.add(data['name'])
                        progress.complete_profile(data['name'])

                    elif data['type'] == 'error':
                        self.logger.error(f"Error in {data['name']}: {data.get('error')}")
                        completed.add(data['name'])
                        progress.complete_profile(data['name'], success=False)

                except:
                    break

            # Update visualization
            if self.viz_manager and len(checkpoint_data) > 0:
                for name, data in checkpoint_data.items():
                    if 'state' in data:
                        self.viz_manager.current_state[name] = data['state']

                self.viz_manager.update_3d_plots()
                checkpoint_data = {}

            # Update progress display
            progress.display()

            # Allow GUI updates
            if self.visualization and INTERACTIVE_AVAILABLE:
                plt.pause(0.01)

            time.sleep(0.1)

        # Wait for all processes to complete
        for p, name in processes:
            p.join()

        # Final progress display
        progress.final_display()

        # Load final results
        self.results = {}
        for name, _ in self.meshes.items():
            result_file = self.results_dir / f"{name}_result.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        self.results[name] = json.load(f)
                except:
                    pass

        print("\n[COMPLETE] All simulations finished")

    def run_post_visualization(self):
        """Run visualization as post-processing"""
        if not self.visualization:
            return

        print("\n" + "=" * 60)
        print("Post-Processing Visualization")
        print("=" * 60)

        try:
            if ENHANCED_VIZ_AVAILABLE:
                viz = EnhancedVisualizationManager(
                    output_dir=self.output_dir,
                    mode='post',
                    show_plots=True,
                    debug=False
                )

                viz.setup_3d_dashboard()

                if viz.load_and_display_results():
                    viz.update_3d_plots()
                    viz.save_final_plots()
                    viz.save_summary_report()

                    print("\nVisualization created successfully!")
                    print(f"Plots saved to: {viz.plots_dir}")

                    if viz.show_plots and INTERACTIVE_AVAILABLE:
                        print("\nVisualization window opened. Press Ctrl+C to close...")
                        try:
                            plt.show(block=True)
                        except KeyboardInterrupt:
                            pass
                else:
                    self.logger.error("No profiles could be loaded for visualization")
            else:
                self.logger.error("Enhanced visualization module not available")

        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            if self.debug_mode:
                import traceback
                self.logger.debug(traceback.format_exc())

    def save_summary(self):
        """Save simulation summary to JSON file"""
        summary_file = self.output_dir / "simulation_summary.json"

        total_comp_time = sum(r.get('computation_time', 0) for r in self.results.values() if 'error' not in r)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'mesh_resolution': self.mesh_resolution,
                'simulation_time': self.simulation_time,
                'gpu_available': self.gpu_available,
                'n_cpu_cores': self.n_cpu_cores,
                'n_gpus': self.n_gpus if hasattr(self, 'n_gpus') else 0
            },
            'performance': {
                'total_computation_time': total_comp_time,
                'n_simulations': len(self.results),
                'n_successful': len([r for r in self.results.values() if 'error' not in r]),
                'n_failed': len([r for r in self.results.values() if 'error' in r])
            },
            'profiles': self.results
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Summary saved to {summary_file}")

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'manager') and self.manager:
                try:
                    self.manager.shutdown()
                    time.sleep(0.5)
                except:
                    pass

            if self.gpu_available and self.use_gpu:
                torch.cuda.empty_cache()

        except Exception as e:
            if self.debug_mode:
                self.logger.debug(f"Cleanup error: {e}")

    def run(self):
        """Main execution method"""
        try:
            start_time = time.time()

            # Create manager
            self.manager = Manager()

            # Create or load meshes
            self.create_meshes()

            # Run simulations
            self.run_simulations()

            # Save summary
            self.save_summary()

            total_elapsed = time.time() - start_time

            print("\n" + "=" * 60)
            print("Simulation Complete!")
            print("=" * 60)
            print(f"Total runtime: {total_elapsed:.1f}s")
            print(f"Results saved to: {self.output_dir}")

            # Handle visualization
            if self.visualization:
                if ENHANCED_VIZ_AVAILABLE and self.viz_manager:
                    self.viz_manager.update_3d_plots()

                    if INTERACTIVE_AVAILABLE:
                        print("\nVisualization window will remain open.")
                        print("Close the window to exit.")
                        try:
                            plt.show(block=True)
                        except KeyboardInterrupt:
                            pass
                else:
                    # Run post-processing visualization
                    self.run_post_visualization()

        except Exception as e:
            self.logger.error(f"Simulation failed: {e}", exc_info=True)

        finally:
            self.cleanup()


def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(
        description='Rocket Thermal Simulation - FEA Analysis of 6 Nose Profiles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (auto-generated if not specified)')

    parser.add_argument('--resolution', type=str, default='medium',
                        choices=['coarse', 'medium', 'fine'],
                        help='Mesh resolution quality')

    parser.add_argument('--time', type=float, default=60.0,
                        help='Simulation time in seconds')

    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization')

    parser.add_argument('--new-mesh', action='store_true',
                        help='Force creation of new meshes')

    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU mode (disable GPU)')

    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ROCKET THERMAL ANALYSIS SYSTEM")
    print("FEA Simulation of 6 Nose Profiles")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  Mesh resolution: {args.resolution.upper()}")
    print(f"  Simulation time: {args.time}s")
    print(f"  Visualization: {'Disabled' if args.no_viz else 'Enabled'}")
    print(f"  GPU: {'Disabled' if args.cpu else 'Enabled (if available)'}")

    if not args.new_mesh:
        print("\nSearching for existing mesh files...")
    else:
        print("\nWill create new mesh files")

    # Create and run simulation manager
    manager = SimulationManager(
        output_dir=args.output_dir,
        mesh_resolution=args.resolution,
        simulation_time=args.time,
        visualization=not args.no_viz,
        checkpoint_interval=1.0,
        force_new_mesh=args.new_mesh,
        debug_mode=args.debug,
        use_gpu=not args.cpu
    )

    manager.run()


if __name__ == "__main__":
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Set environment variables for optimal performance
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    main()
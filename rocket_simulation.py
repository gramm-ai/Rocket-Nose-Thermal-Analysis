"""
rocket_simulation.py - Main Simulation Manager for Rocket Thermal Analysis
Optimized for 6 parallel simulations on 24-core CPU and GPU

Features:
- Optimized parallel simulation for 6 nose profiles
- GPU memory-aware batch processing
- Automatic mesh generation for 6 nose profiles
- Real-time monitoring and logging
- GPU-accelerated physics simulation
"""

import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Process, Manager
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

# Import rocket modules
from create_rocket_noses import RocketNoseGenerator
from rocket_physics import RocketPhysicsEngine, SimulationConfig, MaterialProperties
from rocket_mesh_hex import HexahedralRocketMesh, RocketShapeParameters


# Define simulation process function outside the class to avoid pickle issues
def run_simulation_subprocess(name: str, shape_params: Dict, mesh_config: Dict,
                              instance_id: int, device_id: int, result_file: str,
                              output_dir: str, simulation_time: float):
    """Run simulation in separate process - standalone function for multiprocessing"""
    try:
        # Set up environment for subprocess
        import sys
        import os
        import warnings
        warnings.filterwarnings('ignore')

        # Set environment variables for subprocess
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

        # Ensure proper imports in subprocess
        from pathlib import Path
        import json
        import numpy as np
        import torch
        import time

        # Recreate mesh in subprocess to avoid pickle issues
        from rocket_mesh_hex import HexahedralRocketMesh, RocketShapeParameters
        from rocket_physics import RocketPhysicsEngine, SimulationConfig

        # Recreate shape parameters
        shape = RocketShapeParameters(
            nose_type=shape_params['nose_type'],
            nose_length=shape_params['nose_length'],
            body_radius=shape_params['body_radius'],
            wall_thickness=shape_params['wall_thickness'],
            variable_thickness=shape_params['variable_thickness'],
            thickness_profile=shape_params['thickness_profile']
        )

        # Set optional parameters
        for key in ['nose_sharpness', 'nose_power', 'nose_haack_c']:
            if key in shape_params:
                setattr(shape, key, shape_params[key])

        # Recreate mesh
        mesh = HexahedralRocketMesh(
            shape_params=shape,
            n_axial=mesh_config['n_axial'],
            n_circumferential=mesh_config['n_circumferential'],
            n_radial=mesh_config['n_radial'],
            nose_only=True,
            mesh_resolution=mesh_config['mesh_resolution']
        )

        # Set process-specific GPU device
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)

        # Create simulation config
        config = SimulationConfig(
            time_step=0.01,
            adaptive_timestep=True,
            monitor_interval=10,
            use_mixed_precision=torch.cuda.is_available()
        )

        # Create physics engine
        engine = RocketPhysicsEngine(
            mesh=mesh,
            config=config,
            instance_id=instance_id,
            device_id=device_id if torch.cuda.is_available() else 0
        )

        # Run simulation
        states = []
        last_report_time = 0.0
        report_interval = simulation_time / 10
        start_time = time.time()

        while engine.time < simulation_time:
            state = engine.step()
            states.append(state.copy())

            # Progress reporting
            if engine.time - last_report_time >= report_interval:
                print(f"    [{name}] t={engine.time:6.1f}s: "
                      f"v={state['velocity']:4.0f} m/s, "
                      f"alt={state['altitude'] / 1000:5.1f} km, "
                      f"T_max={state['nose_max_temperature']:6.1f}°C")
                last_report_time = engine.time

        # Get final results
        monitor_data = engine.get_monitor_data()
        performance = engine.get_performance_stats()
        final_temps = engine.get_temperature_field().numpy()

        # Cleanup
        engine.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed_time = time.time() - start_time

        # Save result to file (JSON serializable data only)
        save_result = {
            'name': name,
            'computation_time': elapsed_time,
            'max_temperature': max([s['nose_max_temperature'] for s in states]) if states else 0,
            'final_temperature_max': states[-1]['nose_max_temperature'] if states else 0,
            'error': None
        }

        # Save NPZ file with full data
        npz_file = result_file.replace('.json', '.npz')
        np.savez_compressed(
            npz_file,
            states=states,
            monitor_data=monitor_data,
            temperature_field=final_temps,
            performance=performance
        )

        # Save JSON summary
        with open(result_file, 'w') as f:
            json.dump(save_result, f, indent=2)

        print(f"[Process {instance_id}] Completed {name}")

    except Exception as e:
        print(f"Error in simulation {name}: {e}")
        import traceback
        traceback.print_exc()

        # Save error result
        error_result = {
            'name': name,
            'error': str(e),
            'computation_time': 0,
            'max_temperature': 0,
            'final_temperature_max': 0
        }
        try:
            with open(result_file, 'w') as f:
                json.dump(error_result, f, indent=2)
        except:
            pass


class SimulationManager:
    """
    Manages thermal simulations for 6 rocket nose profiles
    Optimized for 24-core CPU and GPU utilization
    """

    def __init__(self,
                 output_dir: str = None,
                 mesh_resolution: str = 'medium',
                 simulation_time: float = 160.0,
                 parallel_mode: bool = True,  # Default to parallel
                 visualization_mode: bool = True,
                 log_level: str = 'INFO'):
        """
        Initialize simulation manager for 6 parallel simulations

        Args:
            output_dir: Output directory for results
            mesh_resolution: Mesh quality ('coarse', 'medium', 'fine')
            simulation_time: Total simulation time in seconds
            parallel_mode: Run simulations in parallel (default True for performance)
            visualization_mode: Enable visualization
            log_level: Logging level
        """
        self.mesh_resolution = mesh_resolution
        self.simulation_time = simulation_time
        self.parallel_mode = parallel_mode
        self.visualization_mode = visualization_mode
        self.expected_profiles = 6  # Optimized for 6 profiles

        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"simulation_6profiles_{timestamp}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup directories
        self.mesh_dir = self.output_dir / "meshes"
        self.results_dir = self.output_dir / "results"
        self.logs_dir = self.output_dir / "logs"

        for dir_path in [self.mesh_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging(log_level)

        # Initialize mesh generator
        self.mesh_generator = None
        self.nose_profiles = {}
        self.meshes = {}

        # Simulation engines and results
        self.engines = {}
        self.results = {}

        # Check hardware capabilities
        self._detect_hardware()

        # Visualization interface
        self.viz_queue = None
        self.viz_thread = None
        self.viz_manager = None
        self.manager = None  # Keep reference to multiprocessing Manager

    def _detect_hardware(self):
        """Detect and log hardware capabilities optimized for 6 simulations"""
        # CPU detection
        self.n_cpu_cores = mp.cpu_count()
        self.n_physical_cores = os.cpu_count() or self.n_cpu_cores

        # For 6 parallel simulations on 24-core CPU
        # Optimal configuration: use 6-12 cores (1-2 cores per simulation)
        if self.n_physical_cores >= 12:
            self.max_parallel_processes = min(12, self.n_physical_cores - 2)
            self.optimal_parallel = 6  # Optimal for 6 profiles
        else:
            self.max_parallel_processes = min(6, self.n_physical_cores - 1)
            self.optimal_parallel = min(6, self.max_parallel_processes)

        self.logger.info(f"CPU detected: {self.n_cpu_cores} logical cores, {self.n_physical_cores} physical cores")
        self.logger.info(f"Optimized for 6 parallel simulations")
        self.logger.info(f"Will use up to {self.optimal_parallel} parallel processes for 6 profiles")

        # GPU detection
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.n_gpus = torch.cuda.device_count()
            for i in range(self.n_gpus):
                props = torch.cuda.get_device_properties(i)
                self.logger.info(f"GPU {i}: {props.name}")
                self.logger.info(f"  Memory: {props.total_memory / 1e9:.1f} GB")
                self.logger.info(f"  CUDA Capability: {props.major}.{props.minor}")

                # Estimate how many simulations can fit in GPU memory
                # Assuming ~3-4GB per simulation for medium resolution
                self.gpu_memory_gb = props.total_memory / 1e9
                self.max_gpu_simulations = int(self.gpu_memory_gb / 3.5)
                self.logger.info(f"  Estimated capacity: {self.max_gpu_simulations} concurrent simulations")

                if self.max_gpu_simulations >= 6:
                    self.logger.info(f"  [OK] Sufficient for all 6 parallel simulations")
                else:
                    self.logger.info(
                        f"  [WARNING] May need to batch simulations (can run {self.max_gpu_simulations} concurrently)")
        else:
            self.logger.warning("No GPU detected - running on CPU (slower)")
            self.n_gpus = 0
            self.max_gpu_simulations = 0

    def _setup_logging(self, log_level: str):
        """Setup logging configuration"""
        log_file = self.logs_dir / f"simulation_6profiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Create logger
        self.logger = logging.getLogger('SimulationManager')
        self.logger.setLevel(getattr(logging, log_level))

        # File handler with UTF-8 encoding
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        # Console handler - use default encoding to avoid issues on Windows
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level))

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info("=" * 60)
        self.logger.info("Rocket Thermal Simulation Manager (6 Profiles)")
        self.logger.info("=" * 60)
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Mesh resolution: {self.mesh_resolution}")
        self.logger.info(f"Simulation time: {self.simulation_time}s")
        self.logger.info(f"Parallel mode: {self.parallel_mode}")
        self.logger.info(f"Visualization: {self.visualization_mode}")
        self.logger.info(f"Expected profiles: {self.expected_profiles}")

    def create_meshes(self):
        """Create meshes for 6 nose profiles"""
        self.logger.info("\nCreating 6 nose profile meshes...")

        # Initialize mesh generator
        self.mesh_generator = RocketNoseGenerator(
            base_radius=1.83,  # Falcon 9 radius
            thermal_equivalent=True,
            mesh_resolution=self.mesh_resolution,
            output_dir=str(self.mesh_dir)
        )

        # Generate 6 profiles
        self.nose_profiles = self.mesh_generator.create_nose_profiles()

        # Verify we have exactly 6 profiles
        if len(self.nose_profiles) != self.expected_profiles:
            self.logger.warning(f"Expected {self.expected_profiles} profiles, got {len(self.nose_profiles)}")

        # Extract meshes
        for name, profile_data in self.nose_profiles.items():
            self.meshes[name] = profile_data['mesh']
            self.logger.info(f"  Created {name}: {profile_data['nodes']:,} nodes, "
                             f"{profile_data['elements']:,} elements")

        # Export mesh data
        self.mesh_generator.export_profiles_data(self.nose_profiles,
                                                 filename="mesh_6profiles.json")

        self.logger.info(f"[OK] Created {len(self.meshes)} mesh profiles")

    def run_sequential(self):
        """Run 6 simulations sequentially (fallback mode)"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Running 6 Sequential Simulations")
        self.logger.info("=" * 60)

        # Create manager if not already created
        if not self.manager:
            self.manager = Manager()

        all_results = {}

        profile_list = list(self.meshes.items())
        n_profiles = len(profile_list)

        for idx, (name, mesh) in enumerate(profile_list):
            self.logger.info(f"\n[{idx + 1}/{n_profiles}] Simulating {name}...")

            # Run single simulation with visualization updates
            result = self._run_single_simulation_sequential(name, mesh, idx)
            all_results[name] = result

            # Save results to file
            result_file = self.results_dir / f"{name}_result.npz"
            np.savez_compressed(
                result_file,
                states=result.get('states', []),
                monitor_data=result.get('monitor_data', {}),
                temperature_field=result.get('final_temperature', []),
                performance=result.get('performance', {})
            )

            # Log summary
            self._log_simulation_summary(name, result)

        self.results = all_results
        self.logger.info(f"\n[COMPLETE] {n_profiles} sequential simulations complete")

    def _run_single_simulation_sequential(self, name: str, mesh, instance_id: int, device_id: int = 0) -> Dict:
        """Run a single simulation in sequential mode with visualization updates"""
        start_time = time.time()

        # Create simulation config
        config = SimulationConfig(
            time_step=0.01,
            adaptive_timestep=True,
            monitor_interval=10,
            use_mixed_precision=torch.cuda.is_available()
        )

        # Create physics engine
        engine = RocketPhysicsEngine(
            mesh=mesh,
            config=config,
            instance_id=instance_id,
            device_id=device_id if torch.cuda.is_available() else 0
        )

        # Run simulation
        states = []
        last_report_time = 0.0
        report_interval = self.simulation_time / 10  # 10 reports total

        while engine.time < self.simulation_time:
            state = engine.step()
            states.append(state.copy())

            # Progress reporting
            if engine.time - last_report_time >= report_interval:
                self.logger.info(f"    [{name}]\tt={engine.time:4.2f}s: "
                                 f"v={state['velocity']:4.0f} m/s, "
                                 f"alt={state['altitude'] / 1000:5.1f} km, "
                                 f"T_max={state['nose_max_temperature']:6.1f}°C")
                last_report_time = engine.time

                # Send to visualization if enabled
                if self.visualization_mode and self.viz_queue:
                    try:
                        self.viz_queue.put_nowait({
                            'type': 'update',
                            'name': name,
                            'state': state,
                            'temperature_field': None
                        })
                    except:
                        pass  # Queue might be full

        # Get final results
        monitor_data = engine.get_monitor_data()
        performance = engine.get_performance_stats()
        final_temps = engine.get_temperature_field().numpy()

        # Cleanup
        engine.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed_time = time.time() - start_time

        return {
            'name': name,
            'states': states,
            'monitor_data': monitor_data,
            'final_temperature': final_temps,
            'performance': performance,
            'computation_time': elapsed_time,
            'max_temperature': max([s['nose_max_temperature'] for s in states]) if states else 0,
            'final_temperature_max': states[-1]['nose_max_temperature'] if states else 0
        }

    def run_parallel(self):
        """Run 6 simulations in parallel with optimized resource usage"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Running 6 Parallel Simulations (Optimized)")
        self.logger.info("=" * 60)

        # Calculate optimal number of parallel processes for 6 simulations
        n_simulations = len(self.meshes)

        if n_simulations != self.expected_profiles:
            self.logger.warning(f"Expected {self.expected_profiles} profiles, running {n_simulations}")

        if self.gpu_available:
            # GPU mode: balance between GPU memory and CPU cores
            n_processes = min(
                n_simulations,
                self.max_gpu_simulations,  # GPU memory limit
                self.optimal_parallel  # Optimal for 6 profiles
            )
            self.logger.info(f"GPU mode: Using {n_processes} parallel processes for {n_simulations} profiles")
            if n_processes < n_simulations:
                self.logger.info(f"  Will run in batches due to resource constraints")
        else:
            # CPU-only mode: use optimal cores for 6 simulations
            n_processes = min(n_simulations, self.optimal_parallel)
            self.logger.info(f"CPU mode: Using {n_processes} parallel processes for {n_simulations} profiles")

        # Profile list
        profile_items = list(self.meshes.items())

        # Track active processes
        active_processes = []
        completed = 0

        # Process all simulations
        profile_index = 0

        while profile_index < len(profile_items) or active_processes:
            # Start new processes if we have capacity
            while len(active_processes) < n_processes and profile_index < len(profile_items):
                name, mesh = profile_items[profile_index]

                # Assign GPU device (round-robin if multiple GPUs)
                if self.gpu_available:
                    device_id = profile_index % self.n_gpus
                else:
                    device_id = 0

                # Extract shape parameters to avoid pickle issues
                # Convert all values to basic Python types
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
                    'mesh_resolution': str(getattr(mesh, 'mesh_resolution', 'medium'))
                }

                # Create output path for this process
                result_file = str(self.results_dir / f"{name}_result.json")

                # Use standalone function to avoid pickle issues
                p = Process(
                    target=run_simulation_subprocess,
                    args=(name, shape_params, mesh_config, profile_index, device_id,
                          result_file, str(self.output_dir), self.simulation_time)
                )
                p.start()
                active_processes.append((p, name, time.time()))

                self.logger.info(f"  Started: {name} (profile {profile_index + 1}/{n_simulations})")
                if self.gpu_available:
                    self.logger.info(f"    Assigned to GPU {device_id}")

                profile_index += 1

                # Small delay to stagger process starts
                time.sleep(0.1)

            # Check for completed processes
            still_active = []
            for p, name, start_time in active_processes:
                if p.is_alive():
                    still_active.append((p, name, start_time))
                else:
                    p.join()
                    completed += 1
                    elapsed = time.time() - start_time
                    self.logger.info(f"  Completed: {name} ({completed}/{n_simulations}) in {elapsed:.1f}s")

            active_processes = still_active

            # Short sleep to prevent CPU spinning
            time.sleep(0.5)

        # Load results from files
        all_results = {}
        for name, _ in profile_items:
            result_file = self.results_dir / f"{name}_result.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        result = json.load(f)
                    all_results[name] = result
                    self._log_simulation_summary(name, result)
                except Exception as e:
                    self.logger.error(f"Failed to load results for {name}: {e}")

        self.results = all_results

        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"[COMPLETE] {n_simulations} parallel simulations complete")
        self.logger.info(f"  Total simulations: {n_simulations}")
        self.logger.info(f"  Successful: {len(all_results)}")
        self.logger.info(f"  Parallel processes used: {n_processes}")
        if n_simulations == self.expected_profiles:
            self.logger.info(f"  [OK] Successfully completed all {self.expected_profiles} profile simulations")
        self.logger.info("=" * 60)

    def _log_simulation_summary(self, name: str, result: Dict):
        """Log simulation summary"""
        if 'error' in result:
            self.logger.error(f"  {name}: FAILED - {result['error']}")
        else:
            self.logger.info(f"  {name}: "
                             f"Max T={result['max_temperature']:.1f}°C, "
                             f"Final T={result['final_temperature_max']:.1f}°C, "
                             f"Time={result['computation_time']:.1f}s")

    def start_visualization(self):
        """Start visualization process for 6 profiles"""
        if not self.visualization_mode:
            return

        self.logger.info("\nStarting visualization interface for 6 profiles...")

        # Create manager if not already created and keep reference
        if not self.manager:
            self.manager = Manager()

        # Create communication queue with larger size
        self.viz_queue = self.manager.Queue(maxsize=1000)  # Larger queue for parallel updates

        # Import here to avoid issues with multiprocessing
        from rocket_visualization import VisualizationManager

        # Create visualization manager
        self.viz_manager = VisualizationManager(
            queue=self.viz_queue,
            output_dir=self.output_dir
        )

        # Start in separate thread for non-blocking
        self.viz_thread = threading.Thread(target=self._run_visualization_safely, daemon=True)
        self.viz_thread.start()

        self.logger.info("[OK] Visualization interface started for 6 profiles")

    def _run_visualization_safely(self):
        """Run visualization with exception handling"""
        try:
            self.viz_manager.run()
        except Exception as e:
            self.logger.error(f"Visualization error: {e}")
            # Don't propagate the error to avoid crashing the main simulation

    def save_summary(self):
        """Save simulation summary with performance metrics for 6 profiles"""
        summary_file = self.output_dir / "simulation_summary_6profiles.json"

        # Calculate total computation time
        total_comp_time = sum(r.get('computation_time', 0) for r in self.results.values() if 'error' not in r)

        # Calculate parallel efficiency safely
        if self.parallel_mode:
            # Get list of valid computation times
            valid_comp_times = [r.get('computation_time', 0) for r in self.results.values() if
                                'error' not in r and r.get('computation_time', 0) > 0]

            if valid_comp_times:
                # If we have valid times, calculate efficiency
                max_comp_time = max(valid_comp_times)
                parallel_efficiency = total_comp_time / (
                            max_comp_time * len(self.results)) if max_comp_time > 0 else 0.0
            else:
                # No valid computation times available
                parallel_efficiency = 0.0
        else:
            # Sequential mode always has efficiency of 1.0
            parallel_efficiency = 1.0

        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_profiles': self.expected_profiles,
            'configuration': {
                'mesh_resolution': self.mesh_resolution,
                'simulation_time': self.simulation_time,
                'parallel_mode': self.parallel_mode,
                'gpu_available': self.gpu_available,
                'n_cpu_cores': self.n_cpu_cores,
                'n_gpus': self.n_gpus,
                'optimal_parallel': self.optimal_parallel
            },
            'performance': {
                'total_computation_time': total_comp_time,
                'n_simulations': len(self.results),
                'n_successful': len([r for r in self.results.values() if 'error' not in r]),
                'n_failed': len([r for r in self.results.values() if 'error' in r]),
                'avg_time_per_simulation': total_comp_time / max(
                    len([r for r in self.results.values() if 'error' not in r]), 1),
                'parallel_efficiency': parallel_efficiency
            },
            'profiles': {}
        }

        # Add results for each profile
        for name, result in self.results.items():
            if 'error' not in result:
                # Check if nose_profiles exists and has the profile
                if hasattr(self, 'nose_profiles') and name in self.nose_profiles:
                    nodes = self.nose_profiles[name].get('nodes', 0)
                    elements = self.nose_profiles[name].get('elements', 0)
                else:
                    nodes = 0
                    elements = 0

                summary['profiles'][name] = {
                    'max_temperature': result.get('max_temperature', 0),
                    'final_temperature': result.get('final_temperature_max', 0),
                    'computation_time': result.get('computation_time', 0),
                    'nodes': nodes,
                    'elements': elements
                }
            else:
                # Log failed profiles
                summary['profiles'][name] = {
                    'error': result.get('error', 'Unknown error'),
                    'max_temperature': 0,
                    'final_temperature': 0,
                    'computation_time': 0,
                    'nodes': 0,
                    'elements': 0
                }

        # Rank profiles by max temperature (only successful ones)
        successful_profiles = {k: v for k, v in summary['profiles'].items() if 'error' not in v}

        if successful_profiles:
            ranked = sorted(successful_profiles.items(),
                            key=lambda x: x[1]['max_temperature'])
            summary['ranking'] = {
                'coolest': ranked[0][0],
                'hottest': ranked[-1][0],
                'coolest_temp': ranked[0][1]['max_temperature'],
                'hottest_temp': ranked[-1][1]['max_temperature']
            }

        # Save the summary
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"\n[SAVED] Summary saved to {summary_file}")
        except Exception as e:
            self.logger.error(f"Failed to save summary: {e}")

        # Print performance metrics
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PERFORMANCE METRICS (6 PROFILES)")
        self.logger.info("=" * 60)
        self.logger.info(f"Total computation time: {total_comp_time:.1f}s")
        self.logger.info(
            f"Successful simulations: {summary['performance']['n_successful']}/{summary['performance']['n_simulations']}")

        if summary['performance']['n_failed'] > 0:
            self.logger.warning(f"Failed simulations: {summary['performance']['n_failed']}")

        self.logger.info(f"Average per simulation: {summary['performance']['avg_time_per_simulation']:.1f}s")

        if self.parallel_mode and parallel_efficiency > 0:
            self.logger.info(f"Parallel efficiency: {parallel_efficiency:.1%}")

        # Print ranking if available
        if 'ranking' in summary:
            self.logger.info("\n" + "=" * 60)
            self.logger.info("THERMAL PERFORMANCE RANKING (6 PROFILES)")
            self.logger.info("=" * 60)

            # Sort all results, handling both successful and failed
            all_profiles = []
            for name, result in self.results.items():
                if 'error' not in result:
                    all_profiles.append(
                        (name, result.get('max_temperature', 0), result.get('final_temperature_max', 0), None))
                else:
                    all_profiles.append((name, 0, 0, result.get('error', 'Unknown error')))

            # Sort by max temperature (failed ones will be at the bottom with 0 temperature)
            all_profiles.sort(key=lambda x: x[1], reverse=True)

            rank = 1
            for name, max_temp, final_temp, error in all_profiles:
                if error is None:
                    self.logger.info(f"{rank:2d}. {name:15s}: "
                                     f"Max={max_temp:6.1f}°C, "
                                     f"Final={final_temp:6.1f}°C")
                    rank += 1
                else:
                    self.logger.error(f" X. {name:15s}: FAILED - {error[:50]}")

            if summary['performance']['n_successful'] == self.expected_profiles:
                self.logger.info(f"\n[OK] All {self.expected_profiles} profiles ranked successfully")
            else:
                self.logger.warning(
                    f"\n[WARNING] Only {summary['performance']['n_successful']}/{self.expected_profiles} profiles completed successfully")

    def run(self):
        """Main execution method"""
        try:
            start_total = time.time()

            # Create manager at the start for the entire simulation
            self.manager = Manager()

            # Create meshes (6 profiles)
            self.create_meshes()

            # Start visualization if enabled
            if self.visualization_mode:
                self.start_visualization()
                time.sleep(2)  # Give visualization time to start

            # Run simulations
            if self.parallel_mode:
                self.run_parallel()
            else:
                self.run_sequential()

            # Save summary - now with proper error handling
            self.save_summary()

            total_elapsed = time.time() - start_total

            self.logger.info("\n" + "=" * 60)
            self.logger.info("Simulation Complete!")
            self.logger.info("=" * 60)
            self.logger.info(f"Total runtime: {total_elapsed:.1f}s")
            self.logger.info(f"Results saved to: {self.output_dir}")

            # Check if all simulations completed successfully
            n_successful = len([r for r in self.results.values() if 'error' not in r])
            if n_successful == self.expected_profiles:
                self.logger.info(f"[OK] Successfully completed all {self.expected_profiles} profile simulations")
            else:
                self.logger.warning(
                    f"[WARNING] Completed {n_successful}/{self.expected_profiles} simulations successfully")

            # Keep visualization running if enabled and successful
            if self.visualization_mode and hasattr(self, 'viz_thread') and self.viz_thread:
                self.logger.info("\nVisualization is running. Press Ctrl+C to exit...")
                try:
                    while self.viz_thread.is_alive():
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.logger.info("\nShutting down visualization...")

        except Exception as e:
            self.logger.error(f"Simulation failed: {e}", exc_info=True)
            # Don't re-raise here, handle cleanup instead

        finally:
            # Cleanup - more robust error handling
            try:
                # Send shutdown signal to visualization if it exists
                if hasattr(self, 'viz_queue') and self.viz_queue:
                    try:
                        self.viz_queue.put_nowait({'type': 'shutdown'})
                    except:
                        pass  # Queue might be closed or full

                    time.sleep(0.5)  # Give visualization time to shutdown

                # Wait for visualization thread to finish
                if hasattr(self, 'viz_thread') and self.viz_thread and self.viz_thread.is_alive():
                    self.viz_thread.join(timeout=2.0)  # Don't wait forever

                # Shutdown manager properly
                if hasattr(self, 'manager') and self.manager:
                    try:
                        self.manager.shutdown()
                        time.sleep(0.5)  # Give manager time to cleanup
                    except:
                        pass  # Manager might already be shut down

                # Clear GPU cache if used
                if hasattr(self, 'gpu_available') and self.gpu_available:
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except:
                        pass  # GPU cleanup is optional

            except Exception as cleanup_error:
                # Don't let cleanup errors crash the program
                self.logger.debug(f"Cleanup error (non-critical): {cleanup_error}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Rocket Thermal Simulation Manager - Optimized for 6 Parallel Simulations'
    )

    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--mesh-resolution', type=str, default='medium',
                        choices=['coarse', 'medium', 'fine'],
                        help='Mesh resolution quality')
    parser.add_argument('--simulation-time', type=float, default=60.0,
                        help='Total simulation time in seconds')
    parser.add_argument('--sequential', action='store_true',
                        help='Run simulations sequentially (default is parallel)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization (text-only mode)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ROCKET THERMAL ANALYSIS SYSTEM")
    print("Optimized for 6 Parallel Nose Profile Simulations")
    print("=" * 60)

    # Create and run simulation manager
    manager = SimulationManager(
        output_dir=args.output_dir,
        mesh_resolution=args.mesh_resolution,
        simulation_time=args.simulation_time,
        parallel_mode=not args.sequential,  # Default to parallel
        visualization_mode=not args.no_viz,
        log_level=args.log_level
    )

    manager.run()


if __name__ == "__main__":
    # Set multiprocessing start method for Windows compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    # Set OMP threads to prevent oversubscription
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    main()
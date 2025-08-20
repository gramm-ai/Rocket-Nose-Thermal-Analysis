# Rocket Nose Thermal Analysis

Read Full Article:
[Full Article](https://medium.com/@jisoo_63794/from-stable-diffusion-to-rocket-science-accelerating-physics-simulations-with-ai-712a2150679f)

## 🎯 Project Vision: From Stable Diffusion to Rocket Science

This project explores an intriguing question: could the principles behind Stable Diffusion's image generation be applied to physics simulations? Specifically, could deep neural networks be trained to emulate physics processes in a way that permits much larger time steps than traditional finite-element analysis (FEA), while preserving accuracy?

### The Process: FEA → APE → GenAI + APE

Traditional Finite Element Analysis (FEA) for complex systems can take hours — or even days — to complete. Engineers often spend months exploring design spaces. But what if we could train a Physics Engine DNN on thousands of FEA runs, producing an Accelerated Physics Engine (APE) capable of predicting physical behavior in a single forward pass?

**This repository is Step 1:** A GPU-accelerated differential physics FEA system designed to generate high-fidelity training data for the next phase.

### Why Rocket Nose Thermal Analysis?

As a test case, I've chosen a thermodynamic problem: simulating the thermal heating at the top of Falcon-9 rocket during the first 60 seconds of launch, examining how aerodynamic heating differs between sharp and rounded profiles. This provides:

- **Complex Physics**: Four coupled heat transfer mechanisms
- **Design Relevance**: Direct impact on spacecraft survivability
- **Rich Dataset**: 36,000 timesteps across 6 profiles for AI training
- **Validation Data**: Publicly available Falcon 9 specifications

### The Three-Phase Journey

1. **Current Phase (This Repo)**: High-fidelity FEA simulation with GPU acceleration
2. **Next Phase**: Train DNN on FEA results to create Accelerated Physics Engine
3. **Final Phase**: Integrate APE into generative AI for natural language design

Imagine a system where an engineer can simply describe their requirements in natural language: "Design a nose cone that minimizes peak temperature while maintaining a drag coefficient below 0.45 and fitting within a 5-meter envelope"

## 🚀 Overview

Advanced Finite Element Analysis (FEA) system for thermal analysis of rocket nose cone designs during atmospheric flight. This GPU-accelerated implementation integrates all four heat transfer mechanisms simultaneously, solving the coupled differential equations at every node in the mesh, at every timestep. For a fine-resolution mesh with ~120,000 nodes and 60 seconds of simulation time at 0.01s timesteps, this means approximately 720 million calculations per profile.

**Performance Achievement**: What would take more than 5–20 hours with traditional methods now completes in 5 minutes for a single nose profile through strategic optimizations.

**Key Features:**
- **GPU-Accelerated Physics**: Optimized for NVIDIA RTX 3090 with CUDA support
- **Adaptive Time Stepping**: Fourier-number-based stability control for accurate heat diffusion
- **Parallel Processing**: Simultaneous simulation of 6 nose profiles using multiprocessing
- **Real-Time Visualization**: Live 3D temperature distribution with FEA mesh overlay
- **High-Fidelity Meshes**: Hexahedral elements with 19,200-46,080 nodes per profile
- **Thermal Equivalent Modeling**: Accounts for structural mass (stringers, frames) in heat capacity

<br>
<img width="1024" height="710" alt="nose_profiles_comparison_20250810_011300" src="https://github.com/user-attachments/assets/d284a2dc-ef3d-4629-bc3c-6a75a571af4a" />

<br>

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION MANAGER                       │
│                  (rocket_simulation.py)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐     ┌───────────────┐   ┌──────────────┐  │
│  │ Mesh Engine  │     │Physics Engine │   │Visualization │  │
│  │(rocket_mesh) │ ──► (rocket_physics)  ──►(rocket_viz)  │  │
│  └──────────────┘     └───────────────┘   └──────────────┘  │
│         │                    │                  │           │
│    Hexahedral         GPU-Accelerated       Flight Time     │
│     Mesh Gen            Heat Transfer        Heat Monitor   │
└─────────────────────────────────────────────────────────────┘
```

## 🔬 Nose Profiles Analyzed

After 60 seconds of flight simulation, thermal performance varies significantly:

| Profile | Type | Length | Max Temp | Characteristics |
|---------|------|--------|----------|-----------------|
| **Elliptical** | Blunt | 3.5m | 245.9°C | Best heat distribution |
| **Von Karman** | Optimized | 7.0m | 251.0°C | Minimum drag design |
| **Ogive (Falcon 9)** | Standard | 6.5m | 251.9°C | Baseline reference |
| **Parabolic** | Smooth | 5.5m | 252.4°C | Gradual transition |
| **Power Series** | n=0.5 | 5.0m | 249.6°C | Enhanced curvature |
| **Conical** | Sharp | 4.0m | 262.1°C | Maximum heating |

The elliptical profile, despite its shorter length (4m vs 6.5m for ogive), demonstrates superior heat distribution, validating the importance of shape optimization beyond simple drag reduction.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 24GB+ GPU memory recommended (RTX 3090 or better)
- 32GB+ system RAM

### Dependencies
```bash
pip install numpy scipy matplotlib torch torchvision
pip install multiprocessing pathlib datetime
```

### Quick Start
```bash
# Clone the repository
git clone https://github.com/gramm-ai/Rocket-Nose-Thermal-Analysis.git
cd Rocket-Nose-Thermal-Analysis

# Run the simulation (60 seconds flight time, medium resolution)
python rocket_simulation.py --time 60 --resolution medium

# For high-fidelity simulation
python rocket_simulation.py --time 160 --resolution fine --no-viz
```

## 🏃 Running Simulations

### Command Line Options
```bash
python rocket_simulation.py [OPTIONS]

Options:
  --output-dir PATH     Output directory (auto-generated if not specified)
  --resolution LEVEL    Mesh resolution: coarse|medium|fine (default: medium)
  --time SECONDS        Simulation time in seconds (default: 60.0)
  --no-viz             Disable real-time visualization
  --new-mesh           Force creation of new meshes
  --cpu                Force CPU mode (disable GPU)
  --debug              Enable debug output
```

### Mesh Resolution Details

| Resolution | Nodes/Profile | Elements | Time Step | Runtime (60s) |
|------------|--------------|----------|-----------|---------------|
| **Coarse** | ~8,000 | ~5,600 | 10ms | 2-3 min |
| **Medium** | ~19,200 | ~13,440 | 5ms | 5-10 min |
| **Fine** | ~46,080 | ~32,256 | 2ms | 30-60 min |

## 📈 Physics Simulation

During launch, a rocket nose cone experiences a violent symphony of physical phenomena:

### 1. Aerodynamic Heating (Stagnation & Boundary Layer)

Air molecules can't get out of the way fast enough. They pile up at the stagnation point, converting kinetic energy into heat through compression.

**Recovery Temperature:**
```
T_recovery = T_ambient × (1 + r × (γ-1)/2 × M²)
```
where:
- `T_ambient` = Atmospheric temperature (K)
- `r` = Recovery factor (0.87-0.90 for turbulent flow)
- `γ` = Specific heat ratio (1.4 for air)
- `M` = Mach number

**Stagnation Point Heat Flux (Sutton-Graves):**
```
q_stag = K_sg × √(ρ/R_nose) × V³
```
where:
- `K_sg` = Sutton-Graves constant (1.1×10⁻⁴)
- `ρ` = Air density (kg/m³)
- `R_nose` = Nose radius (m)
- `V` = Velocity (m/s)

### 2. Heat Conduction (Fourier's Law)

The accumulating heat doesn't stay at the surface. It conducts through the aluminum-lithium alloy structure following Fourier's law.

**3D Transient Heat Equation:**
```
∂T/∂t = α × ∇²T = α × (∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²)
```
where:
- `α` = Thermal diffusivity = `k/(ρ×cp)` (m²/s)
- `k` = Thermal conductivity (120 W/m·K for Al-Li)
- `ρ` = Density (2700 kg/m³)
- `cp` = Specific heat (900 J/kg·K)

**FEM Discretization:**
```
[M]{Ṫ} + [K]{T} = {Q}
```
where:
- `[M]` = Mass matrix
- `[K]` = Stiffness (conductivity) matrix
- `{Q}` = Heat flux vector

### 3. Radiation Cooling (Stefan-Boltzmann)

Following Stefan-Boltzmann law, the hot surface radiates energy to the cold sky.

**Radiation Heat Flux:**
```
q_rad = ε × σ × (T_surface⁴ - T_sky⁴)
```
where:
- `ε` = Surface emissivity (0.8 for oxidized aluminum)
- `σ` = Stefan-Boltzmann constant (5.67×10⁻⁸ W/m²·K⁴)
- `T_surface` = Surface temperature (K)
- `T_sky` = Sky temperature (K)

### 4. Convection (Forced & Natural)

**Convective Heat Transfer:**
```
q_conv = h × (T_surface - T_ambient)
```

**Heat Transfer Coefficient (Turbulent Flow):**
```
Nu = 0.037 × Re^0.8 × Pr^(1/3)
h = Nu × k_air / L_characteristic
```
where:
- `Nu` = Nusselt number
- `Re` = Reynolds number = `ρVL/μ`
- `Pr` = Prandtl number (≈0.7 for air)
- `k_air` = Thermal conductivity of air

### Adaptive Time Stepping

The simulation uses multiple stability criteria:

**Fourier Number Criterion:**
```
Fo = α × Δt / L² < 0.4
Δt_fourier = 0.4 × L² / α
```

**Temperature Change Criterion:**
```
Δt_temp = ΔT_target / max(|dT/dt|)
```

**Optimal Time Step:**
```
Δt = min(Δt_fourier, Δt_temp, Δt_max)
```

### Material Properties (Al-Li 2195)
- Thermal Conductivity: k = 120 W/(m·K)
- Density: ρ = 2700 kg/m³
- Specific Heat: cp = 900 J/(kg·K)
- Thermal Diffusivity: α = k/(ρ×cp) = 4.94×10⁻⁵ m²/s
- Emissivity: ε = 0.8
- Melting Point: 620°C (893 K)

## 📊 Visualization System

### Real-Time 3D Display
- Temperature distribution with color mapping (blue=cold, red=hot)
- Black mesh overlay for structural visibility
- Flight status display (altitude, velocity, Mach number)
- Per-profile statistics (max, mean, std deviation)

<br>
<img width="947" height="883" alt="rocket_nose_thermal_distr" src="https://github.com/user-attachments/assets/b480ba00-eff5-47e7-a701-b587e6534d1f" />
<br>
<br>

### Post-Processing
```python
# Generate standalone visualization from results
python rocket_visualization.py simulation_6profiles_YYYYMMDD_HHMMSS/
```

### 📁 Project Structure

```
Rocket-Nose-Thermal-Analysis/
├── rocket_simulation.py      # Main simulation manager
├── rocket_mesh.py            # Mesh generation & nose profiles
├── rocket_physics.py         # GPU-accelerated physics engine
├── rocket_visualization.py   # 3D visualization system
├── README.md                 # This file
└── simulation_*/             # Output directories
    ├── meshes/              # Mesh files (VTK format)
    ├── results/             # Simulation results (NPZ, JSON)
    ├── plots/               # Visualization outputs
    └── logs/                # Simulation logs
```




## 🎯 The Road Ahead: From FEA to GenAI

The key insight from SR3 and similar diffusion-based super-resolution models was about discovering that intermediate steps contain learnable patterns that can be distilled into direct mappings.

Consider how progressive distillation works in super-resolution: a teacher network trained on 1000-step denoising learns the complete trajectory from noise to image. A student network then learns to predict not the next step, but perhaps 10 steps ahead, using the teacher’s intermediate states as training targets. Through multiple rounds of this process, researchers compressed 1000 steps → 100 steps → 10 steps → eventually a single forward pass. The same principle may apply to FEA time-stepping when the underlying physics is dominated by differential process such as diffusion, where our “teacher” is the full-resolution simulation with millisecond timesteps (in this rocket example), and our “student” learns to predict states at tens to hundred millisecond or even second-scale intervals.

The challenge unique to physics simulation is maintaining conservation laws and numerical stability. While a slightly incorrect pixel in super-resolution might be imperceptible, violating energy conservation in thermal simulation leads to unphysical temperature divergence. (·•᷄‎ࡇ•᷅ )

This suggests we likely need a way to combine the learned components with hard physical constraint and not just through a cost function but a method of introducing and enforcing one or more hard constraints.

Imagine a network that predicts temperature changes in a learned latent space, but then projects these predictions back through a physics-constrained (not physics-informed ) decoder that explicitly enforces energy balance. The latent space would capture the essential dynamics such as heat flow patterns, boundary layer effects, material responses while the decoder acts to ensure that every prediction respects fundamental thermodynamics.

For additional discussions please read the full article  here:
[Article](https://medium.com/@jisoo_63794/from-stable-diffusion-to-rocket-science-accelerating-physics-simulations-with-ai-712a2150679f)

## 📚 References

1. Anderson, J.D. (2006). *Hypersonic and High-Temperature Gas Dynamics*
2. Sutton, K., & Graves, R.A. (1971). *A General Stagnation-Point Convective Heating Equation*
3. NASA TM-2010-216293: *Thermal Analysis of Aerospace Structures*
4. AIAA 2018-4693: *Falcon 9 Structural Design Considerations*
5. Lee, J. (2024). "From Stable Diffusion to Rocket Science: Accelerating Physics Simulations with AI"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Run with profiling
python rocket_simulation.py --debug --resolution coarse --time 10
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team who are accelerating the ML research for the entire earth community
- SpaceX for publicly available Falcon 9 specifications that informed our model parameters
- NASA for technical reports and validation data
- Open-source FEA community for numerical methods

## 📧 Contact

**Author**: Jisoo Lee, PhD, Managing Director / Founder, Fairbuild (www.fairb.com)

For questions or collaboration:
- GitHub Issues: [Project Issues](https://github.com/gramm-ai/Rocket-Nose-Thermal-Analysis/issues)
- Medium: [@jisoo_63794](https://medium.com/@jisoo_63794)

---

**Last Updated**: December 2024  
**Version**: 2.0.0 (GPU-Accelerated FEA with Adaptive Time Stepping)  
**Status**: Active Development - Phase 1 of FEA→APE→GenAI Pipeline  
**Next Phase**: Training the Accelerated Physics Engine (Coming Soon)

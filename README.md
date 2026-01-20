# Electron Configuration Dynamics - DFT Phase Space Analysis

**Proof of Concept:** Treating electron configurations as a dynamical system using DFT-calculated energies.

## What This Does

1. **Generate configs:** Creates various electron configurations for elements H-Ne (Z=1-10)
2. **Run DFT:** Calculates total energy for each configuration using PySCF
3. **Analyze:** Builds phase space, finds stable attractors (noble gases, half-filled shells), generates visualizations

## The Physics

Instead of just memorizing "noble gases are stable," we model electron configurations as a dynamical system with:
- **State variables:** (n_eff, l_eff, S_total, E)
- **Fixed points:** Stable configurations (He, Ne, etc.)
- **Phase portraits:** Trajectories flowing toward ground states
- **Emergent stability:** Shell structure from first principles

## Setup

```bash
# Create environment
mamba create -n electron-dft python=3.11 -y
mamba activate electron-dft

# Install dependencies
pip install -r requirements.txt

# Create data directory
mkdir -p data/figures
```

## Usage

```bash
# Step 1: Generate electron configurations (~10 per element, Z=1-10)
python generate_configs.py
# Output: data/configs.json

# Step 2: Run DFT calculations (parallelized across 32 cores)
python run_dft.py
# Output: data/results.json
# Time: ~10 hours on Threadripper 7970X

# Step 3: Analyze phase space and generate plots
python analyze.py
# Output: data/figures/*.png
```

## Expected Outputs

**Phase 1 (Proof of Concept):**
- 100 DFT calculations (10 elements × 10 configs)
- Energy database mapping (n,l,S,Z) → E
- Phase space visualization showing noble gas attractors
- Stability analysis confirming He, Ne as fixed points

**Validation:**
- Compare DFT energies to NIST ionization energies
- Confirm noble gases have negative eigenvalues (stable)
- Verify half-filled configurations show secondary stability

## Hardware Requirements

**Minimum:** 16 GB RAM, 4 cores
**Recommended:** 64+ GB RAM, 16+ cores (Phase 2 scaling)
**Your rig:** 128 GB RAM, 32 cores - perfect for this

## What's Next

**Phase 2:** Scale to Z=1-86, 100 configs per element
**Phase 3:** GPU acceleration with TeraChem for large systems
**Phase 4:** Predictive modeling for superheavy elements (Z>118)

## Theory

See docs/theory.md for mathematical formulation of the dynamical system.

## License

MIT

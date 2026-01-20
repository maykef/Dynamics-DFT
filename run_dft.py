#!/usr/bin/env python3
"""
Run DFT calculations for all electron configurations.

Reads data/configs.json and runs PySCF calculations in parallel.
Saves results to data/results.json.

Expected runtime: ~10 hours on Threadripper 7970X (32 cores).
"""

import json
import time
from pathlib import Path
from datetime import timedelta
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    from pyscf import gto, dft
except ImportError:
    print("ERROR: PySCF not installed!")
    print("Install with: pip install pyscf")
    exit(1)


# DFT Settings
DFT_FUNCTIONAL = 'B3LYP'  # Hybrid functional, good accuracy
BASIS_SET = '6-31g*'       # Double-zeta with polarization
CONVERGENCE = 1e-8         # SCF convergence threshold
MAX_CYCLES = 100           # Maximum SCF iterations


def setup_atom(Z, symbol, occupation, S_total):
    """
    Setup PySCF atom object with specific electron configuration.
    
    Args:
        Z: atomic number
        symbol: element symbol
        occupation: orbital occupation vector [1s, 2s, 2p, 3s, 3p]
        S_total: total spin
    
    Returns:
        PySCF Mole object
    """
    # Calculate total electrons
    n_electrons = sum(occupation)
    
    # Calculate spin multiplicity (2S + 1)
    # S_total is in units of ℏ/2, so multiplicity = 2*S_total + 1
    spin = int(round(2 * S_total))  # Number of unpaired electrons
    
    # Create molecule object (atom at origin)
    mol = gto.Mole()
    mol.atom = f'{symbol} 0 0 0'
    mol.basis = BASIS_SET
    mol.charge = Z - n_electrons  # Ionization state
    mol.spin = spin  # Number of unpaired electrons
    mol.verbose = 0  # Suppress output
    
    try:
        mol.build()
        return mol
    except Exception as e:
        print(f"Warning: Could not build molecule for {symbol} with occupation {occupation}")
        print(f"  Error: {e}")
        return None


def run_single_dft(config):
    """
    Run DFT calculation for a single configuration.
    
    Args:
        config: dictionary with configuration data
    
    Returns:
        dict with results (energy, convergence, etc.)
    """
    Z = config['Z']
    symbol = config['symbol']
    occupation = config['occupation']
    S_total = config['S_total']
    config_string = config['config_string']
    
    start_time = time.time()
    
    try:
        # Setup atom
        mol = setup_atom(Z, symbol, occupation, S_total)
        if mol is None:
            return {
                **config,
                'energy': None,
                'converged': False,
                'error': 'Failed to build molecule',
                'runtime': 0,
            }
        
        # Run DFT calculation
        # Use unrestricted KS for open-shell systems
        if mol.spin > 0:
            mf = dft.UKS(mol)  # Unrestricted Kohn-Sham
        else:
            mf = dft.RKS(mol)  # Restricted Kohn-Sham
        
        mf.xc = DFT_FUNCTIONAL
        mf.conv_tol = CONVERGENCE
        mf.max_cycle = MAX_CYCLES
        
        # Run SCF
        energy = mf.kernel()
        converged = mf.converged
        
        # Extract orbital energies
        if hasattr(mf.mo_energy, '__len__'):
            # Unrestricted: average alpha and beta orbital energies
            if isinstance(mf.mo_energy, tuple):
                homo_energy = float(max(mf.mo_energy[0][mf.mo_occ[0] > 0].max(),
                                       mf.mo_energy[1][mf.mo_occ[1] > 0].max()))
            else:
                homo_energy = float(mf.mo_energy[mf.mo_occ > 0].max())
        else:
            homo_energy = None
        
        runtime = time.time() - start_time
        
        return {
            **config,
            'energy': float(energy) if converged else None,
            'homo_energy': homo_energy,
            'converged': bool(converged),
            'runtime': round(runtime, 2),
            'error': None,
        }
        
    except Exception as e:
        runtime = time.time() - start_time
        return {
            **config,
            'energy': None,
            'converged': False,
            'error': str(e),
            'runtime': round(runtime, 2),
        }


def estimate_total_time(sample_runtimes, total_configs, n_jobs):
    """Estimate total runtime based on sample calculations."""
    avg_time = np.mean(sample_runtimes)
    total_time = (avg_time * total_configs) / n_jobs
    return timedelta(seconds=int(total_time))


def main():
    """Main execution."""
    print("=" * 70)
    print("DFT CALCULATION ENGINE")
    print("=" * 70)
    
    # Load configurations
    config_file = Path('data/configs.json')
    if not config_file.exists():
        print(f"ERROR: {config_file} not found!")
        print("Run: python generate_configs.py first")
        exit(1)
    
    with open(config_file, 'r') as f:
        configs = json.load(f)
    
    print(f"\nLoaded {len(configs)} configurations from {config_file}")
    print(f"\nDFT Settings:")
    print(f"  Functional: {DFT_FUNCTIONAL}")
    print(f"  Basis set: {BASIS_SET}")
    print(f"  Convergence: {CONVERGENCE}")
    print(f"  Max cycles: {MAX_CYCLES}")
    
    # Detect number of cores
    import os
    n_jobs = int(os.environ.get('N_CORES', -1))  # -1 = use all cores
    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    
    print(f"\nParallelization:")
    print(f"  CPU cores: {n_jobs}")
    
    # Run quick test to estimate time
    print(f"\nRunning test calculations to estimate runtime...")
    test_configs = configs[:min(5, len(configs))]
    test_results = []
    for config in test_configs:
        result = run_single_dft(config)
        test_results.append(result)
        if result['converged']:
            print(f"  {result['symbol']} ({result['config_string']}): "
                  f"{result['energy']:.6f} Ha in {result['runtime']:.1f}s")
    
    # Estimate total time
    test_times = [r['runtime'] for r in test_results if r['converged']]
    if test_times:
        estimated_time = estimate_total_time(test_times, len(configs), n_jobs)
        print(f"\nEstimated total runtime: {estimated_time}")
        print(f"  (Based on {len(test_times)} successful test calculations)")
    
    # Confirm before proceeding
    print("\n" + "=" * 70)
    response = input("Proceed with full calculation? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        exit(0)
    
    print("\n" + "=" * 70)
    print("STARTING FULL DFT CALCULATIONS")
    print("=" * 70)
    print("This will take a while. Progress bar shows completed calculations.")
    print("You can safely interrupt (Ctrl+C) and resume later (not implemented yet).")
    print()
    
    start_time = time.time()
    
    # Run all calculations in parallel with progress bar
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(run_single_dft)(config) 
        for config in tqdm(configs, desc="DFT calculations", unit="config")
    )
    
    total_time = time.time() - start_time
    
    # Save results
    output_file = Path('data/results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Statistics
    print("\n" + "=" * 70)
    print("CALCULATION COMPLETE")
    print("=" * 70)
    
    converged = [r for r in results if r['converged']]
    failed = [r for r in results if not r['converged']]
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nStatistics:")
    print(f"  Total calculations: {len(results)}")
    print(f"  Converged: {len(converged)} ({100*len(converged)/len(results):.1f}%)")
    print(f"  Failed: {len(failed)} ({100*len(failed)/len(results):.1f}%)")
    print(f"  Total runtime: {timedelta(seconds=int(total_time))}")
    print(f"  Avg time/calc: {total_time/len(results):.1f}s")
    
    if converged:
        energies = [r['energy'] for r in converged]
        print(f"\nEnergy statistics (converged calculations):")
        print(f"  Min: {min(energies):.6f} Ha")
        print(f"  Max: {max(energies):.6f} Ha")
        print(f"  Mean: {np.mean(energies):.6f} Ha")
    
    if failed:
        print(f"\nFailed calculations:")
        for r in failed[:10]:  # Show first 10
            print(f"  {r['symbol']} ({r['config_string']}): {r['error']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more")
    
    # Energy per element
    print(f"\nGround state energies:")
    for Z in sorted(set(r['Z'] for r in results)):
        element_results = [r for r in converged if r['Z'] == Z and r['is_ground_state']]
        if element_results:
            r = element_results[0]
            print(f"  {r['symbol']:>2} (Z={Z:>2}): {r['energy']:>12.6f} Ha  ({r['config_string']})")
    
    print("\n" + "=" * 70)
    print("Next step: python analyze.py")
    print("=" * 70)


if __name__ == '__main__':
    main()

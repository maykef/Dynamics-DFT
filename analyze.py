#!/usr/bin/env python3
"""
Analyze DFT results and generate phase space visualizations.

Reads data/results.json, fits energy landscape, finds stable configurations,
and generates phase portraits showing electron dynamics.

Usage:
    python analyze.py              # Analyze full results
    python analyze.py --test       # Analyze test results
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.interpolate import Rbf
from sklearn.linear_model import LinearRegression
import matplotlib.patheffects as path_effects


def load_results(test_mode=False):
    """Load DFT results from JSON file."""
    if test_mode:
        results_file = Path('data/results_test.json')
    else:
        results_file = Path('data/results.json')
    
    if not results_file.exists():
        print(f"ERROR: {results_file} not found!")
        if test_mode:
            print("Run: python run_dft.py --test")
        else:
            print("Run: python run_dft.py")
        exit(1)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Filter to only converged calculations
    converged = [r for r in results if r.get('converged', False)]
    
    print(f"Loaded {len(converged)} converged calculations from {results_file}")
    
    return converged


def extract_phase_space_coords(results):
    """
    Extract phase space coordinates (n, l, S, E) from results.
    
    Returns:
        Arrays of n_eff, l_eff, S_total, energy, Z, symbols
    """
    n_vals = []
    l_vals = []
    S_vals = []
    E_vals = []
    Z_vals = []
    symbols = []
    is_ground = []
    
    for r in results:
        n_vals.append(r['n_eff'])
        l_vals.append(r['l_eff'])
        S_vals.append(r['S_total'])
        E_vals.append(r['energy'])
        Z_vals.append(r['Z'])
        symbols.append(r['symbol'])
        is_ground.append(r['is_ground_state'])
    
    return (np.array(n_vals), np.array(l_vals), np.array(S_vals), 
            np.array(E_vals), np.array(Z_vals), np.array(symbols), 
            np.array(is_ground))


def fit_energy_landscape(n, l, S, Z, E):
    """
    Fit smooth energy surface E(n, l, S, Z).
    
    Uses polynomial regression for simplicity.
    """
    print("\nFitting energy landscape E(n, l, S, Z)...")
    
    # Create feature matrix
    X = np.column_stack([
        n, l, S, Z,
        n**2, l**2, S**2, Z**2,
        n*l, n*S, l*S,
        n*Z, l*Z, S*Z,
    ])
    
    # Fit linear model
    model = LinearRegression()
    model.fit(X, E)
    
    # Compute R²
    r2 = model.score(X, E)
    print(f"  R² = {r2:.4f}")
    
    # Predict energies
    E_pred = model.predict(X)
    rmse = np.sqrt(np.mean((E - E_pred)**2))
    print(f"  RMSE = {rmse:.6f} Ha")
    
    return model


def find_stable_configs(results):
    """Identify stable configurations (ground states, noble gases, etc.)."""
    stable = []
    
    # Noble gases (Z = 2, 10, 18, ...)
    noble_Z = [2, 10, 18, 36, 54, 86]
    
    for r in results:
        if r['is_ground_state']:
            stability = 'ground'
            if r['Z'] in noble_Z:
                stability = 'noble_gas'
            
            stable.append({
                'symbol': r['symbol'],
                'Z': r['Z'],
                'n': r['n_eff'],
                'l': r['l_eff'],
                'S': r['S_total'],
                'E': r['energy'],
                'config': r['config_string'],
                'stability': stability,
            })
    
    return stable


def create_3d_phase_portrait(n, l, S, E, Z, symbols, is_ground, stable_configs, output_dir):
    """Create 3D phase space visualization (n, l, S)."""
    print("\nGenerating 3D phase portrait...")
    
    fig = plt.figure(figsize=(20, 16), facecolor='#0a0a14')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a14')
    
    # Color by energy
    scatter = ax.scatter(n, l, S, c=E, s=80, cmap='plasma', 
                        alpha=0.6, edgecolors='white', linewidths=0.5)
    
    # Highlight ground states
    ground_mask = is_ground
    ax.scatter(n[ground_mask], l[ground_mask], S[ground_mask], 
              s=200, c=E[ground_mask], cmap='plasma',
              marker='*', edgecolors='yellow', linewidths=2, alpha=1, zorder=10)
    
    # Label stable configs
    for config in stable_configs[:10]:  # Label first 10
        ax.text(config['n'], config['l'], config['S'], 
               f"  {config['symbol']}", 
               fontsize=9, color='yellow', weight='bold', zorder=15)
    
    # Styling
    ax.set_xlabel('n (Principal QN)', fontsize=14, color='white', weight='bold', labelpad=10)
    ax.set_ylabel('l (Angular Momentum)', fontsize=14, color='white', weight='bold', labelpad=10)
    ax.set_zlabel('S (Total Spin)', fontsize=14, color='white', weight='bold', labelpad=10)
    ax.set_title('ELECTRON CONFIGURATION PHASE SPACE\nGround States (★) as Attractors',
                fontsize=20, color='white', weight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Energy (Hartree)', rotation=270, labelpad=25, color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Grid and axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#1a1a2e')
    ax.yaxis.pane.set_edgecolor('#1a1a2e')
    ax.zaxis.pane.set_edgecolor('#1a1a2e')
    ax.grid(True, alpha=0.15, color='white')
    ax.tick_params(colors='white', labelsize=10)
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    output_file = output_dir / 'phase_space_3d.png'
    plt.savefig(output_file, dpi=300, facecolor='#0a0a14', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")


def create_energy_landscape(n, l, S, E, Z, symbols, is_ground, output_dir):
    """Create energy vs n and l projections."""
    print("\nGenerating energy landscape plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor='#0a0a14')
    
    # Plot 1: E vs n (colored by Z)
    scatter1 = ax1.scatter(n, E, c=Z, s=100, cmap='viridis', 
                          alpha=0.7, edgecolors='white', linewidths=0.5)
    ax1.scatter(n[is_ground], E[is_ground], s=300, 
               marker='*', c=Z[is_ground], cmap='viridis',
               edgecolors='yellow', linewidths=2, zorder=10)
    
    ax1.set_xlabel('n (Principal Quantum Number)', fontsize=14, color='white', weight='bold')
    ax1.set_ylabel('Energy (Hartree)', fontsize=14, color='white', weight='bold')
    ax1.set_title('Energy vs Shell (n)\nGround states marked with ★',
                 fontsize=16, color='white', weight='bold', pad=15)
    ax1.grid(True, alpha=0.2, color='white')
    ax1.tick_params(colors='white', labelsize=11)
    ax1.set_facecolor('#0a0a14')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#4ECDC4')
        spine.set_linewidth(2)
    
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Atomic Number (Z)', rotation=270, labelpad=20, color='white', fontsize=11)
    cbar1.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white')
    
    # Plot 2: E vs l (colored by Z)
    scatter2 = ax2.scatter(l, E, c=Z, s=100, cmap='viridis',
                          alpha=0.7, edgecolors='white', linewidths=0.5)
    ax2.scatter(l[is_ground], E[is_ground], s=300,
               marker='*', c=Z[is_ground], cmap='viridis',
               edgecolors='yellow', linewidths=2, zorder=10)
    
    ax2.set_xlabel('l (Angular Momentum)', fontsize=14, color='white', weight='bold')
    ax2.set_ylabel('Energy (Hartree)', fontsize=14, color='white', weight='bold')
    ax2.set_title('Energy vs Orbital Type (l)\ns=0, p=1, d=2, f=3',
                 fontsize=16, color='white', weight='bold', pad=15)
    ax2.grid(True, alpha=0.2, color='white')
    ax2.tick_params(colors='white', labelsize=11)
    ax2.set_facecolor('#0a0a14')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#FFD93D')
        spine.set_linewidth(2)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Atomic Number (Z)', rotation=270, labelpad=20, color='white', fontsize=11)
    cbar2.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')
    
    plt.tight_layout()
    output_file = output_dir / 'energy_landscape.png'
    plt.savefig(output_file, dpi=300, facecolor='#0a0a14', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")


def create_stability_diagram(stable_configs, output_dir):
    """Create diagram showing stable configurations by element."""
    print("\nGenerating stability diagram...")
    
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='#0a0a14')
    ax.set_facecolor('#0a0a14')
    
    # Group by Z
    Z_values = sorted(set(c['Z'] for c in stable_configs))
    
    for i, Z in enumerate(Z_values):
        configs = [c for c in stable_configs if c['Z'] == Z]
        
        for config in configs:
            # Color by stability type
            color = '#00FF41' if config['stability'] == 'noble_gas' else '#4ECDC4'
            marker = '*' if config['stability'] == 'noble_gas' else 'o'
            size = 400 if config['stability'] == 'noble_gas' else 200
            
            ax.scatter([Z], [config['E']], s=size, c=color, marker=marker,
                      edgecolors='white', linewidths=2, alpha=0.9, zorder=10)
            
            # Label
            label = f"{config['symbol']}\n{config['config']}"
            ax.text(Z, config['E'] - 0.5, label, 
                   fontsize=8, color=color, ha='center', va='top',
                   weight='bold', zorder=11)
    
    ax.set_xlabel('Atomic Number (Z)', fontsize=14, color='white', weight='bold')
    ax.set_ylabel('Ground State Energy (Hartree)', fontsize=14, color='white', weight='bold')
    ax.set_title('STABLE ELECTRON CONFIGURATIONS\n★ Noble Gases (Closed Shells)',
                fontsize=18, color='white', weight='bold', pad=20)
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white', labelsize=11)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#00FF41')
        spine.set_linewidth(2)
    
    plt.tight_layout()
    output_file = output_dir / 'stability_diagram.png'
    plt.savefig(output_file, dpi=300, facecolor='#0a0a14', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")


def create_summary_table(stable_configs, output_dir):
    """Generate text summary of stable configurations."""
    print("\nGenerating summary table...")
    
    summary = []
    summary.append("=" * 80)
    summary.append("STABLE ELECTRON CONFIGURATIONS - SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    summary.append(f"{'Z':<4} {'Symbol':<6} {'Config':<20} {'n_eff':<8} {'l_eff':<8} {'S':<8} {'Energy (Ha)':<15} {'Type'}")
    summary.append("-" * 80)
    
    for config in stable_configs:
        stability_type = "NOBLE GAS" if config['stability'] == 'noble_gas' else "Ground"
        summary.append(
            f"{config['Z']:<4} {config['symbol']:<6} {config['config']:<20} "
            f"{config['n']:<8.3f} {config['l']:<8.3f} {config['S']:<8.3f} "
            f"{config['E']:<15.6f} {stability_type}"
        )
    
    summary.append("=" * 80)
    
    summary_text = "\n".join(summary)
    print(summary_text)
    
    # Save to file
    output_file = output_dir / 'summary.txt'
    with open(output_file, 'w') as f:
        f.write(summary_text)
    
    print(f"\nSaved: {output_file}")


def main():
    """Main execution."""
    # Check for test mode
    test_mode = '--test' in sys.argv
    
    print("=" * 70)
    if test_mode:
        print("PHASE SPACE ANALYSIS - TEST MODE")
    else:
        print("PHASE SPACE ANALYSIS")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path('data/figures')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load results
    results = load_results(test_mode)
    
    if len(results) < 3:
        print("\nERROR: Not enough converged calculations for analysis!")
        print(f"  Found: {len(results)} converged")
        print(f"  Need: At least 3")
        exit(1)
    
    # Extract phase space coordinates
    n, l, S, E, Z, symbols, is_ground = extract_phase_space_coords(results)
    
    print(f"\nPhase space dimensions:")
    print(f"  n range: [{n.min():.2f}, {n.max():.2f}]")
    print(f"  l range: [{l.min():.2f}, {l.max():.2f}]")
    print(f"  S range: [{S.min():.2f}, {S.max():.2f}]")
    print(f"  E range: [{E.min():.6f}, {E.max():.6f}] Ha")
    print(f"  Z range: [{int(Z.min())}, {int(Z.max())}]")
    
    # Fit energy landscape
    energy_model = fit_energy_landscape(n, l, S, Z, E)
    
    # Find stable configurations
    stable_configs = find_stable_configs(results)
    print(f"\nIdentified {len(stable_configs)} stable configurations:")
    for config in stable_configs:
        print(f"  {config['symbol']:>2} (Z={config['Z']:>2}): {config['config']:<15} "
              f"E={config['E']:>10.6f} Ha  [{config['stability']}]")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    create_3d_phase_portrait(n, l, S, E, Z, symbols, is_ground, stable_configs, output_dir)
    create_energy_landscape(n, l, S, E, Z, symbols, is_ground, output_dir)
    create_stability_diagram(stable_configs, output_dir)
    create_summary_table(stable_configs, output_dir)
    
    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nGenerated files in {output_dir}:")
    print(f"  • phase_space_3d.png       - 3D phase portrait")
    print(f"  • energy_landscape.png     - Energy vs n and l")
    print(f"  • stability_diagram.png    - Stable configs by element")
    print(f"  • summary.txt              - Text summary table")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()

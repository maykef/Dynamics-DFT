#!/usr/bin/env python3
"""
Generate electron configurations for DFT calculations.

Creates ground state + excited state configurations for elements H through Ne (Z=1-10).
Saves to data/configs.json for processing by run_dft.py.
"""

import json
import os
from pathlib import Path
from itertools import combinations


# Element data
ELEMENTS = [
    {'Z': 1, 'symbol': 'H', 'name': 'Hydrogen'},
    {'Z': 2, 'symbol': 'He', 'name': 'Helium'},
    {'Z': 3, 'symbol': 'Li', 'name': 'Lithium'},
    {'Z': 4, 'symbol': 'Be', 'name': 'Beryllium'},
    {'Z': 5, 'symbol': 'B', 'name': 'Boron'},
    {'Z': 6, 'symbol': 'C', 'name': 'Carbon'},
    {'Z': 7, 'symbol': 'N', 'name': 'Nitrogen'},
    {'Z': 8, 'symbol': 'O', 'name': 'Oxygen'},
    {'Z': 9, 'symbol': 'F', 'name': 'Fluorine'},
    {'Z': 10, 'symbol': 'Ne', 'name': 'Neon'},
]

# Orbital order: 1s, 2s, 2p, 3s, 3p (enough for Ne)
# Each orbital can hold: s=2, p=6
ORBITAL_NAMES = ['1s', '2s', '2p', '3s', '3p']
ORBITAL_CAPACITIES = [2, 2, 6, 2, 6]  # max electrons per orbital


def aufbau_filling(n_electrons):
    """Generate ground state configuration using Aufbau principle."""
    occupation = [0] * len(ORBITAL_NAMES)
    remaining = n_electrons
    
    for i, capacity in enumerate(ORBITAL_CAPACITIES):
        if remaining == 0:
            break
        occupation[i] = min(remaining, capacity)
        remaining -= occupation[i]
    
    return occupation


def compute_quantum_numbers(occupation):
    """
    Compute effective quantum numbers from occupation vector.
    
    Returns:
        n_eff: weighted average principal quantum number
        l_eff: weighted average angular momentum
        S_total: total spin (assuming Hund's rule for unpaired electrons)
    """
    # Map orbitals to (n, l) values
    orbital_nl = {
        '1s': (1, 0),
        '2s': (2, 0),
        '2p': (2, 1),
        '3s': (3, 0),
        '3p': (3, 1),
    }
    
    total_electrons = sum(occupation)
    if total_electrons == 0:
        return 1.0, 0.0, 0.0
    
    # Weighted average
    n_sum = 0
    l_sum = 0
    unpaired = 0
    
    for i, (orbital, occ) in enumerate(zip(ORBITAL_NAMES, occupation)):
        n, l = orbital_nl[orbital]
        n_sum += n * occ
        l_sum += l * occ
        
        # Count unpaired electrons (Hund's rule approximation)
        capacity = ORBITAL_CAPACITIES[i]
        if l == 0:  # s orbital
            unpaired += abs(occ - capacity) if occ < capacity else 0
        else:  # p orbital (simplified: assume parallel filling up to half)
            half_capacity = capacity // 2
            if occ <= half_capacity:
                unpaired += occ
            else:
                unpaired += capacity - occ
    
    n_eff = n_sum / total_electrons
    l_eff = l_sum / total_electrons
    S_total = unpaired / 2.0  # Each unpaired electron contributes 1/2
    
    return n_eff, l_eff, S_total


def generate_excited_states(ground_state, n_electrons, max_excitations=2):
    """
    Generate excited state configurations by promoting electrons.
    
    Args:
        ground_state: ground state occupation vector
        n_electrons: total number of electrons
        max_excitations: how many electrons to promote
    
    Returns:
        List of excited state occupation vectors
    """
    excited = []
    
    # Single excitations: move one electron from occupied to empty orbital
    for i in range(len(ORBITAL_NAMES)):
        if ground_state[i] > 0:  # Occupied orbital
            for j in range(i+1, len(ORBITAL_NAMES)):
                if ground_state[j] < ORBITAL_CAPACITIES[j]:  # Empty space
                    config = ground_state.copy()
                    config[i] -= 1
                    config[j] += 1
                    excited.append(config)
    
    # Add some high-spin configurations (Hund's rule violators)
    # For example, promote to create more unpaired spins
    for i in range(len(ORBITAL_NAMES)):
        if ground_state[i] == ORBITAL_CAPACITIES[i]:  # Fully filled
            for j in range(i+1, len(ORBITAL_NAMES)):
                if ground_state[j] < ORBITAL_CAPACITIES[j] - 1:
                    config = ground_state.copy()
                    config[i] -= 2  # Remove pair
                    config[j] += 2  # Add pair elsewhere
                    if config[i] >= 0:  # Valid
                        excited.append(config)
    
    # Limit number of configurations
    return excited[:15]  # Return ~15 excited states per element


def occupation_to_string(occupation):
    """Convert occupation vector to readable string like '1s2 2s2 2p4'."""
    parts = []
    for orbital, occ in zip(ORBITAL_NAMES, occupation):
        if occ > 0:
            parts.append(f"{orbital}{occ}")
    return " ".join(parts) if parts else "empty"


def generate_all_configs():
    """Generate all configurations for all elements."""
    all_configs = []
    
    for element in ELEMENTS:
        Z = element['Z']
        symbol = element['symbol']
        n_electrons = Z  # Neutral atoms
        
        print(f"\nGenerating configs for {symbol} (Z={Z}, {n_electrons} electrons)...")
        
        # Ground state
        ground = aufbau_filling(n_electrons)
        n_eff, l_eff, S = compute_quantum_numbers(ground)
        
        all_configs.append({
            'Z': Z,
            'symbol': symbol,
            'n_electrons': n_electrons,
            'occupation': ground,
            'n_eff': round(n_eff, 3),
            'l_eff': round(l_eff, 3),
            'S_total': round(S, 3),
            'config_string': occupation_to_string(ground),
            'description': 'Ground state',
            'is_ground_state': True,
        })
        print(f"  Ground: {occupation_to_string(ground)} -> (n={n_eff:.2f}, l={l_eff:.2f}, S={S:.2f})")
        
        # Excited states
        excited_states = generate_excited_states(ground, n_electrons)
        for i, excited in enumerate(excited_states):
            n_eff, l_eff, S = compute_quantum_numbers(excited)
            
            all_configs.append({
                'Z': Z,
                'symbol': symbol,
                'n_electrons': n_electrons,
                'occupation': excited,
                'n_eff': round(n_eff, 3),
                'l_eff': round(l_eff, 3),
                'S_total': round(S, 3),
                'config_string': occupation_to_string(excited),
                'description': f'Excited state {i+1}',
                'is_ground_state': False,
            })
        
        print(f"  Generated {len(excited_states)} excited states")
    
    return all_configs


def main():
    """Main execution."""
    print("=" * 60)
    print("ELECTRON CONFIGURATION GENERATOR")
    print("=" * 60)
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Generate configurations
    configs = generate_all_configs()
    
    # Save to JSON
    output_file = data_dir / 'configs.json'
    with open(output_file, 'w') as f:
        json.dump(configs, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✓ Generated {len(configs)} configurations")
    print(f"✓ Saved to {output_file}")
    print("=" * 60)
    
    # Summary statistics
    elements_count = len(set(c['Z'] for c in configs))
    ground_states = sum(1 for c in configs if c['is_ground_state'])
    excited_states = len(configs) - ground_states
    
    print(f"\nSummary:")
    print(f"  Elements: {elements_count} (H through Ne)")
    print(f"  Ground states: {ground_states}")
    print(f"  Excited states: {excited_states}")
    print(f"  Average configs/element: {len(configs)/elements_count:.1f}")
    print(f"\nNext step: python run_dft.py")


if __name__ == '__main__':
    main()

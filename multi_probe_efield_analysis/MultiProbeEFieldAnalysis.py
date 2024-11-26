import os
import re
import time
from datetime import datetime
import argparse
import warnings
import multiprocessing as mp
from itertools import product
from functools import partial
import numpy as np
from scipy.constants import epsilon_0, elementary_charge
import MDAnalysis as mda

warnings.filterwarnings("ignore", message="Found no information for attr:")
warnings.filterwarnings("ignore", message="Found missing chainIDs")

protein_residues = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                   'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
                   'TYR', 'VAL', 'HSD', 'HSE', 'HSP', 'HID', 'HIE', 'HIP', 'HIO'
                   'CYO', 'PRN', 'AS4', 'GL4'}

#Electric Field Calculation Related Functions
def debug_positions(atoms, ref_position, prefix=""):
    """Print detailed position information for atoms."""
    print(f"\n{prefix} Position Analysis:")
    print(f"Reference position: {ref_position}")
    print("\nFirst 5 residues' positions:")
    print("ResID  AtomName  AtomIndex  X        Y        Z        Dist_to_ref")
    print("-" * 70)

    seen_residues = set()
    for atom in atoms:
        if atom.residue.resid in seen_residues:
            continue
        seen_residues.add(atom.residue.resid)
        if len(seen_residues) > 10:
            break

        dist = np.linalg.norm(atom.position - ref_position)
        print(f"{atom.residue.resid:<6d} {atom.name:<8s} {atom.index:<9d} "
              f"{atom.position[0]:8.3f} {atom.position[1]:8.3f} {atom.position[2]:8.3f} "
              f"{dist:8.3f}")

def pack_around(atom_group, center, boxinfo, frame_num=None, debug=False, stats_file=None, collect_stats=False):
    """
    Properly wrap molecules around a center point considering periodic boundary conditions.
    Now includes wrapping statistics tracking.
    
    Parameters
    ----------
    atom_group : MDAnalysis AtomGroup
        Group of atoms to be wrapped
    center : numpy.ndarray
        Center point for wrapping
    boxinfo : numpy.ndarray
        Box dimensions
    frame_num : int, optional
        Current frame number for statistics tracking
    debug : bool, optional
        Whether to print debug information
    stats_file : file object, optional
        Open file handle for writing wrapping statistics
    
    Returns
    -------
    MDAnalysis AtomGroup
        Wrapped atom group
    """
    if debug:
        print("\nPack Around Debug:")
        print(f"Frame: {frame_num}")
        print(f"Box dimensions: {boxinfo[:3]}")
        print(f"Center position: {center}")
        print(f"Number of atoms to pack: {len(atom_group)}")
        print(f"Number of residues to pack: {len(atom_group.residues)}")
        debug_positions(atom_group, center, "Initial")

    # Initialize stats_string at the start
    stats_string = "0  0  0  0  0  0  0  0  0  0\n"

    # Return immediately with empty stats if atom_group is empty
    if len(atom_group) == 0:
        if collect_stats:
            return atom_group, stats_string
        return atom_group

    # Create a copy and store original positions
    tmp_group = atom_group.copy()
    original_positions = tmp_group.positions.copy()
    box = boxinfo[:3]

    # Process each residue
    residues_shifted = 0
    atoms_shifted = 0
    shifts = []
    
    for residue in tmp_group.residues:
        # Get residue center and atoms
        residue_atoms = tmp_group.select_atoms(f'resid {residue.resid} and resname {residue.resname}')
        res_center = residue_atoms.center_of_geometry()
        
        # Calculate displacement from target center
        sub = res_center - center
        
        # Check if residue needs wrapping
        needs_wrapping = False
        wrap_vector = np.zeros(3)
        
        for i in range(3):
            if abs(sub[i]) > box[i] / 2:
                needs_wrapping = True
                wrap_vector[i] = -box[i] * np.sign(sub[i])
        
        if needs_wrapping:
            if debug:
                print(f"\nResidue {residue.resname}{residue.resid} needs wrapping:")
                print(f"  Center: {res_center}")
                print(f"  Displacement: {sub}")
                print(f"  Wrap vector: {wrap_vector}")
                
            residues_shifted += 1
            pre_wrap = residue_atoms.positions.copy()
            residue_atoms.positions += wrap_vector
            post_wrap = residue_atoms.positions
            
            shift_dist = np.linalg.norm(wrap_vector)
            shifts.append((residue.resid, residue.resname, shift_dist))
            atoms_shifted += len(residue_atoms)
            
            if debug:
                for i, atom in enumerate(residue_atoms):
                    print(f"  {atom.name}: {pre_wrap[i]} -> {post_wrap[i]}")
    
    if debug:
        print(f"\nResidues shifted: {residues_shifted} of {len(tmp_group.residues)}")
        print(f"Atoms shifted: {atoms_shifted} of {len(tmp_group)}")
        if shifts:
            print("\nShifts by residue:")
            print("ResID  Resname  Shift(Å)")
            print("-" * 30)
            for resid, resname, shift in shifts:
                print(f"{resid:<6d} {resname:>8s} {shift:8.3f}")
        
        debug_positions(tmp_group, center, "Final")

    stats_string = None
    if (frame_num is not None) and (stats_file is not None or collect_stats):
        # Count residues by type
        water_shifts = sum(1 for _, resname, _ in shifts if resname in ['WAT', 'SOL'])
        ion_shifts = sum(1 for _, resname, _ in shifts if resname in ['Na+', 'Cl-'])
        other_shifts = len(shifts) - water_shifts - ion_shifts
        
        # Count total residues by type
        total_water = len([res for res in tmp_group.residues if res.resname in ['WAT', 'SOL']])
        total_ions = len([res for res in tmp_group.residues if res.resname in ['Na+', 'Cl-']])
        total_other = len(tmp_group.residues) - total_water - total_ions
        
        # Create statistics string
        stats_string = (f"{len(tmp_group.residues):9d}  "
                       f"{residues_shifted:11d}  "
                       f"{len(tmp_group):11d}  "
                       f"{atoms_shifted:12d}  "
                       f"{total_water:9d}  "
                       f"{water_shifts:12d}  "
                       f"{total_ions:7d}  "
                       f"{ion_shifts:11d}  "
                       f"{total_other:9d}  "
                       f"{other_shifts:12d}\n")
        
        if stats_file is not None:
            if frame_num == 0:
                stats_file.write("# Frame  Total_Res  Wrapped_Res  Total_Atoms  Wrapped_Atoms  "
                               "Water_Res  Wrapped_Water  Ion_Res  Wrapped_Ions  Other_Res  Wrapped_Other\n")
            stats_file.write(f"{frame_num:6d}  {stats_string}")

#   if debug:
#       print(f"\nDebug - Frame stats for cutoff {cutoff}:")
#       print("Stats:", frame_stats)

    if collect_stats:
        return tmp_group, stats_string
    return tmp_group

def calc_ElectricField(atom, refposition):
    """
    Calculate electric field contribution from a single atom.
    Returns field in MV/cm.
    """
    # Constants
    Epsilon = 8.8541878128e-12  # C²/N·m²
    k = 1/(4*np.pi*Epsilon)     # N·m²/C²

    # Convert positions to meters
    ref_pos = np.array(refposition) * 1e-10  # Å to m
    atom_pos = atom.position * 1e-10         # Å to m

    # Calculate distance vector
    rvec = ref_pos - atom_pos
    rmag = np.linalg.norm(rvec)
    rhat = rvec/rmag

    # Convert charge to Coulombs and round to 6 digits for better numerical stability
    charge = round(atom.charge, 6)
    charge_coulomb = charge * 1.60217733e-19

    # Calculate field - E = k*Q/r² in direction of rhat
    Ef = rhat * (k * charge_coulomb/rmag**2)
    # Convert to MV/cm
    Ef = Ef * 1e-8

    return Ef

def validate_packing(universe, ref_position, packed_atoms, original_atoms, cutoff, debug=False):
    """
    Validate that pack_around gives consistent results with MDAnalysis distance calculations.

    Parameters
    ----------
    universe : MDAnalysis Universe
        The molecular system
    ref_position : numpy.ndarray
        Reference position for distance calculations
    packed_atoms : MDAnalysis AtomGroup
        Atoms after pack_around
    original_atoms : MDAnalysis AtomGroup
        Original atoms before pack_around
    cutoff : float
        Distance cutoff used for selection
    debug : bool
        Whether to print detailed debugging information
    """
    print("\nValidating pack_around results:")
    print(f"Number of atoms - Original: {len(original_atoms)}, Packed: {len(packed_atoms)}")

    # Calculate distances using MDAnalysis distance calculations (PBC-aware)
    mda_distances = mda.lib.distances.distance_array(
        ref_position.reshape(1, 3),
        original_atoms.positions,
        box=universe.dimensions,
        backend='openmp'
    ).flatten()

    # Calculate distances to packed coordinates (direct, no PBC)
    packed_distances = np.linalg.norm(packed_atoms.positions - ref_position, axis=1)

    # Compare results
    mda_within_cutoff = np.sum(mda_distances <= cutoff)
    packed_within_cutoff = np.sum(packed_distances <= cutoff)

    print(f"\nAtoms within {cutoff}Å cutoff:")
    print(f"MDAnalysis distance calculation: {mda_within_cutoff}")
    print(f"Packed coordinates: {packed_within_cutoff}")

    # Check for significant differences
    if debug:
        print("\nDetailed distance comparison:")
        print("Residue     Atom    MDAnalysis_dist    Packed_dist    Difference")
        print("-" * 65)
        for i, (atom, mda_dist, packed_dist) in enumerate(zip(packed_atoms, mda_distances, packed_distances)):
            diff = abs(mda_dist - packed_dist)
            if diff > 0.01:  # Show only significant differences
                print(f"{atom.resname}{atom.resid:<4d} {atom.name:<8s} "
                      f"{mda_dist:12.3f} {packed_dist:12.3f} {diff:12.3f}")

    # Calculate and report statistics
    distance_diff = np.abs(mda_distances - packed_distances)
    max_diff = np.max(distance_diff)
    mean_diff = np.mean(distance_diff)
    std_diff = np.std(distance_diff)

    print("\nDistance difference statistics:")
    print(f"Maximum difference: {max_diff:.3f} Å")
    print(f"Mean difference: {mean_diff:.3f} Å")
    print(f"Standard deviation: {std_diff:.3f} Å")

    # Check for any major discrepancies
    significant_diff = distance_diff > 0.1  # Threshold for significant differences
    if np.any(significant_diff):
        print(f"\nFound {np.sum(significant_diff)} atoms with distance differences > 0.1 Å")
        if debug:
            print("\nAtoms with significant differences:")
            for i in np.where(significant_diff)[0]:
                atom = packed_atoms[i]
                print(f"Residue {atom.resname}{atom.resid} Atom {atom.name}: "
                      f"MDA dist = {mda_distances[i]:.3f}, "
                      f"Packed dist = {packed_distances[i]:.3f}, "
                      f"Diff = {distance_diff[i]:.3f}")

    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'n_significant_diff': np.sum(significant_diff)
    }

def calculate_field_with_solvent_and_residues(universe, ref_atom_sel, env_sel, solvent_sel=None,
                                            solvent_cutoff=None, frame_num=None, debug=False,
                                            residue_contributions=None, track_residues=False,
                                            wrapping_stats_file=None, collect_wrapping_stats=False):
    """
    Calculate electric field including proper solvent handling with optional residue tracking.
    
    Parameters
    ----------
    universe : MDAnalysis Universe
        The molecular system
    ref_atom_sel : str
        Selection string for reference atom(s)
    env_sel : str
        Selection string for environment atoms
    solvent_sel : str, optional
        Selection string for solvent
    solvent_cutoff : float, optional
        Cutoff distance for solvent selection
    frame_num : int, optional
        Current frame number for debugging
    debug : bool, optional
        Whether to print debug information
    residue_contributions : dict, optional
        Dictionary to accumulate residue contributions over time
    track_residues : bool, optional
        Whether to track per-residue contributions (False for parallel mode)
        
    Returns
    -------
    tuple
        (total_field, all_atoms, frame_contributions)
        where frame_contributions is a dict of this frame's contributions (empty if track_residues=False)
    """

    # Initialize stats at the beginning
    stats = "0  0  0  0  0  0  0  0  0  0" if collect_wrapping_stats else None

    # Initialize residue tracking if needed and requested
    if track_residues and residue_contributions is None:
        residue_contributions = {}
        # Initialize component tracking
        residue_contributions['_stats'] = {
            'protein': {'frames': 0, 'sum_field': np.zeros(3), 'sum_squared_field': np.zeros(3)},
            'solvent': {'frames': 0, 'sum_field': np.zeros(3), 'sum_squared_field': np.zeros(3)}
        }

    # Make selections
    ref_atoms = universe.select_atoms(ref_atom_sel)
    env_atoms = universe.select_atoms(env_sel)

    if debug:
        print(f"Reference atoms: {len(ref_atoms)}")
        print(f"Environment atoms: {len(env_atoms)}")

    # Get reference position
    if len(ref_atoms) > 1:
        ref_position = ref_atoms.center_of_geometry()
    else:
        ref_position = ref_atoms.positions[0]

    if debug:
        print(f"Reference position: {ref_position}")

    # Handle solvent if specified
    if solvent_sel and solvent_cutoff:
        try:
            if debug:
                original_solvent = universe.select_atoms(
                    f"same residue as (({solvent_sel}) and around {solvent_cutoff} global {ref_atom_sel})",
                    periodic=True
                )
                original_positions = original_solvent.positions.copy()

                print("\nBefore applying the solvent cutoff:")
                all_solvent = universe.select_atoms(solvent_sel)
                waters = all_solvent.select_atoms('resname WAT')
                ions = all_solvent.select_atoms('resname Na+')
                print(f"Total solvent atoms: {len(all_solvent)}")
                print(f"Water atoms: {len(waters)} ({len(waters.residues)} molecules)")
                print(f"Ion atoms: {len(ions)} ({len(ions.residues)} ions)")

                print(f"\nSelection string for solvent cutoff: ({solvent_sel}) and around {solvent_cutoff} global {ref_atom_sel}")

            solvent_atoms = universe.select_atoms(
                f"({solvent_sel}) and around {solvent_cutoff} global {ref_atom_sel}",
                periodic=True
            )
            
            if len(solvent_atoms) == 0:
                if debug:
                    print(f"No solvent atoms found within {solvent_cutoff} Å")
                # Set up proper wrapping stats even when no solvent found
                if collect_wrapping_stats:
                    stats = f"{len(env_atoms.residues):9d}  {0:11d}  {len(env_atoms):11d}  {0:12d}  {0:9d}  {0:12d}  {0:7d}  {0:11d}  {0:9d}  {0:12d}"
                all_atoms = env_atoms
            else:
                if debug:
                    print("\nAfter applying the solvent cutoff:")
                    wat = solvent_atoms.select_atoms('resname WAT')
                    ion = solvent_atoms.select_atoms('resname Na+')
                    print(f"Total solvent atoms selected: {len(solvent_atoms)}")
                    print(f"Water atoms: {len(wat)} ({len(wat.residues)} molecules)")
                    print(f"Ion atoms: {len(ion)} ({len(ion.residues)} ions)")

                # To reproduce the behavior of TUPA 
                # (J. Comp. Cem. 2022, 43 (16), 1113-1119), uncomment the next line 
                # and comment-out the "Get complete residues" block. 
                # TUPA wrapps atoms. We choose instead to wrap molecules to maintain 
                # molecular integrity.
#               complete_solvent = solvent_atoms

                # Get complete residues for all selected atoms
                if debug:
                    print("\nMaking selected residues whole.. ")
                complete_solvent = universe.atoms[[]]
                for residue in solvent_atoms.residues:
                    complete_solvent += residue.atoms

                if debug:
                    print("\nAfter completing selected residues within the solvent cutoff:")
                    wat = complete_solvent.select_atoms('resname WAT or resname SOL')
                    ion = complete_solvent.select_atoms('not (resname WAT or resname SOL)')
                    print(f"Total solvent atoms: {len(complete_solvent)}")
                    print(f"Water atoms: {len(wat)} ({len(wat.residues)} molecules)")
                    print(f"Ion atoms: {len(ion)} ({len(ion.residues)} ions)")

                    print("\nPositions after selection, before pack_around:")
                    debug_positions(complete_solvent, ref_position, "Pre-pack")

                    print("\nApplying pack_around...")
                    orig_pos = complete_solvent.positions.copy()

                if collect_wrapping_stats:
                    solvent_atoms, stats = pack_around(complete_solvent, ref_position, universe.dimensions,
                                            frame_num=frame_num, debug=debug,
                                            collect_stats=True)
#                   print(f"states: \n {stats}")
                else:
                    solvent_atoms = pack_around(complete_solvent, ref_position, universe.dimensions,
                                            frame_num=frame_num, debug=debug,
                                            stats_file=wrapping_stats_file)
                if debug:
                    max_shift = np.max(np.linalg.norm(solvent_atoms.positions - orig_pos, axis=1))
                    print(f"Maximum atom shift during pack_around: {max_shift:.2f} Å")
                    print("\nPositions after selection, after pack_around:")
                    debug_positions(solvent_atoms, ref_position, "Post-pack")        

                if debug:
                    print("Starting Validation: ")
                    validation_results = validate_packing(
                        universe,
                        ref_position,
                        solvent_atoms,
                        original_solvent,
                        solvent_cutoff,
                        debug=True 
                    )
                    print("\nPacking validation results:", validation_results)

                all_atoms = env_atoms + solvent_atoms
        except Exception as e:
            if debug:
                print(f"Error in solvent selection: {str(e)}")
            # Use only environment atoms but create proper stats
            if collect_wrapping_stats:
                stats = f"{len(env_atoms.residues):9d}  {0:11d}  {len(env_atoms):11d}  {0:12d}  {0:9d}  {0:12d}  {0:7d}  {0:11d}  {0:9d}  {0:12d}"
            all_atoms = env_atoms
    else:
        all_atoms = env_atoms
        if collect_wrapping_stats:
            stats = f"{len(env_atoms.residues):9d}  {0:11d}  {len(env_atoms):11d}  {0:12d}  {0:9d}  {0:12d}  {0:7d}  {0:11d}  {0:9d}  {0:12d}"

    # Calculate total field (and optionally track residue contributions)
    total_field = np.zeros(3)
    atoms_used = set()
    frame_contributions = {}
    
    if track_residues:
        # Initialize frame component totals
        frame_totals = {
            'protein': np.zeros(3),
            'solvent': np.zeros(3),
            'other': np.zeros(3)  # Add other to frame totals
        }

    # Process atoms
    if track_residues:
        # Process by residue for tracking
        for residue in all_atoms.residues:
            residue_field = np.zeros(3)
            residue_atoms = all_atoms.select_atoms(f'resid {residue.resid} and resname {residue.resname}')
            
            for atom in residue_atoms:
                if atom in ref_atoms or atom.index in atoms_used:
                    continue
                    
                contribution = calc_ElectricField(atom, ref_position)
                residue_field += contribution
                total_field += contribution
                atoms_used.add(atom.index)
            
            # Determine component type and update statistics
            if residue.resname in ['WAT', 'SOL', 'Na+', 'Cl-']:
                component_type = 'solvent'
            elif residue.resname in protein_residues:
                component_type = 'protein'
            else:
                component_type = 'other'
            
            # Add to frame totals (for all component types)
            frame_totals[component_type] += residue_field
            
            # Store contribution for this frame
            res_key = f"{residue.resname}-{residue.resid}"
            frame_contributions[res_key] = {
                'field': residue_field,
                'magnitude': np.linalg.norm(residue_field),
                'type': component_type
            }
            
            # Update running statistics
            if res_key not in residue_contributions:
                residue_contributions[res_key] = {
                    'sum_field': residue_field,
                    'sum_squared_field': residue_field**2,
                    'count': 1,
                    'type': component_type
                }
            else:
                residue_contributions[res_key]['sum_field'] += residue_field
                residue_contributions[res_key]['sum_squared_field'] += residue_field**2
                residue_contributions[res_key]['count'] += 1

    else:
        # Simple atom-by-atom processing for parallel mode
        for atom in all_atoms:
            if atom in ref_atoms or atom.index in atoms_used:
                continue
                
            contribution = calc_ElectricField(atom, ref_position)
            total_field += contribution
            atoms_used.add(atom.index)

    # Update component statistics if tracking residues
    if track_residues:
        for component in ['protein', 'solvent', 'other']:  
            if np.any(frame_totals[component]):
                stats = residue_contributions['_stats'][component]
                stats['frames'] += 1
                stats['sum_field'] += frame_totals[component]
                stats['sum_squared_field'] += frame_totals[component]**2

    if debug:
        print(f"\nFinal Calculation Summary:")
        print(f"Total atoms used: {len(atoms_used)}")
        field_magnitude = np.linalg.norm(total_field)
        print(f"Field magnitude: {field_magnitude:.6f}")
        print(f"Field components: [{total_field[0]:.6f}, {total_field[1]:.6f}, {total_field[2]:.6f}]")

    # Return stats string if collecting
    if collect_wrapping_stats:
        return total_field, all_atoms, frame_contributions, stats
    return total_field, all_atoms, frame_contributions, None

def write_average_residue_contributions(residue_contributions, output_file='average_residue_contributions.dat'):
    """
    Write average per-residue contributions to file, including component statistics.

    Parameters
    ----------
    residue_contributions : dict
        Dictionary of accumulated residue contributions and component statistics
    output_file : str, optional
        Output file name (default: 'average_residue_contributions.dat')
    """
    with open(output_file, 'w') as out:
        # Write header
        out.write('# Average per-residue electric field contributions\n')
        out.write('#\n')

        # Write component statistics first
        out.write('=== Component Statistics ===\n')
        out.write('#{:>12s} {:>8s} {:>16s} {:>12s} {:>16s} {:>16s} {:>16s}\n'.format(
            'Type', 'Frames', 'Avg_Magnitude', 'Std_Dev',
            'Avg_Efield_X', 'Avg_Efield_Y', 'Avg_Efield_Z'))

        stats = residue_contributions.pop('_stats')  # Remove stats for residue processing

        # Calculate totals for verification
        total_field = np.zeros(3)
        frames_counted = 0

        component_fields = {'protein': np.zeros(3), 'solvent': np.zeros(3), 'other': np.zeros(3)}
        component_counts = {'protein': 0, 'solvent': 0, 'other': 0}

        # First pass: collect statistics for each component including 'other'
        for res_name, data in residue_contributions.items():
            comp_type = data['type']
            if comp_type in component_fields:
                avg_field = data['sum_field'] / data['count']
                component_fields[comp_type] += avg_field
                component_counts[comp_type] += 1

        # Write statistics for each component
        for component in ['protein', 'solvent', 'other']:
            data = stats[component]
            frames = data['frames']
            if frames > 0:  # Only write if component was present
                avg_field = data['sum_field'] / frames
                avg_squared_field = data['sum_squared_field'] / frames
                variance = avg_squared_field - avg_field**2
                std_dev = np.sqrt(np.sum(variance))
                avg_magnitude = np.linalg.norm(avg_field)
                total_field += avg_field
                frames_counted = frames
                
                out.write('{:>13s} {:8d} {:16.6f} {:12.6f} {:16.6f} {:16.6f} {:16.6f}\n'.format(
                    component, frames, avg_magnitude, std_dev,
                    avg_field[0], avg_field[1], avg_field[2]))

        # Write total after components
        total_magnitude = np.linalg.norm(total_field)
        out.write('{:>13s} {:8d} {:16.6f} {:>12s} {:16.6f} {:16.6f} {:16.6f}\n'.format(
            'TOTAL', frames_counted, total_magnitude, '-',
            total_field[0], total_field[1], total_field[2]))

        # Write individual residue contributions
        out.write('\n=== Individual Residue Contributions ===\n')
        out.write('#{:>14s} {:>8s} {:>8s} {:>16s} {:>12s} {:>16s} {:>16s} {:>16s}\n'.format(
            'Residue', 'Type', 'Frames', 'Avg_Magnitude', 'Std_Dev',
            'Avg_Efield_X', 'Avg_Efield_Y', 'Avg_Efield_Z'))

        # Calculate and sort by average magnitude
        averages = []
        for res_name, data in residue_contributions.items():
            count = data['count']
            avg_field = data['sum_field'] / count
            avg_squared_field = data['sum_squared_field'] / count

            # Calculate standard deviation
            variance = avg_squared_field - avg_field**2
            std_dev = np.sqrt(np.sum(variance))

            avg_magnitude = np.linalg.norm(avg_field)

            averages.append((
                res_name,
                data['type'],
                count,
                avg_magnitude,
                std_dev,
                avg_field
            ))

        # Sort by magnitude
        averages.sort(key=lambda x: abs(x[3]), reverse=True)

        # Write sorted data
        for res_name, res_type, count, magnitude, std_dev, field in averages:
            out.write('{:>15s} {:>8s} {:8d} {:16.6f} {:12.6f} {:16.6f} {:16.6f} {:16.6f}\n'.format(
                res_name, res_type, count, magnitude, std_dev,
                field[0], field[1], field[2]))

        # Add back stats to the dictionary since we popped it earlier
        residue_contributions['_stats'] = stats

def process_frames_parallel(chunk_info):
    """
    Process a chunk of frames using a single Universe instance.
    Handles both single cutoff and cutoff range analysis.
    
    Parameters
    ----------
    chunk_info : tuple
        Contains (frame_nums, topology, trajectory, ref_sel, env_sel, 
                solvent_sel, solvent_cutoff, cutoff_range)
    
    Returns
    -------
    tuple
        (results, probe_positions, wrapping_stats) where results and probe_positions 
        are lists, and wrapping_stats is a dictionary keyed by cutoff values
    """
    frame_nums, topology, trajectory, ref_sel, env_sel, solvent_sel, solvent_cutoff, cutoff_range = chunk_info

    # Create one Universe per process
    universe = mda.Universe(topology, trajectory)

    # Determine if we're doing cutoff range analysis
    if cutoff_range is not None:
        cutoff_start, cutoff_stop, cutoff_step = cutoff_range
        cutoffs = np.arange(cutoff_start, cutoff_stop + cutoff_step/2, cutoff_step)
    else:
        cutoffs = [solvent_cutoff] if solvent_cutoff is not None else [None]

    # Early feedback that processor has started
    print(f"\rProcessor started on frames {frame_nums[0]}-{frame_nums[-1]}", flush=True)
    
    results = []
    probe_positions = []
    wrapping_stats = {}  # Dictionary keyed by cutoff values

    # Process each frame
    for frame_idx, frame_num in enumerate(frame_nums):
#       if frame_idx % 10 == 0:  # Report every 10 frames
#           print(f"\rProcessor working on frame {frame_num}", flush=True)

        # Load frame
        universe.trajectory[frame_num]
        
        # Store original positions
        original_positions = universe.atoms.positions.copy()

        # Get reference atoms and position
        ref_atoms = universe.select_atoms(ref_sel)
        ref_position = ref_atoms.center_of_geometry() if len(ref_atoms) > 1 else ref_atoms.positions[0]

        # Calculate field for each cutoff
        frame_results = []
        for cutoff in cutoffs:
            field, packed_atoms, frame_contribs, frame_stats = calculate_field_with_solvent_and_residues(
                universe, 
                ref_sel, 
                env_sel,
                solvent_sel, 
                cutoff,
                frame_num=frame_num,
                debug=False,
                track_residues=False,
                collect_wrapping_stats=True
            )

            magnitude = np.linalg.norm(field)
            frame_results.append((cutoff, magnitude, field))
            
            # Initialize the list for this cutoff if it doesn't exist
            if cutoff not in wrapping_stats:
                wrapping_stats[cutoff] = []
            
            # Store the wrapping stats for this cutoff and frame
            if frame_stats is not None:
                wrapping_stats[cutoff].append((frame_num, frame_stats))
        
        if len(cutoffs) > 1:
            results.append((frame_num, frame_results))
        else:
            # For single cutoff, store just the results without the cutoff value
            results.append((frame_num, magnitude, field))
            
        probe_positions.append((frame_num, ref_position))
        
        # Restore original positions for next frame
        universe.atoms.positions = original_positions

    return results, probe_positions, wrapping_stats

def analyze_trajectory_parallel(universe, ref_sel, env_sel, solvent_sel=None, solvent_cutoff=None,
                              cutoff_range=None, start=None, stop=None, step=None, n_processors=1, debug=False):
    """
    Parallel analysis of trajectory for total electric field calculation.
    Handles both single cutoff and cutoff range analysis.

    Parameters
    ----------
    universe : MDAnalysis Universe
        The molecular system
    ref_sel : str
        Selection string for reference atom(s)
    env_sel : str
        Selection string for environment atoms
    solvent_sel : str, optional
        Selection string for solvent
    solvent_cutoff : float, optional
        Single cutoff distance for solvent selection
    cutoff_range : tuple, optional
        (start, stop, step) for cutoff range analysis
    start : int, optional
        Starting frame
    stop : int, optional
        Ending frame
    step : int, optional
        Frame step size
    n_processors : int
        Number of processors to use
    debug : bool
        Whether to print debug information
    """
    print(f"\n=== Parallel Electric Field Analysis Started ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using {n_processors} processors for total field calculation")
    print(f"Analyzing frames {start}:{stop}:{step}")
    start_time = time.time()

    if cutoff_range:
        cutoff_start, cutoff_stop, cutoff_step = cutoff_range
        cutoffs = np.arange(cutoff_start, cutoff_stop + cutoff_step/2, cutoff_step)
        if debug:
            print("\nDebug - Analysis parameters:")
            print("Cutoff range:", cutoff_range)
            print("Generated cutoffs:", cutoffs)
        print(f"Analyzing cutoffs from {cutoff_start} to {cutoff_stop} Å with step {cutoff_step} Å")
    elif solvent_cutoff:
        print(f"Using single solvent cutoff: {solvent_cutoff} Å")

    # Create probe directory
    probe_dir = ensure_probe_directory(ref_sel)

    # Set up parallel processing
    frame_nums = list(range(start, stop, step if step else 1))
    total_frames = len(frame_nums)
    chunk_size = total_frames // n_processors + (1 if total_frames % n_processors else 0)
    frame_chunks = [frame_nums[i:i + chunk_size] for i in range(0, total_frames, chunk_size)]

    print(f"\nParallel processing setup:")
    print(f"Total frames to process: {total_frames}")
    print(f"Chunk size: {chunk_size}")
    print(f"Number of chunks: {len(frame_chunks)}")

    # Create chunk information
    chunk_info = [
        (chunk, universe.filename, universe.trajectory.filename,
        ref_sel, env_sel, solvent_sel, solvent_cutoff, cutoff_range)
        for chunk in frame_chunks
    ]

    # Run parallel analysis
    print(f"\nProcessing trajectory in parallel...")
    results = []
    all_probe_positions = []
    all_wrapping_stats = []

    print(f"\nInitializing {n_processors} processors...")
    print("First progress update will appear after initialization and first chunk completion")
    print("Processing frames...")

    try:
        with mp.Pool(processes=n_processors) as pool:
            for i, chunk_result in enumerate(pool.imap_unordered(process_frames_parallel, chunk_info)):
                chunk_fields, chunk_probes, chunk_stats = chunk_result
                results.extend(chunk_fields)
                all_probe_positions.extend(chunk_probes)
                all_wrapping_stats.append(chunk_stats)

                completed_frames = min((i + 1) * chunk_size, total_frames)
                if completed_frames % max(1, total_frames // 20) == 0 or completed_frames == total_frames:
                    percent_done = (completed_frames / total_frames) * 100
                    elapsed = time.time() - start_time
                    rate = completed_frames / elapsed
                    eta = (total_frames - completed_frames) / rate if rate > 0 else 0
                    print(f"\rProgress: {completed_frames}/{total_frames} frames "
                          f"({percent_done:.1f}%), Rate: {rate:.1f} frames/s, "
                          f"ETA: {eta:.1f}s", end="", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
        pool.terminate()
        pool.join()
        raise
    except Exception as e:
        print(f"\nError during parallel processing: {str(e)}")
        pool.terminate()
        pool.join()
        raise
    finally:
        if 'pool' in locals():
            pool.close()
            pool.join()

    # Sort probe positions by frame number
    all_probe_positions.sort(key=lambda x: x[0])

    # Write probe positions
    probe_file = os.path.join(probe_dir, 'probe.dat')
    with open(probe_file, 'w') as out:
        out.write('#frame    X(Å)    Y(Å)    Z(Å)\n')
        for frame_num, pos in all_probe_positions:
            out.write(f'{frame_num:<8d} {pos[0]:8.3f} {pos[1]:8.3f} {pos[2]:8.3f}\n')

    if debug:
        print("\nDebug - Processing results:")
        print(f"Total frames processed: {len(results)}")
        print(f"Total chunks with wrapping statistics: {len(all_wrapping_stats)}")
        if len(results) > 0:
            print("First result structure:", results[0])

    # Combine wrapping stats from all chunks
    combined_stats = {}
    for chunk_stats in all_wrapping_stats:
        for cutoff, stats in chunk_stats.items():
            if cutoff not in combined_stats:
                combined_stats[cutoff] = []
            combined_stats[cutoff].extend(stats)

    # Write results based on analysis type
    if cutoff_range:
        cutoff_start, cutoff_stop, cutoff_step = cutoff_range
        cutoffs = np.arange(cutoff_start, cutoff_stop + cutoff_step/2, cutoff_step)

        # Process results for each cutoff
        for cutoff in cutoffs:
            if debug:
                print(f"\nProcessing cutoff {cutoff:.1f}Å")

            # Get all results for this cutoff
            cutoff_results = []
            for frame_num, frame_results in results:
                for c, magnitude, field in frame_results:
                    if c == cutoff:
                        cutoff_results.append((frame_num, magnitude, field))

            # Sort by frame number
            cutoff_results.sort(key=lambda x: x[0])

            # Write electric field data
            cutoff_file = os.path.join(probe_dir, f'ElecField_cutoff_{cutoff:.1f}.dat')
            with open(cutoff_file, 'w') as out:
                out.write('@    title "Electric Field (Cutoff: {:.1f} Å)"\n'.format(cutoff))
                out.write('@    xaxis  label "Frame"\n')
                out.write('@    yaxis  label "MV/cm"\n')
                out.write('#frame    Magnitude          Efield_X            Efield_Y            Efield_Z\n')
                out.write('@type xy\n')

                fields = []
                for frame_num, magnitude, field in cutoff_results:
                    fields.append(field)
                    out.write(f'{frame_num:<8d} {magnitude:<18.6f} {field[0]:<18.6f} '
                             f'{field[1]:<18.6f} {field[2]:<18.6f}\n')

                # Write statistics if we have data
                if fields:
                    fields = np.array(fields)
                    magnitudes = np.linalg.norm(fields, axis=1)

                    avg_magnitude = np.mean(magnitudes)
                    avg_field = np.mean(fields, axis=0)
                    std_magnitude = np.std(magnitudes)
                    std_field = np.std(fields, axis=0)

                    out.write('#---#\n')
                    out.write(f'#AVG:     {avg_magnitude:<18.6f} {avg_field[0]:<18.6f} '
                             f'{avg_field[1]:<18.6f} {avg_field[2]:<18.6f}\n')
                    out.write(f'#STDEV:   {std_magnitude:<18.6f} {std_field[0]:<18.6f} '
                             f'{std_field[1]:<18.6f} {std_field[2]:<18.6f}\n')

            # Write wrapping statistics for this cutoff
            wrapping_file = os.path.join(probe_dir, f'wrapping_statistics_cutoff_{cutoff:.1f}.dat')
            with open(wrapping_file, 'w') as stats_file:
                stats_file.write("# Frame  Total_Res  Wrapped_Res  Total_Atoms  Wrapped_Atoms  "
                               "Water_Res  Wrapped_Water  Ion_Res  Wrapped_Ions  Other_Res  Wrapped_Other\n")
                if cutoff in combined_stats:
                    for frame_num, stat_string in sorted(combined_stats[cutoff]):
                        stats_file.write(f"{frame_num:6d}  {stat_string}")

        # Write cutoff summary file
        summary_file = os.path.join(probe_dir, 'ElecField_cutoff_summary.dat')
        with open(summary_file, 'w') as out:
            out.write('# Average electric field magnitude vs. solvent cutoff\n')
            out.write('#Cutoff(Å)  AvgMagnitude(MV/cm)  StdDev(MV/cm)\n')

            for cutoff in cutoffs:
                cutoff_results = []
                for _, frame_results in results:
                    for c, mag, _ in frame_results:
                        if c == cutoff:
                            cutoff_results.append(mag)
                if cutoff_results:
                    avg_mag = np.mean(cutoff_results)
                    std_mag = np.std(cutoff_results)
                    out.write(f'{cutoff:8.1f}  {avg_mag:18.6f}  {std_mag:12.6f}\n')

        # Write wrapping statistics summary
        wrapping_summary_file = os.path.join(probe_dir, 'wrapping_statistics_summary.dat')
        with open(wrapping_summary_file, 'w') as out:
            out.write('# Average wrapping statistics vs. solvent cutoff\n')
            out.write('#Cutoff(Å)  Avg_Total_Res  Avg_Wrapped_Res  Avg_Total_Atoms  Avg_Wrapped_Atoms  ' 
                     'Avg_Water_Res  Avg_Wrapped_Water  Avg_Ion_Res  Avg_Wrapped_Ions  Avg_Other_Res  Avg_Wrapped_Other\n')
            
            for cutoff in cutoffs:
                if cutoff in combined_stats:
                    # Extract values for each column
                    stats_values = []
                    for _, stat_string in combined_stats[cutoff]:
                        values = [float(x) for x in stat_string.split()]
                        stats_values.append(values)
                    
                    if stats_values:
                        # Calculate averages
                        avg_stats = np.mean(stats_values, axis=0)
                        out.write(f'{cutoff:8.1f}  {avg_stats[0]:13.1f}  {avg_stats[1]:15.1f}  '
                                f'{avg_stats[2]:15.1f}  {avg_stats[3]:17.1f}  {avg_stats[4]:13.1f}  '
                                f'{avg_stats[5]:17.1f}  {avg_stats[6]:11.1f}  {avg_stats[7]:15.1f}  '
                                f'{avg_stats[8]:13.1f}  {avg_stats[9]:17.1f}\n')
    else:
        # Process single cutoff results
        results.sort(key=lambda x: x[0])  # Sort by frame number

        # Write electric field data
        field_file = os.path.join(probe_dir, 'ElecField.dat')
        with open(field_file, 'w') as out:
            out.write('@    title "Electric Field"\n')
            out.write('@    xaxis  label "Frame"\n')
            out.write('@    yaxis  label "MV/cm"\n')
            out.write('#frame    Magnitude          Efield_X            Efield_Y            Efield_Z\n')
            out.write('@type xy\n')

            fields = []
            for frame_num, magnitude, field in results:
                fields.append(field)
                out.write(f'{frame_num:<8d} {magnitude:<18.6f} {field[0]:<18.6f} '
                         f'{field[1]:<18.6f} {field[2]:<18.6f}\n')

            if fields:
                fields = np.array(fields)
                magnitudes = np.linalg.norm(fields, axis=1)

                avg_magnitude = np.mean(magnitudes)
                avg_field = np.mean(fields, axis=0)
                std_magnitude = np.std(magnitudes)
                std_field = np.std(fields, axis=0)

                out.write('#---#\n')
                out.write(f'#AVG:     {avg_magnitude:<18.6f} {avg_field[0]:<18.6f} '
                         f'{avg_field[1]:<18.6f} {avg_field[2]:<18.6f}\n')
                out.write(f'#STDEV:   {std_magnitude:<18.6f} {std_field[0]:<18.6f} '
                         f'{std_field[1]:<18.6f} {std_field[2]:<18.6f}\n')

        # Write single wrapping statistics file
        if combined_stats:
            wrapping_file = os.path.join(probe_dir, 'wrapping_statistics.dat')
            with open(wrapping_file, 'w') as stats_file:
                stats_file.write("# Frame  Total_Res  Wrapped_Res  Total_Atoms  Wrapped_Atoms  "
                               "Water_Res  Wrapped_Water  Ion_Res  Wrapped_Ions  Other_Res  Wrapped_Other\n")
                cutoff = solvent_cutoff if solvent_cutoff is not None else None
                if cutoff in combined_stats:
                    for frame_num, stats in sorted(combined_stats[cutoff]):
                        stats_file.write(f"{frame_num:6d}  {stats}")

    end_time = time.time()
    print(f"\n\nParallel analysis completed in {end_time - start_time:.2f} seconds")
    print("Created files:")
    if cutoff_range:
        print(f"- Multiple ElecField_cutoff_X.X.dat files in {probe_dir}/")
        print(f"- Multiple wrapping_statistics_cutoff_X.X.dat files in {probe_dir}/")
        print(f"- {probe_dir}/ElecField_cutoff_summary.dat: Summary of cutoff analysis")
        print(f"- {probe_dir}/wrapping_statistics_summary.dat: Summary of wrapping statistics")
    else:
        print(f"- {probe_dir}/ElecField.dat: Electric field values and statistics")
        print(f"- {probe_dir}/wrapping_statistics.dat: Wrapping statistics")
    print(f"- {probe_dir}/probe.dat: Reference selection positions")

    return results, all_probe_positions

def analyze_trajectory_sequential(universe, ref_sel, env_sel, solvent_sel=None, solvent_cutoff=None,
                                start=None, stop=None, step=None, debug=False):
    """
    Sequential analysis of trajectory for per-residue contributions.
    
    Parameters
    ----------
    universe : MDAnalysis Universe
        The molecular system
    ref_sel : str
        Selection string for reference atom(s)
    env_sel : str
        Selection string for environment atoms
    solvent_sel : str, optional
        Selection string for solvent
    solvent_cutoff : float, optional
        Cutoff distance for solvent selection
    start : int, optional
        Starting frame
    stop : int, optional
        Ending frame
    step : int, optional
        Frame step size
    debug : bool
        Whether to print debug information
    """
    print("\n=== Sequential Analysis for Residue Contributions ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()

    # Create probe directory and get output file path
    probe_dir = ensure_probe_directory(ref_sel)
    output_file = os.path.join(probe_dir, 'average_residue_contributions.dat')

    # Initialize residue contributions
    residue_contributions = {
        '_stats': {
            'protein': {'frames': 0, 'sum_field': np.zeros(3), 'sum_squared_field': np.zeros(3)},
            'solvent': {'frames': 0, 'sum_field': np.zeros(3), 'sum_squared_field': np.zeros(3)},
            'other': {'frames': 0, 'sum_field': np.zeros(3), 'sum_squared_field': np.zeros(3)}
        }
    }

    # Calculate total number of frames for progress reporting
    frame_nums = range(start, stop, step if step else 1)
    total_frames = len(frame_nums)
    frames_processed = 0

    print(f"\nAnalyzing {total_frames} frames for residue contributions...")
    print(f"Frame range: {start}:{stop}:{step}")

    # Process frames sequentially
    for ts in universe.trajectory[start:stop:step]:
        # Calculate field and track residue contributions
        field, atoms, frame_contribs, frame_stats = calculate_field_with_solvent_and_residues(
            universe, ref_sel, env_sel,
            solvent_sel, solvent_cutoff,
            frame_num=ts.frame,
            debug=(debug and ts.frame == start),
            residue_contributions=residue_contributions,
            track_residues=True,
            collect_wrapping_stats=False
        )

        # Progress update
        frames_processed += 1
        if (frames_processed % max(1, total_frames//20) == 0):
            percent_done = (frames_processed / total_frames) * 100
            elapsed = time.time() - start_time
            rate = frames_processed / elapsed
            eta = (total_frames - frames_processed) / rate if rate > 0 else 0
                
            print(f"\rProgress: {frames_processed}/{total_frames} frames "
                f"({percent_done:.1f}%), Rate: {rate:.1f} frames/s, "
                f"ETA: {eta:.1f}s", end="", flush=True)

    # Process and write residue contributions
    print("\n\nProcessing contribution statistics...")
    
    # Get contribution counts
    protein_residues = sum(1 for data in residue_contributions.values() 
                         if data.get('type') == 'protein')
    solvent_residues = sum(1 for data in residue_contributions.values() 
                          if data.get('type') == 'solvent')
    other_residues = sum(1 for data in residue_contributions.values() 
                        if data.get('type') == 'other')

    print(f"\nContribution statistics:")
    print(f"Protein residues: {protein_residues}")
    print(f"Solvent residues: {solvent_residues}")
    print(f"Other residues: {other_residues}")
    print(f"Total residues: {len(residue_contributions) - 1}")  # -1 for _stats
    
    # Write average residue contributions
    write_average_residue_contributions(residue_contributions, output_file)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nSequential analysis completed in {elapsed:.2f} seconds")
    print(f"Average processing rate: {total_frames/elapsed:.1f} frames/s")
    print("Created files:")
    print(f"- {probe_dir}/average_residue_contributions.dat")
    
    return residue_contributions

#Trajecotry Processing
def track_box_dimensions(universe, output_file='box.dat', start=None, stop=None, step=None):
    """
    Track and save box dimensions for each frame.
    
    Parameters:
    -----------
    universe : MDAnalysis Universe
        The molecular system
    output_file : str
        Output file for box dimensions
    start, stop, step : int
        Frame selection parameters
    """
    with open(output_file, 'w') as out:
        out.write("# Frame   Box_X(Å)   Box_Y(Å)   Box_Z(Å)   Alpha(°)   Beta(°)   Gamma(°)   Volume(Å³)\n")
        
        for ts in universe.trajectory[start:stop:step]:
            box = ts.dimensions
            volume = box[0] * box[1] * box[2]
            out.write(f"{ts.frame:6d}  {box[0]:9.3f}  {box[1]:9.3f}  {box[2]:9.3f}  "
                     f"{box[3]:8.2f}  {box[4]:8.2f}  {box[5]:8.2f}  {volume:11.1f}\n")

#Interface Related Functions
def generate_template_config():
    """
    Generate a template configuration file with default values and documentation.
    """
    template = """# Select the topology and trajectory
# Any format recognized by MDAnalysis
# is acceptable. The trajectory should
# be re-imaged.
topology       = OmcE_r659.parm7
trajectory     = imaged.xtc

# Select the type of analysis
# - total         : Compute total electric field vector.
#                   Calculate is parallelized
# - contributions : Compute per-residue and protein/solvent
#                   contributions to the electric field.
#                   Calculation is NOT parallelized.
# - all           : The parallelized total electric field
#                   calculation is performed first, and then
#                   the single-processor contribution analysis
#                   is performed.
analysis       = total

# Select which frame(s) or all to analyze
frames         = all

# Select one or more probe atoms. Add a
# separate probe entry for each probe atom
# NOTE: For the XXX replacement in environment
# to work, probe selections MUST include a
# 'resid' specification (e.g., resid 653)
probe          = (resname HEH and resid 653 and name FE)

# Select a distance dependent non-solvent environment (1st command)
# or all non-solvent residues excluding the probe atom (2nd command).
# Either way, XXX is a placeholder that will be replaced by the probe residue
# ID by the program automatically. This feature allows multiple probe atoms
# for separate calculations to be specified in the same input file, while
# ensuring that the probe each time is always excluded from the environment
# definition.
#environment   = same residue as ((not (resname WAT) and not (resname Na+) and not resid XXX) and around 20.0 resid XXX)
environment    = (not resname WAT and not resname Na+ and not resname Cl-) and not (resname HEH and resid XXX)

# Select water and solvent ions
solvent        = resname WAT or resname Na+

# Define a cutoff for including solvent molecules
# or a range of cutoffs (START:STOP:STEP Å) that
# will each be assessed.
solvent_cutoff = 10
#cutoff_range  = 1:30:2

# Select the number of processors for the calculation
processors     = 10
"""
    with open('temp.config', 'w') as f:
        f.write(template)
    print("Template configuration file 'temp.config' has been generated.")

def parse_arguments():
    """
    Parse command line arguments for frame selection and atom selections.
    """
    parser = argparse.ArgumentParser(description='Calculate electric field from MD trajectory.')

    parser.add_argument('--settings', type=str,
                       help='Text file containing analysis settings (overrides other arguments)')

    parser.add_argument('--frames', type=str, default='all',
                       help='Frame selection. Can be "start:end", "start:end:step", or "all"')

    parser.add_argument('--topology', type=str,
                       help='Topology file (e.g., parm7, pdb)')

    parser.add_argument('--trajectory', type=str,
                       help='Trajectory file (e.g., nc, xtc)')

    parser.add_argument('--probe', type=str,
                       help='Selection of atom(s) at which electric field is calculated')

    parser.add_argument('--environment', type=str,
                       help='Non-solvent atoms that contribute to the electric field (e.g., protein, cofactors)')

    parser.add_argument('--solvent', type=str,
                       help='Mobile molecules (water, ions) that contribute to the field')

    parser.add_argument('--solvent-cutoff', type=float,
                       help='Cutoff distance for solvent selection in Angstroms')

    parser.add_argument('--cutoff-range', type=str,
                       help='Range of solvent cutoffs to analyze (start:stop:step in Angstroms)')
    
    parser.add_argument('--processors', type=int, default=1,
                       help='Number of processors for parallel calculation')

    parser.add_argument('--analysis', type=str, default='all', 
                       choices=['total', 'contributions', 'all'],
                       help='Analysis type: total (parallel field analysis), '
                            'contributions (sequential residue analysis), or all (both)')

    parser.add_argument('--generate-template', action='store_true',
                       help='Generate a template configuration file and exit')

    return parser

def read_settings(filename):
    """
    Read analysis settings from a text file.
    Handles multiple probe lines and parameter settings.

    Valid settings include:
    - probe: Selection of atom(s) at which electric field is calculated (multiple allowed)
    - topology: Topology file path
    - trajectory: Trajectory file path
    - environment: Non-solvent atoms that contribute to the electric field
    - solvent: Mobile molecules that contribute to the field
    - solvent_cutoff: Single cutoff distance for solvent selection
    - cutoff_range: Range of cutoffs to analyze (format: start:stop:step)
    - frames: Frame selection ("start:end", "start:end:step", or "all")
    - processors: Number of processors for parallel calculation
    - analysis: Analysis type ("total", "contributions", or "all")

    Parameters:
    -----------
    filename : str
        Path to settings file

    Returns:
    --------
    dict
        Dictionary of settings with 'probes' as a list of probe selections
    """
    settings = {}
    probe_selections = []
    
    with open(filename, 'r') as f:
        for line in f:
            # Skip empty lines and comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                key, value = [part.strip() for part in line.split('=', 1)]
            except ValueError:
                print(f"Warning: Skipping invalid line format: {line}")
                continue

            # Handle multiple probe lines
            if key == 'probe':
                probe_selections.append(value)
                continue
                
            # Handle cutoff specifications
            if key == 'cutoff_range':
                try:
                    start, stop, step = map(float, value.split(':'))
                    settings['cutoff_range'] = (start, stop, step)
                except ValueError:
                    print(f"Warning: Invalid cutoff range format: {value}. Expected start:stop:step")
                    continue
                continue
                
            if key == 'solvent_cutoff':
                try:
                    settings[key] = float(value)
                except ValueError:
                    print(f"Warning: Invalid solvent cutoff value: {value}. Expected float")
                    continue
                continue

            # Handle numeric values
            if key in ['processors', 'step']:
                try:
                    settings[key] = int(value)
                    continue
                except ValueError:
                    pass  # Fall through to general case if not a valid integer

            # Handle boolean values
            if value.lower() in ['true', 'yes', 'on']:
                settings[key] = True
            elif value.lower() in ['false', 'no', 'off']:
                settings[key] = False
            else:
                settings[key] = value

    # Validate settings
    if not probe_selections:
        print("Warning: No probe selections found in settings file")
    
    # Check for conflicting cutoff specifications
    if 'solvent_cutoff' in settings and 'cutoff_range' in settings:
        print("Warning: Both solvent_cutoff and cutoff_range specified. Using cutoff_range for analysis.")
        
    # Store probe selections in settings
    settings['probes'] = probe_selections

    # Print summary of loaded settings
    print("\nLoaded settings:")
    for key, value in settings.items():
        if key != 'probes':  # Handle probes separately
            print(f"{key}: {value}")
    print(f"Number of probe selections: {len(probe_selections)}")

    return settings

def get_probe_directory(probe_sel):
    """
    Generate directory name for probe based on its selection string.
    
    Parameters:
    -----------
    probe_sel : str
        Probe selection string
    
    Returns:
    --------
    str
        Directory name in format FE-HEH{X}
    """
    # Extract resid from probe selection
    resid_match = re.search(r'resid\s+(\d+)', probe_sel)
    if not resid_match:
        raise ValueError(f"Could not extract resid from probe selection: {probe_sel}")
    
    resid = resid_match.group(1)
    return f"FE-HEH{resid}"

def ensure_probe_directory(probe_sel):
    """
    Create directory for probe if it doesn't exist.
    
    Parameters:
    -----------
    probe_sel : str
        Probe selection string
    
    Returns:
    --------
    str
        Path to probe directory
    """
    dir_name = get_probe_directory(probe_sel)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def parse_frame_selection(frame_str, total_frames):
    """
    Parse frame selection string into start, stop, step values.

    Parameters:
    -----------
    frame_str : str or int
        Frame selection. Can be "start:end", "start:end:step", "all", or a single frame number
    total_frames : int
        Total number of frames in trajectory

    Returns:
    --------
    tuple
        (start, stop, step) values for frame selection
    """
    # Convert to string if an integer was passed
    frame_str = str(frame_str)

    if frame_str.lower() == 'all':
        return 0, total_frames, 1

    if ':' in frame_str:
        parts = frame_str.split(':')
        if len(parts) == 2:
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1]) if parts[1] else total_frames
            return start, stop, 1
        elif len(parts) == 3:
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1]) if parts[1] else total_frames
            step = int(parts[2]) if parts[2] else 1
            return start, stop, step
    else:
        try:
            frame = int(frame_str)
            return frame, frame + 1, 1
        except ValueError:
            raise ValueError(f"Invalid frame selection: {frame_str}")

def get_residue_from_selection(selection_string):
    """
    Extract residue number from a selection string.
    
    Parameters:
    -----------
    selection_string : str
        MDAnalysis selection string containing 'resid X'
    
    Returns:
    --------
    int
        Residue number
    """
    match = re.search(r'resid\s+(\d+)', selection_string)
    if match:
        return int(match.group(1))
    raise ValueError("Could not extract residue number from selection string")

def modify_environment_selection(env_sel, probe_sel):
    """
    Modify environment selection string by replacing 'XXX' (case-insensitive) 
    with the probe residue ID.

    Parameters
    ----------
    env_sel : str
        Environment selection string with 'XXX' placeholder
    probe_sel : str
        Probe selection string containing residue ID

    Returns
    -------
    str
        Modified environment selection with correct residue ID
    """
    # Extract residue from probe selection
    resid_match = re.search(r'resid\s+(\d+)', probe_sel)
    if not resid_match:
        print(f"Warning: No residue ID found in probe selection: {probe_sel}")
        return env_sel
    
    resid = resid_match.group(1)
    
    # Check if XXX exists in environment selection (case-insensitive)
    if not re.search(r'xxx', env_sel, re.IGNORECASE):
        print(f"Warning: No 'XXX' placeholder found in environment selection: {env_sel}")
        print("The environment selection will be used without modification.")
        return env_sel
        
    # Replace XXX with the actual residue ID (case-insensitive)
    modified_env_sel = re.sub(r'xxx', resid, env_sel, flags=re.IGNORECASE)
    return modified_env_sel

def analyze_selection(universe, selection, name="Selection"):
    """
    Analyze composition of an atom selection.
    """
    atoms = universe.select_atoms(selection)
    if len(atoms) == 0:
        print(f"\n{name} is empty! Selection string: '{selection}'")
        return

    print(f"\n=== {name} Analysis ===")
    print(f"Selection string: '{selection}'")
    print(f"Total atoms: {len(atoms)}")

    # Analyze residues
    residues = atoms.residues
    print(f"Total residues: {len(residues)}")

    # Count residue types and list in numerical order
#   res_counts = {}
#   for res in residues:
#       res_counts[res.resname] = res_counts.get(res.resname, 0) + 1
#   print("\nResidue composition:")
#   for resname, count in sorted(res_counts.items(), key=lambda x: x[1], reverse=True):
#       print(f"  {resname}: {count}")

    res_counts = {}
    for res in residues:
        res_counts[res.resname] = res_counts.get(res.resname, 0) + 1
    print("\nResidue composition:")
    for resname, count in sorted(res_counts.items()):
        print(f"  {resname}: {count}")

    # Calculate total charge
    total_charge = np.sum(atoms.charges)
    print(f"\nTotal charge: {total_charge:.3f}e")
    print("=" * (len(name) + 11))

def create_banner(lines, width=87, border_char='#'):
    """
    Create a formatted banner with centered text and borders.

    Parameters
    ----------
    lines : list of str
        Lines of text to be centered in the banner
    width : int
        Total width of the banner including borders
    border_char : str
        Character to use for the border

    Returns
    -------
    str
        Formatted banner string
    """
    # Create the top/bottom border
    border = border_char * width

    # Create the formatted lines
    formatted_lines = []
    for line in lines:
        # Calculate padding for centering
        padding = width - 2  # -2 for the border characters
        formatted_line = f"{border_char}{line.center(padding)}{border_char}"
        formatted_lines.append(formatted_line)

    # Combine all parts
    return f"\n{border}\n" + \
           "\n".join(formatted_lines) + \
           f"\n{border}\n"

def print_program_banner():
    """
    Print the program banner with current date/time.
    """
    banner_lines = [
        "Electric Field Analysis at Probe Atoms",
        "Version 1",
        "Written by Matthew J. Guberman-Pfeffer",
        "Last Modified: 11/17/2024",
        "",
        f"Started on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ]

    print(create_banner(banner_lines))

def main():
    print_program_banner()

    parser = parse_arguments()
    args = parser.parse_args()

    # Check if user requested template generation
    if args.generate_template:
        generate_template_config()
        return

    # Initialize settings for probe selection
    probes = []
    
    # If settings file provided, read settings
    if args.settings:
        print(f"\nReading settings from {args.settings}")
        settings = read_settings(args.settings)
        
        # Handle multiple probes from settings
        if 'probes' not in settings or not settings['probes']:
            parser.error("No probe selections found in settings file")
        probes = settings['probes']
        
        # Override arguments with all other settings
        args.topology = settings.get('topology', args.topology)
        args.trajectory = settings.get('trajectory', args.trajectory)
        args.environment = settings.get('environment', args.environment)
        args.solvent = settings.get('solvent', args.solvent)
        if 'solvent_cutoff' in settings:
            args.solvent_cutoff = float(settings['solvent_cutoff'])
        if 'cutoff_range' in settings:  # Add this block
            args.cutoff_range = settings['cutoff_range']  # Already a tuple from read_settings
        args.frames = settings.get('frames', args.frames)
        if 'processors' in settings:
            args.processors = int(settings['processors'])
        args.analysis = settings.get('analysis', args.analysis)
    elif args.probe:
        # If no settings file but probe argument provided, use single probe
        probes = [args.probe]
    else:
        parser.error("No probe selection provided. Use either --settings file or --probe argument")

    # Check required arguments
    required = ['topology', 'trajectory', 'environment']
    missing = [arg for arg in required if not getattr(args, arg)]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")

    # Load universe
    print(f"\nLoading system...")
    print(f"Topology: {args.topology}")
    print(f"Trajectory: {args.trajectory}")
    try:
        u = mda.Universe(args.topology, args.trajectory)
        print(f"System contains {len(u.atoms)} atoms")
    except Exception as e:
        print(f"Error loading system: {str(e)}")
        return

    # Analysis setup review
    print("\nAnalysis plan:")
    if args.analysis == 'all':
        print("Running both total field and contribution analyses")
    elif args.analysis == 'total':
        print("Running parallel total field analysis")
    else:  # contributions
        print("Running sequential residue contribution analysis")

    # Print all probe selections
    print("\nProbe selections to process:")
    for i, probe in enumerate(probes, 1):
        print(f"{i}. {probe}")
        try:
            probe_atoms = u.select_atoms(probe)
            if len(probe_atoms) == 0:
                print(f"Warning: Probe selection {i} returned no atoms!")
        except Exception as e:
            print(f"Error with probe selection {i}: {str(e)}")
            return

    # Store original environment selection for reuse with each probe
    original_env_sel = args.environment

    # Parse frame selection
    try:
        start, stop, step = parse_frame_selection(args.frames, len(u.trajectory))
        print(f"\nAnalyzing frames {start} to {stop} with step {step}")
    except Exception as e:
        print(f"Error parsing frame selection: {str(e)}")
        return
    
#   # Track box dimensions in main directory
#   try:
#       track_box_dimensions(u, 'box.dat', start=start, stop=stop, step=step)
#   except Exception as e:
#       print(f"Warning: Error tracking box dimensions: {str(e)}")

    # Process each probe
    for probe_idx, probe_sel in enumerate(probes, 1):
        print(f"\n=== Processing probe {probe_idx}/{len(probes)}: {probe_sel} ===")

        # Modify environment selection for this probe
        args.environment = modify_environment_selection(original_env_sel, probe_sel)
        print(f"Modified environment selection: {args.environment}")

        # Create and verify probe directory
        try:
            probe_dir = ensure_probe_directory(probe_sel)
            print(f"Results will be stored in: {probe_dir}")
        except Exception as e:
            print(f"Error creating directory for probe {probe_idx}: {str(e)}")
            continue

        # Analyze selections for current probe
        print("\nAnalysis Groups:")
        try:
            analyze_selection(u, probe_sel, f"Probe {probe_idx}")
            analyze_selection(u, args.environment, f"Environment (excluding probe {probe_idx})")
            if args.solvent:
                analyze_selection(u, args.solvent, "Solvent")
        except Exception as e:
            print(f"Error analyzing selections for probe {probe_idx}: {str(e)}")
            continue

        # Run parallel analysis for total field calculation
        if args.analysis in ['all', 'total']:
            print(f"\nRunning parallel analysis for total electric field...")
            try:
                analyze_trajectory_parallel(
                    u,
                    probe_sel,
                    args.environment,
                    args.solvent,
                    args.solvent_cutoff,
                    cutoff_range=args.cutoff_range,
                    start=start,
                    stop=stop,
                    step=step,
                    n_processors=args.processors,
                    debug=False 
                )
            except Exception as e:
                print(f"Error in parallel analysis for probe {probe_idx}: {str(e)}")
           
        # Run sequential analysis for residue contributions
        if args.analysis in ['all', 'contributions']:
            print(f"\nRunning sequential analysis for residue contributions...")
            try:
                analyze_trajectory_sequential(
                    u,
                    probe_sel,
                    args.environment,
                    args.solvent,
                    args.solvent_cutoff,
                    start=start,
                    stop=stop,
                    step=step,
                    debug=False
                )
            except Exception as e:
                print(f"Error in sequential analysis for probe {probe_idx}: {str(e)}")

    # Final summary
    print("\nAnalysis complete!")
    print("\nPer-probe directories created:")
    for probe_sel in probes:
        dir_name = get_probe_directory(probe_sel)
        print(f"\n{dir_name}/")
        if args.analysis in ['all', 'total']:
            if args.cutoff_range:
                print(f"  - Multiple ElecField_cutoff_X.X.dat files")
                print(f"  - Multiple wrapping_statistics_cutoff_X.X.dat files")
                print(f"  - ElecField_cutoff_summary.dat: Summary of cutoff analysis")
            else:
                print("  - ElecField.dat: Electric field values and statistics")
                print("  - wrapping_statistics.dat: Wrapping statistics")
            print("  - probe.dat: Reference selection positions")
        if args.analysis in ['all', 'contributions']:
            print("  - average_residue_contributions.dat: Per-residue and component statistics")

if __name__ == "__main__":
    main()


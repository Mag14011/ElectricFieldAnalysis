import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
from pathlib import Path
import argparse
import sys

def setup_parser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(description='Analyze field-dependent reorganization energy for multiple proteins')
    
    # Add arguments for each protein
    parser.add_argument('-OmcE', nargs='+', type=int, help='Residue IDs for OmcE protein')
    parser.add_argument('-OmcS', nargs='+', type=int, help='Residue IDs for OmcS protein')
    parser.add_argument('-OmcZ', nargs='+', type=int, help='Residue IDs for OmcZ protein')
    
    return parser

def get_pairs(residues):
    """Generate pairs of adjacent residues from the list"""
    return list(zip(residues[:-1], residues[1:]))

def read_data_file(donor, acceptor, protein):
    """Read and parse a single data file, including protein information"""
    filepath = f"{protein}/r{donor}/{donor}_{acceptor}_md_nofilter/txt/{donor}_{acceptor}_EFieldDependentReorgEng_no_filter.txt"
    print(f"Reading file: {filepath}")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    data_line = None
    for i, line in enumerate(lines):
        if line.strip() == "# Data:":
            data_line = lines[i+2]
            break
    
    if data_line is None:
        raise ValueError(f"Could not find data in file {filepath}")
        
    values = [float(x) for x in data_line.strip().split()]
    
    # Store both original and absolute values, plus protein info
    correlations = {
        'Reactant Donor': (values[0], values[1], values[12], np.sqrt(values[13]), abs(values[12]), protein),
        'Reactant Acceptor': (values[2], values[3], values[14], np.sqrt(values[15]), abs(values[14]), protein),
        'Product Donor': (values[4], values[5], values[20], np.sqrt(values[21]), abs(values[20]), protein),
        'Product Acceptor': (values[6], values[7], values[22], np.sqrt(values[23]), abs(values[22]), protein)
    }
    
    return correlations

def analyze_correlations(all_data, use_absolute=False):
    """Calculate correlation statistics with protein-specific information"""
    energy_idx = 4 if use_absolute else 2
    
    categories = ['Reactant Donor', 'Reactant Acceptor', 'Product Donor', 'Product Acceptor']
    results = {}
    
    # For grouped analysis
    grouped_data = {
        'All Donors': {'fields': [], 'field_errs': [], 'energies': [], 'energy_errs': [], 'proteins': []},
        'All Acceptors': {'fields': [], 'field_errs': [], 'energies': [], 'energy_errs': [], 'proteins': []}
    }
    
    # Process individual categories
    for category in categories:
        fields = [data[category][0] for data in all_data]
        field_errs = [data[category][1] for data in all_data]
        energies = [data[category][energy_idx] for data in all_data]
        energy_errs = [data[category][3] for data in all_data]
        proteins = [data[category][5] for data in all_data]
        
        r, p = stats.pearsonr(fields, energies)
        slope, intercept = np.polyfit(fields, energies, 1)
        
        results[category] = {
            'fields': fields,
            'field_errs': field_errs,
            'energies': energies,
            'energy_errs': energy_errs,
            'proteins': proteins,
            'correlation': r,
            'p_value': p,
            'slope': slope,
            'intercept': intercept
        }
        
        # Add to grouped data
        if 'Donor' in category:
            grouped_data['All Donors']['fields'].extend(fields)
            grouped_data['All Donors']['field_errs'].extend(field_errs)
            grouped_data['All Donors']['energies'].extend(energies)
            grouped_data['All Donors']['energy_errs'].extend(energy_errs)
            grouped_data['All Donors']['proteins'].extend(proteins)
        else:
            grouped_data['All Acceptors']['fields'].extend(fields)
            grouped_data['All Acceptors']['field_errs'].extend(field_errs)
            grouped_data['All Acceptors']['energies'].extend(energies)
            grouped_data['All Acceptors']['energy_errs'].extend(energy_errs)
            grouped_data['All Acceptors']['proteins'].extend(proteins)
    
    # Add grouped results if not using absolute values
    if not use_absolute:
        for group_name, data in grouped_data.items():
            r, p = stats.pearsonr(data['fields'], data['energies'])
            slope, intercept = np.polyfit(data['fields'], data['energies'], 1)
            results[group_name] = {
                'fields': data['fields'],
                'field_errs': data['field_errs'],
                'energies': data['energies'],
                'energy_errs': data['energy_errs'],
                'proteins': data['proteins'],
                'correlation': r,
                'p_value': p,
                'slope': slope,
                'intercept': intercept
            }
    
    # Calculate overall correlation if using absolute values
    if use_absolute:
        all_fields = []
        all_energies = []
        all_field_errs = []
        all_energy_errs = []
        all_proteins = []
        for category in categories:
            all_fields.extend(results[category]['fields'])
            all_energies.extend(results[category]['energies'])
            all_field_errs.extend(results[category]['field_errs'])
            all_energy_errs.extend(results[category]['energy_errs'])
            all_proteins.extend(results[category]['proteins'])
        
        overall_r, overall_p = stats.pearsonr(all_fields, all_energies)
        slope, intercept = np.polyfit(all_fields, all_energies, 1)
        results['Overall'] = {
            'fields': all_fields,
            'field_errs': all_field_errs,
            'energies': all_energies,
            'energy_errs': all_energy_errs,
            'proteins': all_proteins,
            'correlation': overall_r,
            'p_value': overall_p,
            'slope': slope,
            'intercept': intercept
        }
    
    return results

def create_plot(results, ylabel, filename, width=7):
    """Create plot with protein-specific colors and donor/acceptor filling"""
    plt.figure(figsize=(width, width * 1.000))

    # Colors for each protein
    protein_colors = {
        'OmcE': '#ff7f0e',  # orange
        'OmcS': '#2ca02c',  # green
        'OmcZ': '#9467bd'   # purple
    }

    # Abbreviations for proteins
    protein_abbrev = {
        'OmcE': 'E',
        'OmcS': 'S',
        'OmcZ': 'Z'
    }

    # Abbreviations for categories
    category_abbrev = {
        'Reactant Donor': 'R.D.',
        'Product Donor': 'P.D.',
        'Reactant Acceptor': 'R.A.',
        'Product Acceptor': 'P.A.',
        'All Donors': 'Donors',
        'All Acceptors': 'Acceptors',
        'Overall': 'Overall'
    }

    # Markers for different categories
    markers = {
        'Reactant Donor': 'o',
        'Product Donor': 's',
        'Reactant Acceptor': 'o',
        'Product Acceptor': 's',
        'All Donors': '^',
        'All Acceptors': '^',
        'Overall': 'o'
    }

    # Calculate protein-specific correlations for each category
    protein_correlations = {}
    for category in results.keys():
        data = results[category]
        protein_correlations[category] = {}

        for protein in protein_colors:
            protein_idx = [i for i, p in enumerate(data['proteins']) if p == protein]
            if protein_idx:
                fields = [data['fields'][i] for i in protein_idx]
                energies = [data['energies'][i] for i in protein_idx]
                if len(fields) > 1:  # Need at least 2 points for correlation
                    r, _ = stats.pearsonr(fields, energies)
                    protein_correlations[category][protein] = r

    # First, draw error bands
    for category in results.keys():
        data = results[category]
        sort_idx = np.argsort(data['fields'])
        x = np.array(data['fields'])[sort_idx]
        y = np.array(data['energies'])[sort_idx]
        xerr = np.array(data['field_errs'])[sort_idx]
        yerr = np.array(data['energy_errs'])[sort_idx]

        for protein in protein_colors:
            protein_idx = [i for i, p in enumerate(data['proteins']) if p == protein]
            if protein_idx:
                protein_x = x[protein_idx]
                protein_y = y[protein_idx]
                protein_xerr = xerr[protein_idx]
                protein_yerr = yerr[protein_idx]

                plt.fill_between(protein_x, protein_y - protein_yerr, protein_y + protein_yerr,
                               color=protein_colors[protein], alpha=0.1)
                plt.fill_betweenx(protein_y, protein_x - protein_xerr, protein_x + protein_xerr,
                                color=protein_colors[protein], alpha=0.1)

    # Then draw regression lines
    for category in results.keys():
        data = results[category]
        x_range = np.array([min(data['fields']), max(data['fields'])])
        y_range = data['slope'] * x_range + data['intercept']
        plt.plot(x_range, y_range, '--', color='gray', alpha=0.8, zorder=2)

    # Add overall correlation first if it's the overall plot
    if any('Overall' in key for key in results.keys()):
        data = list(results.values())[0]  # Get the overall data
        plt.plot([], [], ' ', label=f"r={data['correlation']:.3f}")

    # Finally plot points
    for category in results.keys():
        data = results[category]

        for protein in protein_colors:
            protein_idx = [i for i, p in enumerate(data['proteins']) if p == protein]
            if protein_idx:
                x = np.array(data['fields'])[protein_idx]
                y = np.array(data['energies'])[protein_idx]

                # Determine if this is a donor or acceptor category
                is_donor = 'Donor' in category

                # Create label
                if category == 'Overall':
                    label = f"{protein_abbrev[protein]}"
                else:
                    if protein in protein_correlations[category]:
                        r = protein_correlations[category][protein]
                        label = f"{protein_abbrev[protein]}-{category_abbrev[category]} (r={r:.3f})"
                    else:
                        label = f"{protein_abbrev[protein]}-{category_abbrev[category]}"

                plt.scatter(x, y,
                          color=protein_colors[protein],
                          marker=markers[category],
                          s=64,
                          facecolors=protein_colors[protein] if is_donor else 'none',
                          edgecolors=protein_colors[protein],
                          label=label,
                          alpha=0.6,
                          zorder=3)

    plt.xlabel('Electric Field Magnitude (V/Å)')
    plt.ylabel(ylabel)
    plt.tick_params(direction='in', which='both', top=True, right=True)

    if 'Overall' not in results:  # Only for individual categories plot
        plt.legend(loc='upper left', ncol=2, framealpha=0.9)
    else:
        plt.legend(loc='upper left', ncol=1, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def print_data_summary(all_data, proteins_processed):
    """Print summary of data being analyzed"""
    print("\nData Summary:")
    print("=" * 50)

    # First, organize data by protein
    protein_data = {protein: [] for protein in proteins_processed}

    # Organize data by protein
    for data in all_data:
        # All categories in a data point have the same protein, so just check the first one
        protein = list(data.values())[0][5]
        protein_data[protein].append(data)

    # Print summary for each protein
    for protein in proteins_processed:
        print(f"\n{protein} Data:")
        print("-" * 20)
        protein_entries = protein_data[protein]
        print(f"Number of pairs: {len(protein_entries)}")

        print("\nPairs processed:")
        for i, data in enumerate(protein_entries, 1):
            print(f"\n  Pair {i}:")
            for category in ['Reactant Donor', 'Reactant Acceptor', 'Product Donor', 'Product Acceptor']:
                field, field_err, energy, energy_err, abs_energy, _ = data[category]
                print(f"    {category}:")
                print(f"      Field: {field:.3f} ± {field_err:.3f} V/Å")
                print(f"      Energy: {energy:.3f} ± {energy_err:.3f} eV")
                print(f"      |Energy|: {abs_energy:.3f} eV")
            print("    " + "-" * 30)  # Separator between pairs

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    # Check if at least one protein was specified
    if not any([args.OmcE, args.OmcS, args.OmcZ]):
        print("Error: At least one protein must be specified")
        parser.print_help()
        sys.exit(1)
    
    # Create output directory
    output_dir = "multi_protein_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each protein's data
    all_data = []
    proteins_processed = []
    
    for protein_name, residues in [('OmcE', args.OmcE), ('OmcS', args.OmcS), ('OmcZ', args.OmcZ)]:
        if residues is None:
            continue
            
        proteins_processed.append(protein_name)
        pairs = get_pairs(residues)
        print(f"\nProcessing {protein_name} pairs: {pairs}")
        
        for donor, acceptor in pairs:
            try:
                data = read_data_file(donor, acceptor, protein_name)
                all_data.append(data)
                print(f"Successfully processed {protein_name} pair {donor}-{acceptor}")
            except FileNotFoundError:
                print(f"Warning: Could not find file for {protein_name} pair {donor}-{acceptor}")
            except Exception as e:
                print(f"Error processing {protein_name} pair {donor}-{acceptor}: {str(e)}")

    if not all_data:
        print("No data was successfully collected. Check file paths and formats.")
        return

    # After collecting data but before analysis
    if all_data:
        print_data_summary(all_data, proteins_processed)

        # Also print the command used to generate this data
        print("\nCommand to reproduce this analysis:")
        cmd = "python script.py"
        if args.OmcE:
            cmd += f" -OmcE {' '.join(map(str, args.OmcE))}"
        if args.OmcS:
            cmd += f" -OmcS {' '.join(map(str, args.OmcS))}"
        if args.OmcZ:
            cmd += f" -OmcZ {' '.join(map(str, args.OmcZ))}"
        print(cmd)
        print("\n" + "=" * 50)  # Separator before analysis

    # Create three different sets of results
    individual_results = {k: v for k, v in analyze_correlations(all_data, use_absolute=False).items() 
                        if k in ['Reactant Donor', 'Reactant Acceptor', 'Product Donor', 'Product Acceptor']}
    
    grouped_results = {k: v for k, v in analyze_correlations(all_data, use_absolute=False).items() 
                      if k in ['All Donors', 'All Acceptors']}
    
    overall_results = analyze_correlations(all_data, use_absolute=True)
    
    # Create plots
    for width in [7]:
        create_plot(individual_results, 
                   'Polarization Energy (eV)',
                   f'{output_dir}/individual_correlations_{width}in.png',
                   width)
        
        create_plot(grouped_results,
                   'Polarization Energy (eV)',
                   f'{output_dir}/grouped_correlations_{width}in.png',
                   width)
        
        create_plot({'Overall': overall_results['Overall']},
                   '|Polarization Energy| (eV)',
                   f'{output_dir}/overall_correlation_{width}in.png',
                   width)
    
    # Save statistics
    with open(f"{output_dir}/correlation_statistics.txt", 'w') as f:
        f.write("Multi-Protein Correlation Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"\nProteins analyzed: {', '.join(proteins_processed)}\n\n")
        
        f.write("\nIndividual Categories (Original Values):\n")
        for category, data in individual_results.items():
            f.write(f"\n{category}:\n")
            f.write(f"Pearson correlation coefficient (r): {data['correlation']:.3f}\n")
            f.write(f"Coefficient of determination (r²): {data['correlation']**2:.3f}\n")
            f.write(f"P-value: {data['p_value']:.3e}\n")
            f.write(f"Regression equation: y = {data['slope']:.3f}x + {data['intercept']:.3f}\n")
            f.write(f"Number of points: {len(data['fields'])}\n")
            f.write(f"Mean field: {np.mean(data['fields']):.3f} ± {np.mean(data['field_errs']):.3f} V/Å\n")
            f.write(f"Mean energy: {np.mean(data['energies']):.3f} ± {np.mean(data['energy_errs']):.3f} eV\n")
            
            # Add protein-specific statistics
            for protein in ['OmcE', 'OmcS', 'OmcZ']:
                protein_idx = [i for i, p in enumerate(data['proteins']) if p == protein]
                if protein_idx:
                    fields = [data['fields'][i] for i in protein_idx]
                    energies = [data['energies'][i] for i in protein_idx]
                    fields = [data['fields'][i] for i in protein_idx]
                    energies = [data['energies'][i] for i in protein_idx]
                    field_errs = [data['field_errs'][i] for i in protein_idx]
                    energy_errs = [data['energy_errs'][i] for i in protein_idx]
                    
                    r, p = stats.pearsonr(fields, energies)
                    slope, intercept = np.polyfit(fields, energies, 1)
                    
                    f.write(f"\n  {protein} Statistics:\n")
                    f.write(f"  Correlation (r): {r:.3f}\n")
                    f.write(f"  R²: {r**2:.3f}\n")
                    f.write(f"  P-value: {p:.3e}\n")
                    f.write(f"  Regression equation: y = {slope:.3f}x + {intercept:.3f}\n")
                    f.write(f"  Number of points: {len(fields)}\n")
                    f.write(f"  Mean field: {np.mean(fields):.3f} ± {np.mean(field_errs):.3f} V/Å\n")
                    f.write(f"  Mean energy: {np.mean(energies):.3f} ± {np.mean(energy_errs):.3f} eV\n")
        
        f.write("\nGrouped Categories (Original Values):\n")
        for category, data in grouped_results.items():
            f.write(f"\n{category}:\n")
            f.write(f"Pearson correlation coefficient (r): {data['correlation']:.3f}\n")
            f.write(f"Coefficient of determination (r²): {data['correlation']**2:.3f}\n")
            f.write(f"P-value: {data['p_value']:.3e}\n")
            f.write(f"Regression equation: y = {data['slope']:.3f}x + {data['intercept']:.3f}\n")
            f.write(f"Number of points: {len(data['fields'])}\n")
            f.write(f"Mean field: {np.mean(data['fields']):.3f} ± {np.mean(data['field_errs']):.3f} V/Å\n")
            f.write(f"Mean energy: {np.mean(data['energies']):.3f} ± {np.mean(data['energy_errs']):.3f} eV\n")
            
            # Add protein-specific statistics
            for protein in ['OmcE', 'OmcS', 'OmcZ']:
                protein_idx = [i for i, p in enumerate(data['proteins']) if p == protein]
                if protein_idx:
                    fields = [data['fields'][i] for i in protein_idx]
                    energies = [data['energies'][i] for i in protein_idx]
                    field_errs = [data['field_errs'][i] for i in protein_idx]
                    energy_errs = [data['energy_errs'][i] for i in protein_idx]
                    
                    r, p = stats.pearsonr(fields, energies)
                    slope, intercept = np.polyfit(fields, energies, 1)
                    
                    f.write(f"\n  {protein} Statistics:\n")
                    f.write(f"  Correlation (r): {r:.3f}\n")
                    f.write(f"  R²: {r**2:.3f}\n")
                    f.write(f"  P-value: {p:.3e}\n")
                    f.write(f"  Regression equation: y = {slope:.3f}x + {intercept:.3f}\n")
                    f.write(f"  Number of points: {len(fields)}\n")
                    f.write(f"  Mean field: {np.mean(fields):.3f} ± {np.mean(field_errs):.3f} V/Å\n")
                    f.write(f"  Mean energy: {np.mean(energies):.3f} ± {np.mean(energy_errs):.3f} eV\n")
        
        f.write("\nOverall Correlation (Absolute Values):\n")
        data = overall_results['Overall']
        f.write(f"Pearson correlation coefficient (r): {data['correlation']:.3f}\n")
        f.write(f"Coefficient of determination (r²): {data['correlation']**2:.3f}\n")
        f.write(f"P-value: {data['p_value']:.3e}\n")
        f.write(f"Regression equation: y = {data['slope']:.3f}x + {data['intercept']:.3f}\n")
        f.write(f"Number of points: {len(data['fields'])}\n")
        f.write(f"Mean field: {np.mean(data['fields']):.3f} ± {np.mean(data['field_errs']):.3f} V/Å\n")
        f.write(f"Mean |energy|: {np.mean(data['energies']):.3f} ± {np.mean(data['energy_errs']):.3f} eV\n")
        
        # Add protein-specific statistics for overall correlation
        for protein in ['OmcE', 'OmcS', 'OmcZ']:
            protein_idx = [i for i, p in enumerate(data['proteins']) if p == protein]
            if protein_idx:
                fields = [data['fields'][i] for i in protein_idx]
                energies = [data['energies'][i] for i in protein_idx]
                field_errs = [data['field_errs'][i] for i in protein_idx]
                energy_errs = [data['energy_errs'][i] for i in protein_idx]
                
                r, p = stats.pearsonr(fields, energies)
                slope, intercept = np.polyfit(fields, energies, 1)
                
                f.write(f"\n  {protein} Statistics:\n")
                f.write(f"  Correlation (r): {r:.3f}\n")
                f.write(f"  R²: {r**2:.3f}\n")
                f.write(f"  P-value: {p:.3e}\n")
                f.write(f"  Regression equation: y = {slope:.3f}x + {intercept:.3f}\n")
                f.write(f"  Number of points: {len(fields)}\n")
                f.write(f"  Mean field: {np.mean(fields):.3f} ± {np.mean(field_errs):.3f} V/Å\n")
                f.write(f"  Mean |energy|: {np.mean(energies):.3f} ± {np.mean(energy_errs):.3f} eV\n")

        # Add separator before data summary
        f.write("\n" + "=" * 50 + "\n")
        f.write("Data Summary\n")
        f.write("=" * 50 + "\n")

        # Add data summary
        for protein in proteins_processed:
            f.write(f"\n{protein} Data:\n")
            f.write("-" * 20 + "\n")

            # Get data for this protein
            protein_entries = [data for data in all_data
                             if list(data.values())[0][5] == protein]

            f.write(f"Number of pairs: {len(protein_entries)}\n")
            f.write("\nPairs processed:\n")

            for i, data in enumerate(protein_entries, 1):
                f.write(f"\n  Pair {i}:\n")
                for category in ['Reactant Donor', 'Reactant Acceptor',
                               'Product Donor', 'Product Acceptor']:
                    field, field_err, energy, energy_err, abs_energy, _ = data[category]
                    f.write(f"    {category}:\n")
                    f.write(f"      Field: {field:.3f} ± {field_err:.3f} V/Å\n")
                    f.write(f"      Energy: {energy:.3f} ± {energy_err:.3f} eV\n")
                    f.write(f"      |Energy|: {abs_energy:.3f} eV\n")
                f.write("    " + "-" * 30 + "\n")

        # Add command to reproduce results
        f.write("\n" + "=" * 50 + "\n")
        f.write("Command to reproduce this analysis:\n")
        cmd = "python script.py"
        if args.OmcE:
            cmd += f" -OmcE {' '.join(map(str, args.OmcE))}"
        if args.OmcS:
            cmd += f" -OmcS {' '.join(map(str, args.OmcS))}"
        if args.OmcZ:
            cmd += f" -OmcZ {' '.join(map(str, args.OmcZ))}"
        f.write(cmd + "\n")

if __name__ == "__main__":
    main()

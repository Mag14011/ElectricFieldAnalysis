import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

def get_pairs(residues):
    """Generate pairs of adjacent residues from the list"""
    return list(zip(residues[:-1], residues[1:]))

def read_data_file(donor, acceptor):
    """Read and parse a single data file"""
    filepath = f"r{donor}/{donor}_{acceptor}_md_nofilter/txt/{donor}_{acceptor}_EFieldDependentReorgEng_no_filter.txt"
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
    
    # Store both original and absolute values
    correlations = {
        'Reactant Donor': (values[0], values[1], values[12], np.sqrt(values[13]), abs(values[12])),
        'Reactant Acceptor': (values[2], values[3], values[14], np.sqrt(values[15]), abs(values[14])),
        'Product Donor': (values[4], values[5], values[20], np.sqrt(values[21]), abs(values[20])),
        'Product Acceptor': (values[6], values[7], values[22], np.sqrt(values[23]), abs(values[22]))
    }
    
    return correlations

def get_regression_equation(x, y):
    """Calculate regression equation in the form y = mx + b"""
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept

def analyze_correlations(all_data, use_absolute=False):
    """Calculate correlation statistics based on original or absolute values"""
    energy_idx = 4 if use_absolute else 2
    
    # Dictionary for abbreviated names
    abbrev = {
        'Reactant Donor': 'R.D.',
        'Reactant Acceptor': 'R.A.',
        'Product Donor': 'P.D.',
        'Product Acceptor': 'P.A.',
        'All Donors': 'Donors',
        'All Acceptors': 'Acceptors',
        'Overall': 'Overall'
    }
    
    categories = ['Reactant Donor', 'Reactant Acceptor', 'Product Donor', 'Product Acceptor']
    results = {}
    
    # For grouped analysis
    grouped_data = {
        'All Donors': {'fields': [], 'field_errs': [], 'energies': [], 'energy_errs': []},
        'All Acceptors': {'fields': [], 'field_errs': [], 'energies': [], 'energy_errs': []}
    }
    
    # Process individual categories
    for category in categories:
        fields = [data[category][0] for data in all_data]
        field_errs = [data[category][1] for data in all_data]
        energies = [data[category][energy_idx] for data in all_data]
        energy_errs = [data[category][3] for data in all_data]
        
        r, p = stats.pearsonr(fields, energies)
        slope, intercept = get_regression_equation(fields, energies)
        
        results[category] = {
            'fields': fields,
            'field_errs': field_errs,
            'energies': energies,
            'energy_errs': energy_errs,
            'correlation': r,
            'p_value': p,
            'slope': slope,
            'intercept': intercept,
            'abbrev': abbrev[category]
        }
        
        # Add to grouped data
        if 'Donor' in category:
            grouped_data['All Donors']['fields'].extend(fields)
            grouped_data['All Donors']['field_errs'].extend(field_errs)
            grouped_data['All Donors']['energies'].extend(energies)
            grouped_data['All Donors']['energy_errs'].extend(energy_errs)
        else:
            grouped_data['All Acceptors']['fields'].extend(fields)
            grouped_data['All Acceptors']['field_errs'].extend(field_errs)
            grouped_data['All Acceptors']['energies'].extend(energies)
            grouped_data['All Acceptors']['energy_errs'].extend(energy_errs)
    
    # Add grouped results if not using absolute values
    if not use_absolute:
        for group_name, data in grouped_data.items():
            r, p = stats.pearsonr(data['fields'], data['energies'])
            slope, intercept = get_regression_equation(data['fields'], data['energies'])
            results[group_name] = {
                'fields': data['fields'],
                'field_errs': data['field_errs'],
                'energies': data['energies'],
                'energy_errs': data['energy_errs'],
                'correlation': r,
                'p_value': p,
                'slope': slope,
                'intercept': intercept,
                'abbrev': abbrev[group_name]
            }
    
    # Calculate overall correlation if using absolute values
    if use_absolute:
        all_fields = []
        all_energies = []
        all_field_errs = []
        all_energy_errs = []
        for category in categories:
            all_fields.extend(results[category]['fields'])
            all_energies.extend(results[category]['energies'])
            all_field_errs.extend(results[category]['field_errs'])
            all_energy_errs.extend(results[category]['energy_errs'])
        
        r, p = stats.pearsonr(all_fields, all_energies)
        slope, intercept = get_regression_equation(all_fields, all_energies)
        results['Overall'] = {
            'fields': all_fields,
            'field_errs': all_field_errs,
            'energies': all_energies,
            'energy_errs': all_energy_errs,
            'correlation': r,
            'p_value': p,
            'slope': slope,
            'intercept': intercept,
            'abbrev': abbrev['Overall']
        }
    
    return results

def create_plot(results, ylabel, filename, width=7):
    """Create a single correlation plot with regression lines and error bands"""
    plt.figure(figsize=(width, width * 1.000))

    colors = {
        'Reactant Donor': '#ff7f0e',  # orange
        'Product Donor': '#d62728',    # red
        'Reactant Acceptor': '#1f77b4',  # blue
        'Product Acceptor': '#9467bd',   # purple
        'All Donors': '#e6550d',         # dark orange
        'All Acceptors': '#3182bd',      # dark blue
        'Overall': '#666666'             # gray
    }

    markers = {
        'Reactant Donor': 'o',
        'Product Donor': 's',
        'Reactant Acceptor': 'o',
        'Product Acceptor': 's',
        'All Donors': '^',
        'All Acceptors': '^',
        'Overall': 'o'
    }

    fillstyles = {k: 'full' if 'Donor' in k or k == 'Overall' else 'none' for k in colors.keys()}

    # First, draw all error bands (with sorted x-values)
    for category in results.keys():
        data = results[category]

        # Sort data points by x-value for continuous error bands
        sort_idx = np.argsort(data['fields'])
        x = np.array(data['fields'])[sort_idx]
        y = np.array(data['energies'])[sort_idx]
        xerr = np.array(data['field_errs'])[sort_idx]
        yerr = np.array(data['energy_errs'])[sort_idx]

        plt.fill_between(x, y - yerr, y + yerr, color=colors[category], alpha=0.1, zorder=1)
        plt.fill_betweenx(y, x - xerr, x + xerr, color=colors[category], alpha=0.1, zorder=1)

    # Then draw regression lines
    for category in results.keys():
        data = results[category]
        x = np.array(data['fields'])
        slope, intercept = np.polyfit(x, data['energies'], 1)
        x_range = np.array([min(x), max(x)])
        y_range = slope * x_range + intercept
        plt.plot(x_range, y_range, '--', color=colors[category], alpha=0.8, zorder=2)

    # Finally plot points
    for category in results.keys():
        data = results[category]
        plt.scatter(data['fields'], data['energies'],
                   color=colors[category],
                   marker=markers[category],
                   s=64,
                   facecolors=colors[category] if fillstyles[category] == 'full' else 'none',
                   edgecolors=colors[category],
                   label=f"{data['abbrev']} (r={data['correlation']:.3f})",
                   alpha=0.6,
                   zorder=3)

    plt.xlabel('Electric Field Magnitude (V/Å)')
    plt.ylabel(ylabel)
    plt.tick_params(direction='in', which='both', top=True, right=True)
    plt.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Check if residues were provided as command line arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py residue1 residue2 residue3 ...")
        print("Example: python script.py 562 553 559 550 541 547 544 844")
        sys.exit(1)

    # Convert command line arguments to list of integers
    try:
        residues = [int(arg) for arg in sys.argv[1:]]
    except ValueError:
        print("Error: All arguments must be integer residue IDs")
        sys.exit(1)

    # Check if we have at least 2 residues to make a pair
    if len(residues) < 2:
        print("Error: Need at least 2 residues to form a pair")
        sys.exit(1)

    pairs = get_pairs(residues)
    print(f"Processing pairs: {pairs}")

    all_data = []
    for donor, acceptor in pairs:
        try:
            data = read_data_file(donor, acceptor)
            all_data.append(data)
            print(f"Successfully processed pair {donor}-{acceptor}")
        except FileNotFoundError:
            print(f"Warning: Could not find file for pair {donor}-{acceptor}")
        except Exception as e:
            print(f"Error processing pair {donor}-{acceptor}: {str(e)}")

    if not all_data:
        print("No data was successfully collected. Check file paths and formats.")
        return

    # Create three different sets of results
    individual_results = {k: v for k, v in analyze_correlations(all_data, use_absolute=False).items()
                        if k in ['Reactant Donor', 'Reactant Acceptor', 'Product Donor', 'Product Acceptor']}

    grouped_results = {k: v for k, v in analyze_correlations(all_data, use_absolute=False).items()
                      if k in ['All Donors', 'All Acceptors']}

    overall_results = analyze_correlations(all_data, use_absolute=True)

    # Create output directory using first and last residue IDs
    output_dir = f"analysis_{residues[0]}_{residues[-1]}"
    os.makedirs(output_dir, exist_ok=True)

    # Create three plots
    for width in [3.3, 7]:
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

    # Save statistics to a file
    with open(f"{output_dir}/correlation_statistics.txt", 'w') as f:
        f.write("Correlation Statistics:\n")
        f.write("-" * 50 + "\n")
        
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
        
        f.write("\nOverall Correlation (Absolute Values):\n")
        data = overall_results['Overall']
        f.write(f"Pearson correlation coefficient (r): {data['correlation']:.3f}\n")
        f.write(f"Coefficient of determination (r²): {data['correlation']**2:.3f}\n")
        f.write(f"P-value: {data['p_value']:.3e}\n")
        f.write(f"Regression equation: y = {data['slope']:.3f}x + {data['intercept']:.3f}\n")
        f.write(f"Number of points: {len(data['fields'])}\n")
        f.write(f"Mean field: {np.mean(data['fields']):.3f} ± {np.mean(data['field_errs']):.3f} V/Å\n")
        f.write(f"Mean |energy|: {np.mean(data['energies']):.3f} ± {np.mean(data['energy_errs']):.3f} eV\n")

if __name__ == "__main__":
    main()

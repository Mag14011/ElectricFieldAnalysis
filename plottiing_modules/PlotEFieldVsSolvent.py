import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects

def read_and_process_data(field_file, wrap_file):
    """
    Read and process the electric field and wrapping statistics data files.
    """
    # Read electric field data
    field_data = pd.read_csv(field_file, 
                            skiprows=2,
                            sep='\s+',
                            names=['Distance', 'AvgMagnitude', 'StdDev'])
    
    # Convert electric field from MV/cm to V/Å
    field_data['AvgMagnitude_VA'] = field_data['AvgMagnitude'] * 0.01
    field_data['StdDev_VA'] = field_data['StdDev'] * 0.01
    
    # Read wrapping statistics data
    wrap_data = pd.read_csv(wrap_file,
                           skiprows=2,
                           sep='\s+',
                           names=['Distance', 'Avg_Total_Res', 'Avg_Wrapped_Res', 
                                'Avg_Total_Atoms', 'Avg_Wrapped_Atoms', 'Avg_Water_Res',
                                'Avg_Wrapped_Water', 'Avg_Ion_Res', 'Avg_Wrapped_Ions',
                                'Avg_Other_Res', 'Avg_Wrapped_Other'])
    
    return field_data, wrap_data

def create_dual_axis_plot(field_data, wrap_data, log_scale=False, output_file='field_solvent_analysis.png'):
    """
    Create a dual-axis plot showing electric field magnitude and solvent molecules.
    """
    # Create figure with square dimensions
    fig, ax1 = plt.subplots(figsize=(3.3, 3.3))
    
    # Colors
    color_field = 'purple'
    color_water = 'blue'
    color_ions = 'orange'
    
    # Direct ticks inward for both axes
    ax1.tick_params(axis='both', direction='in', which='both')
    
    # Plot electric field data with error bars on left axis
    ax1.errorbar(field_data['Distance'], 
                 field_data['AvgMagnitude_VA'],
                 yerr=field_data['StdDev_VA'],
                 color=color_field,
                 marker='o',
                 markeredgecolor='black',
                 linestyle=':',
                 capsize=3)
    
    ax1.set_xlabel('Distance (Å)')
    ax1.set_ylabel('Average Electric Field Magnitude (V/Å)', color=color_field)
    ax1.tick_params(axis='y', labelcolor=color_field)
    
    # Create secondary axis for molecule counts
    ax2 = ax1.twinx()
    
    # Direct ticks inward for secondary axis
    ax2.tick_params(axis='y', direction='in', which='both')
    
    # For log scale, replace zeros with a small number to avoid log(0)
    if log_scale:
        water_data = wrap_data['Avg_Water_Res'].replace(0, 1e-1)
        ion_data = wrap_data['Avg_Ion_Res'].replace(0, 1e-1)
        ax2.set_yscale('log')
    else:
        water_data = wrap_data['Avg_Water_Res']
        ion_data = wrap_data['Avg_Ion_Res']
    
    # Plot water molecules and ions
    ax2.plot(wrap_data['Distance'], 
             water_data,
             color=color_water,
             marker='s',
             markeredgecolor='black',
             linestyle=':')
    
    ax2.plot(wrap_data['Distance'], 
             ion_data,
             color=color_ions,
             marker='^',
             markeredgecolor='black',
             linestyle=':')
    
    # Set up the figure to allow room for right label
    plt.subplots_adjust(right=0.85)
    
    # Create right y-axis label with better spacing
    ax2.text(1.3, 0.71, 'Average # of ', 
             rotation=270,
             transform=ax2.transAxes,
             va='center',
             ha='center')

    water_text = ax2.text(1.31, 0.45, 'Water', 
                         rotation=270,
                         color=color_water,
                         transform=ax2.transAxes,
                         va='center',
                         ha='center')
    water_text.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='black'),
                               path_effects.Normal()])
        
    ax2.text(1.3, 0.35, '/', 
             rotation=270,
             transform=ax2.transAxes,
             va='center',
             ha='center')
    
    ion_text = ax2.text(1.31, 0.28, 'Ion', 
                       rotation=270,
                       color=color_ions,
                       transform=ax2.transAxes,
                       va='center',
                       ha='center')
    ion_text.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='black'),
                             path_effects.Normal()])
    
    ax2.text(1.3, 0.09, 'Molecules', 
             rotation=270,
             transform=ax2.transAxes,
             va='center',
             ha='center')
        
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to execute the analysis and create both linear and log-scale plots.
    """
    field_file = 'ElecField_cutoff_summary.dat'
    wrap_file = 'wrapping_statistics_summary.dat'
    
    try:
        # Read and process data
        field_data, wrap_data = read_and_process_data(field_file, wrap_file)
        
        # Create linear scale plot
        create_dual_axis_plot(field_data, wrap_data, log_scale=False, 
                              output_file='field_solvent_analysis_linear.png')
        
        # Create log scale plot
        create_dual_axis_plot(field_data, wrap_data, log_scale=True, 
                              output_file='field_solvent_analysis_log.png')
        
        print("Analysis complete. Plots saved as 'field_solvent_analysis_linear.png' and 'field_solvent_analysis_log.png'")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find input file - {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")

if __name__ == "__main__":
    main()

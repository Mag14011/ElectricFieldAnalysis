# Electric Field Analysis at Probe Atoms

A Python package for computing and analyzing electric fields at specific probe atoms in molecular dynamics trajectories. This tool is particularly useful for analyzing electric fields in biomolecular systems, supporting both total field calculations and detailed per-residue contribution analysis.

## Features

- Calculate electric fields from molecular dynamics trajectories at specified probe atoms
- Support for parallel processing in total field calculations
- Detailed per-residue contribution analysis
- Flexible solvent handling with customizable cutoffs
- Support for cutoff range analysis
- Comprehensive wrapping statistics for periodic boundary conditions
- Multiple probe analysis in a single run
- Output generation for visualization and further analysis

## Installation

### Prerequisites

- Python >= 3.7
- NumPy
- SciPy
- MDAnalysis

### Installation Steps

1. Clone this repository:
```bash
git clone [repository-url]
cd EFieldAnalysis
```

2. Install using pip:
```bash
pip install .
```

## Usage

The package can be run either using command-line arguments or a configuration file.

### Using Configuration File

1. Generate a template configuration file:
```bash
efieldanalysis --generate-template
```

2. Edit the generated `temp.config` file to specify your analysis parameters:
```ini
# Select the topology and trajectory
topology       = system.parm7
trajectory     = trajectory.xtc

# Select analysis type (total/contributions/all)
analysis       = all

# Frame selection
frames         = all

# Probe atom selection
probe          = (resname HEH and resid 653 and name FE)

# Environment selection
environment    = (not resname WAT and not resname Na+ and not resname Cl-) and not (resname HEH and resid XXX)

# Solvent selection
solvent        = resname WAT or resname Na+

# Solvent cutoff (Å)
solvent_cutoff = 10

# Number of processors
processors     = 10
```

3. Run the analysis:
```bash
efieldanalysis --settings temp.config
```

### Using Command Line Arguments

```bash
efieldanalysis --topology system.parm7 \
               --trajectory trajectory.xtc \
               --probe "resname HEH and resid 653 and name FE" \
               --environment "(not resname WAT) and not (resname HEH and resid 653)" \
               --solvent "resname WAT or resname Na+" \
               --solvent-cutoff 10 \
               --processors 10 \
               --analysis all
```

## Output Files

For each probe atom, a directory is created (e.g., `FE-HEH653/`) containing:

### Total Field Analysis
- `ElecField.dat`: Electric field values and statistics for each frame
- `wrapping_statistics.dat`: Statistics about molecular wrapping
- `probe.dat`: Probe atom positions

### Cutoff Range Analysis
- `ElecField_cutoff_X.X.dat`: Field values for each cutoff
- `wrapping_statistics_cutoff_X.X.dat`: Wrapping statistics for each cutoff
- `ElecField_cutoff_summary.dat`: Summary across all cutoffs

### Contribution Analysis
- `average_residue_contributions.dat`: Per-residue and component statistics

## Analysis Types

1. **Total Field Analysis** (`analysis = total`)
   - Parallel calculation of total electric field
   - Supports cutoff range analysis
   - Generates per-frame field values and statistics

2. **Contribution Analysis** (`analysis = contributions`)
   - Sequential calculation of per-residue contributions
   - Detailed component analysis (protein/solvent/other)
   - Average contribution statistics

3. **Complete Analysis** (`analysis = all`)
   - Performs both total field and contribution analyses

## Technical Notes

- The package uses periodic boundary conditions and handles molecular wrapping
- Electric fields are reported in MV/cm
- Supports multiple probe atoms in a single run
- Automatically excludes probe atoms from environment selections
- Environment selections can use the XXX placeholder for probe residue IDs

## Citation

To be updated.

## License

MIT license


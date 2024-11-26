from setuptools import setup, find_packages

setup(
    name="EFieldAnalysis",
    version="1.0.0",
    description="Compute the electric field at a probe atom.",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "MDAnalysis",
    ],
    entry_points={
        'console_scripts': [
            'efieldanalysis=multi_probe_efield_analysis.MultiProbeEFieldAnalysis:main',
        ],
    },
    author="Matthew Guberman-Pfeffer",
    python_requires=">=3.7",
)


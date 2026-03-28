from setuptools import setup, find_packages

setup(
    name="cascrop",
    version="0.1.0",
    description="CasCrop: Crop Waste as Economic Contagion via Graph Neural Networks",
    author="Keshav Krishnan",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.4.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "xgboost>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "geopandas>=0.13.0",
        "pyarrow>=12.0.0",
        "requests>=2.31.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)

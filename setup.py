"""
Setup configuration for ML Toolbox package
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="ml-toolbox",
    version="1.0.0",
    description="Comprehensive Machine Learning Toolbox with AI Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DJMcC",
    author_email="",
    url="https://github.com/DJMcClellan1966/ML-ToolBox",
    packages=find_packages(exclude=["tests", "examples", "*.tests", "*.tests.*", "tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "full": requirements,
        "advanced": [
            "imbalanced-learn>=0.8.0",
            "statsmodels>=0.13.0",
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
        ],
        "deep-learning": [
            "tensorflow>=2.8.0",
            "torch>=1.11.0",
        ],
        "nlp": [
            "sentence-transformers>=2.2.0",
            "transformers>=4.20.0",
        ],
        "interpretability": [
            "shap>=0.40.0",
            "lime>=0.2.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "plotly>=5.10.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="machine-learning ml ai agents toolbox scikit-learn",
    include_package_data=True,
    zip_safe=False,
)

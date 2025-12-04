import setuptools
from pathlib import Path

here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="embo-framework",
    version="0.0.1",
    author="Subrata Mukherjee",
    author_email="mukherjees2@ornl.gov",
    description="Empirical Model-Based Optimization framework for flowrate scheduling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/submukherjee/EMBO_framework",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "joblib",
        "scipy",
        "openpyxl"
    ],
)

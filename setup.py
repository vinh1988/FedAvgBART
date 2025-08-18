from setuptools import setup, find_packages

setup(
    name="distilbart_federated",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.8.0",
        "transformers>=4.10.0",
        "datasets>=1.12.0",
        "numpy>=1.19.0",
        "tqdm>=4.62.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="DistilBART for Federated Learning",
    url="https://github.com/your-username/distilbart-federated",
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="adversarial-traffic-signs",
    version="1.0.0",
    author="Amitabh Das",
    author_email="das.am@northeastern.edu",
    description="Adversarial Attacks on Traffic Sign Recognition Using Diffusion Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amitabhdas/adversarial-traffic-signs",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=9.0.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
        "timm>=0.9.0",
        "transformers>=4.30.0",
        "torchattacks>=3.4.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "diffusion": [
            "diffusers>=0.20.0",
            "accelerate>=0.20.0",
        ],
        "metrics": [
            "lpips>=0.1.4",
            "scikit-image>=0.21.0",
        ],
        "all": [
            "diffusers>=0.20.0",
            "accelerate>=0.20.0",
            "lpips>=0.1.4",
            "scikit-image>=0.21.0",
        ],
    },
)

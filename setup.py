from setuptools import setup, find_packages

setup(
    name="wan-profiler",
    version="0.1.0",
    author="Ishmael Affum Kwakye",
    author_email="",
    description="Compute profiling of Wan 1.3B for CPU-native video generation research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/IshCPU-VideoGenLab/wan-profiler",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "transformers>=4.35.0",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.19.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.7.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    entry_points={
        "console_scripts": [
            "wan-profiler=wan_profiler.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

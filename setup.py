from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ghana-rainfall-prediction",
    version="1.0.0",
    author="Agboado Bernard",
    description="AI-Powered Weekly and Monthly Rainfall Forecasting across Ghana's Climate Zones",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "flake8>=6.0.0",
            "black>=23.7.0",
            "pylint>=2.17.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "rainfall-predict=src.predict:main",
        ],
    },
)
from setuptools import setup, find_packages

setup(
    name="astro_compass",
    version="0.1.0",
    description="Autonomous spacecraft trajectory optimization using reinforcement learning",
    author="Michael Schmidt",
    author_email="mschmid4@umd.edu",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    python_requires=">=3.8",
    install_requires=[
        "gymnasium",
        "stable-baselines3",
        "torch",
        "matplotlib",
        "numpy",
    ],
)
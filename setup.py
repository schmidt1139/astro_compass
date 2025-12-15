import os

from setuptools import find_packages, setup


# Read requirements from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(req_path, "r") as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith("#"):
                requirements.append(line)
        return requirements


setup(
    name="astro_compass",
    version="0.1.0",
    description="Autonomous spacecraft trajectory optimization using reinforcement learning",
    author="Michael Schmidt",
    author_email="mschmid4@umd.edu",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=read_requirements(),
)

# Linux Migration Guide for astro_compass

This guide provides step-by-step instructions for migrating the astro_compass project from Windows to a Linux environment.

## Pre-Migration Checklist (On Windows)

### 1. Run the Migration Script

```bash
# First, run in dry-run mode to see what will change
python migrate_to_linux.py --dry-run

# If everything looks good, run the actual migration
python migrate_to_linux.py
```

This script will:
- Convert all hardcoded Windows paths (`\`) to use `os.path.join()`
- Ensure `import os` is present in files that need it
- Make paths cross-platform compatible

### 2. Verify Changes

```bash
# Check for any remaining Windows-specific paths
python -c "import re; import pathlib; [print(f) for f in pathlib.Path('.').rglob('*.py') if any('\\\\' in line for line in open(f, encoding='utf-8'))]"
```

### 3. Commit Changes

```bash
git add -A
git commit -m "Migrate paths to be Linux-compatible using os.path.join()"
git push
```

## Migration to Linux Environment

### 1. System Requirements

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv python3-dev
sudo apt-get install -y python3-tk  # For matplotlib GUI
sudo apt-get install -y git
```

**RHEL/CentOS/Fedora:**
```bash
sudo yum install -y python3 python3-pip python3-devel
sudo yum install -y python3-tkinter
sudo yum install -y git
```

### 2. Clone Repository

```bash
cd ~
git clone https://github.com/schmidt1139/astro_compass.git
cd astro_compass
```

Or if you're transferring files manually:
```bash
# On Windows (using WSL or Git Bash):
scp -r c:/Users/micha/MSI_Data/Masters_Thesis/astro_compass user@linux-host:~/

# On Linux:
cd ~/astro_compass
```

### 3. Set Up Python Virtual Environment (IMPORTANT for Shared Systems)

**What is a Virtual Environment?**
A virtual environment creates an **isolated Python environment** that keeps your dependencies separate from:
- The system Python packages
- Other users' Python packages
- Other projects' dependencies

**On a shared Linux system, this means:**
- ✅ Your packages (gymnasium, torch, etc.) **only** affect your project
- ✅ You can use any package versions without conflicts
- ✅ Other users are completely unaffected by your installations
- ✅ You don't need sudo/root access to install packages
- ✅ Each user can have different versions of the same package

**Setup:**
```bash
# Create virtual environment (creates a directory called 'venv' in your project)
python3 -m venv venv

# Activate virtual environment (MUST do this before installing packages!)
source venv/bin/activate  # Note: Different from Windows (venv\Scripts\activate)

# Your prompt will change to show you're in the venv:
# (venv) user@server:~/astro_compass$

# Upgrade pip within your isolated environment
pip install --upgrade pip setuptools wheel

# Check you're using the venv Python (not system Python)
which python  # Should show: /home/youruser/astro_compass/venv/bin/python
```

**IMPORTANT:** Always activate the venv before working:
```bash
# Every time you log in or open a new terminal:
cd ~/astro_compass
source venv/bin/activate

# Now you can run your scripts with isolated dependencies
python src/scripts/datagen_Hamiltonian_TBR_controller.py

# When done, deactivate to return to system Python
deactivate
```

### 4. Install Dependencies (Into Your Isolated Environment)

**CRITICAL:** Make sure your venv is activated (you should see `(venv)` in your prompt)

```bash
# Verify you're in the virtual environment
which python  # Should show: /home/youruser/astro_compass/venv/bin/python

# Option 1: Install using requirements.txt (recommended)
pip install -r requirements.txt

# Option 2: Install the package in development mode (uses requirements.txt)
pip install -e .

# All packages are installed to: venv/lib/python3.X/site-packages/
# They do NOT go to the system Python or affect other users!

# Verify installation
python -c "from astro_compass.envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env; print('✓ Import successful')"

# Check what's installed in YOUR environment only
pip list
```

**What gets installed:**
- `gymnasium>=0.29.0` - Reinforcement learning environment framework
- `torch>=2.0.0` - PyTorch for neural networks
- `stable-baselines3>=2.0.0` - RL algorithms (SAC, etc.)
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.10.0` - Scientific computing (integrators, optimizers)
- `matplotlib>=3.7.0` - Plotting and visualization

**What just happened:**
- Dependencies were installed **only** to `~/astro_compass/venv/`
- System Python and other users are **completely unaffected**
- You can safely use any versions you need

### 5. Install PyTorch (if needed)

Check the [PyTorch website](https://pytorch.org/) for the correct Linux installation command based on your system:

```bash
# Example for CPU-only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 6. Convert Line Endings (Optional but Recommended)

```bash
# Install dos2unix
sudo apt-get install dos2unix  # Ubuntu/Debian
# OR
sudo yum install dos2unix      # RHEL/CentOS/Fedora

# Convert all Python files
find . -name "*.py" -type f -exec dos2unix {} \;

# Convert shell scripts if any
find . -name "*.sh" -type f -exec dos2unix {} \;
```

### 7. Set Executable Permissions

```bash
# Make scripts executable
chmod +x migrate_to_linux.py
find src/scripts -name "*.py" -type f -exec chmod +x {} \;
```

## Best Practices for Shared Systems

### Managing Your Virtual Environment

**DO:**
- ✅ Always activate your venv before working: `source venv/bin/activate`
- ✅ Keep your venv in your project directory (not shared)
- ✅ Add `venv/` to `.gitignore` (virtual envs shouldn't be in git)
- ✅ Use `pip list` to see what's installed in your environment
- ✅ Create a `requirements.txt` to document your dependencies:
  ```bash
  pip freeze > requirements.txt
  ```

**DON'T:**
- ❌ Install packages with `sudo pip` (affects system Python)
- ❌ Try to install to system Python without venv
- ❌ Share your venv directory with other users
- ❌ Commit your venv to git (too large, platform-specific)

### Workflow on Shared System

```bash
# Every time you log in:
cd ~/astro_compass
source venv/bin/activate    # Activate YOUR environment

# Work on your project (all packages are isolated)
python src/scripts/your_script.py

# When done for the session:
deactivate                  # Return to system Python
```

### Checking Isolation

```bash
# Verify you're using YOUR Python, not system Python
which python
# Should show: /home/youruser/astro_compass/venv/bin/python
# NOT: /usr/bin/python3

# Verify packages are in YOUR environment
pip show torch
# Location should show: .../astro_compass/venv/lib/python3.X/site-packages

# Compare system vs venv packages
deactivate                  # Exit venv
pip list                    # System packages (probably minimal)
source venv/bin/activate    # Enter venv
pip list                    # Your packages (gymnasium, torch, etc.)
```

### Multiple Projects on Same System

You can have multiple virtual environments:
```bash
# Project 1
~/astro_compass/venv/       # Has torch 2.0, gymnasium 0.29

# Project 2  
~/other_project/venv/       # Has torch 1.9, gym 0.21

# No conflicts! Each project uses its own environment
```

## Verification Steps

### 1. Run Unit Tests

```bash
# Make sure venv is activated!
source venv/bin/activate

# Run all tests
python3 -m pytest tests/ -v

# Run specific test
python3 -m pytest tests/test_eom_and_propagation.py -v
```

### 2. Test Data Generation

```bash
# Test the main data generation script
python3 src/scripts/datagen_Hamiltonian_TBR_controller.py
```

### 3. Verify Paths Work Correctly

```bash
# Check that data directories are accessible
ls -la data/
ls -la data/test_data/
ls -la data/neural_networks/
```

### 4. Test Plotting (if using GUI)

```bash
# Ensure X11 forwarding works if using SSH
# On your SSH client: ssh -X user@linux-host

# Test matplotlib
python3 -c "import matplotlib.pyplot as plt; plt.plot([1,2,3]); print('✓ Matplotlib works')"
```

## Common Issues and Solutions

### Issue 1: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'xyz'`

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall package
pip install -e .
```

### Issue 2: Permission Denied

**Problem:** `Permission denied` when running scripts

**Solution:**
```bash
chmod +x src/scripts/your_script.py
# Or run with python explicitly:
python3 src/scripts/your_script.py
```

### Issue 3: Display Issues with Matplotlib

**Problem:** `_tkinter.TclError: no display name and no $DISPLAY environment variable`

**Solution:**
```bash
# Use non-GUI backend
export MPLBACKEND=Agg

# Or set in your Python script:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

### Issue 4: Case Sensitivity Issues

**Problem:** `FileNotFoundError` for files that exist

**Solution:**
Linux filesystems are case-sensitive. Check exact filename:
```bash
# Find files case-insensitively
find . -iname "filename.py"

# Rename if needed
mv WrongCase.py correct_case.py
```

### Issue 5: Path Separator Issues

**Problem:** Paths still have Windows separators

**Solution:**
```bash
# Search for remaining Windows paths
grep -r '\\\\' --include="*.py" .

# The migration script should have fixed these, but if you find any:
# Use os.path.join() or pathlib.Path instead
```

## Performance Considerations

### 1. Parallel Processing

Linux often handles parallel processing better than Windows. You can increase worker counts:

```python
# In your parallel scripts
import multiprocessing
num_workers = multiprocessing.cpu_count()  # Use all cores
```

### 2. Monitoring Resources

```bash
# Monitor CPU and memory during long runs
htop

# Monitor disk I/O
iotop

# Check process status
ps aux | grep python
```

### 3. Running Long Jobs

```bash
# Use nohup for long-running jobs
nohup python3 src/scripts/datagen_Hamiltonian_TBR_controller.py > output.log 2>&1 &

# Or use screen/tmux
screen -S datagen
python3 src/scripts/datagen_Hamiltonian_TBR_controller.py
# Detach: Ctrl+A, then D
# Reattach: screen -r datagen
```

## Optimization for Linux

### 1. Use Virtual Environment Management

Consider using `conda` for better environment management:

```bash
# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment
conda create -n astro_compass python=3.10
conda activate astro_compass
pip install -e .
```

### 2. Optimize NumPy/SciPy

Install optimized BLAS libraries:

```bash
sudo apt-get install libopenblas-dev liblapack-dev
pip install --upgrade numpy scipy
```

### 3. GPU Acceleration (if available)

```bash
# Check for NVIDIA GPU
nvidia-smi

# Install CUDA toolkit
# Follow: https://developer.nvidia.com/cuda-downloads

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Directory Structure After Migration

Your directory structure should remain the same:

```
astro_compass/
├── migrate_to_linux.py          # Migration script
├── LINUX_MIGRATION_GUIDE.md     # This guide
├── setup.py
├── README.md
├── data/
│   ├── config/
│   ├── neural_networks/
│   ├── plots/
│   ├── support_files/
│   └── test_data/
├── src/
│   ├── python/
│   │   ├── constants/
│   │   ├── core/
│   │   ├── envs/
│   │   └── utils/
│   └── scripts/
└── tests/
```

## Post-Migration Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -e .`)
- [ ] Migration script executed successfully
- [ ] Line endings converted (optional)
- [ ] Unit tests pass (`python3 -m pytest tests/`)
- [ ] Data generation script runs without errors
- [ ] Plotting works (GUI or file output)
- [ ] File permissions set correctly
- [ ] Git repository status clean

## Additional Resources

- **Python Virtual Environments:** https://docs.python.org/3/tutorial/venv.html
- **PyTorch Installation:** https://pytorch.org/get-started/locally/
- **SciPy Installation:** https://scipy.org/install/
- **Matplotlib Backends:** https://matplotlib.org/stable/users/explain/backends.html

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Verify all paths use `os.path.join()` or `pathlib.Path`
3. Ensure virtual environment is activated
4. Check Python version: `python3 --version`
5. Verify dependencies: `pip list`
6. Check file permissions: `ls -la`

## Contact

For issues specific to this migration, check:
- Project Issues: https://github.com/schmidt1139/astro_compass/issues
- Branch: 86-misc-paper-updates

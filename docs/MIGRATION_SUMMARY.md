# Linux Migration Summary

## Overview
I've created a comprehensive migration package for moving your `astro_compass` project from Windows to Linux.

## What's Been Created

### 1. `migrate_to_linux.py` - Automated Migration Script
This script automatically fixes all Windows-specific paths in your codebase:

### 2. `LINUX_MIGRATION_GUIDE.md` - Complete Migration Guide
Comprehensive step-by-step guide covering:

### 3. `VENV_QUICK_REFERENCE.md` - Virtual Environment Guide for Shared Systems ⭐
**NEW!** Essential guide for working on shared Linux systems:
- Explains how virtual environments keep your dependencies isolated
- Why other users won't be affected by your package installations
- Daily workflow on shared servers
- Troubleshooting common venv issues
- Verification commands to ensure proper isolation

### 4. `setup_linux_env.sh` - Automated Linux Setup Script
Bash script that handles everything on the Linux server:
- Creates virtual environment automatically
- Installs all dependencies in isolation
- Verifies proper installation
- Tests key imports
- Shows isolation status
- Creates helper activation script

## Files Summary

### 1. `migrate_to_linux.py` - Automated Migration Script
This script automatically fixes all Windows-specific paths in your codebase:

**What it does:**
- Converts hardcoded Windows backslashes to use `os.path.join()`
- Fixes paths inside existing `os.path.join()` calls that still have backslashes
- Converts standalone path strings to `os.path.join()` calls
- Automatically adds `import os` where needed
- Provides dry-run mode to preview changes

**Example transformations:**
```python
# Before:
"data\\test_data\\output\\"

# After:
os.path.join("data", "test_data", "output")
```

```python
# Before:
os.path.join(os.getcwd(), "data\\neural_networks\\")

# After:
os.path.join(os.getcwd(), "data", "neural_networks")
```

### 2. `LINUX_MIGRATION_GUIDE.md` - Complete Migration Guide  
Comprehensive step-by-step guide covering:
- Pre-migration checklist
- System requirements for Linux
- **Virtual environment setup for shared systems** ⭐
- Dependency installation (isolated per user)
- Testing procedures
- Common issues and solutions
- Performance optimization tips
- Post-migration checklist

## Quick Start

### On Windows (Before Migration)

**Step 1: Preview changes**
```powershell
python migrate_to_linux.py --dry-run
```

**Step 2: Apply changes**
```powershell
python migrate_to_linux.py
```

**Step 3: Commit to git**
```powershell
git add -A
git commit -m "Migrate to Linux-compatible paths"
git push
```

### On Linux (After Transfer)

**Option A: Automated Setup (Recommended)**
```bash
cd ~/astro_compass
chmod +x setup_linux_env.sh
./setup_linux_env.sh
# This script handles everything automatically!
```

**Option B: Manual Setup**
```bash
cd ~/astro_compass

# Step 1: Create isolated virtual environment (YOUR packages only!)
python3 -m venv venv

# Step 2: Activate YOUR environment
source venv/bin/activate

# Step 3: Install dependencies (only to YOUR venv, not system!)
pip install -e .

# Step 4: Run tests
python3 -m pytest tests/ -v
```

**Daily Usage:**
```bash
cd ~/astro_compass
source venv/bin/activate      # ALWAYS do this first!
python your_script.py         # Uses YOUR isolated packages
deactivate                    # When done
```

## Files That Will Be Updated

The migration script will update **19 files**:

### Scripts (7 files):
1. `src/scripts/datagen_Hamiltonian_TBR_controller.py`
2. `src/scripts/ingest_ephem_and_plot.py`
3. `src/scripts/SAC_seeded_training.py`
4. `src/scripts/train_alpha_network.py`
5. `src/scripts/train_neural_network.py`
6. `src/scripts/solve_two_body_transfer_and_write_ephem_script.py`
7. `src/scripts/generate_nn_training_data_parallel.py`

### Tests (12 files):
1. `tests/test_write_ephemeris.py`
2. `tests/test_TBR_env.py`
3. `tests/test_seeded_SAC_training.py`
4. `tests/test_SAC_training_TBR.py`
5. `tests/test_SAC_training.py`
6. `tests/test_random_env_reset.py`
7. `tests/test_Hamiltonian_TBR_controller.py`
8. `tests/test_Hamiltonians.py`
9. `tests/test_eom_and_propagation.py`
10. `tests/test_env_step_with_nn_action.py`
11. `tests/test_env_step_with_action.py`
12. `tests/test_env_step_no_action.py`

## Key Benefits

### ✅ Cross-Platform Compatibility
- All paths will work on Windows, Linux, and macOS
- No more hardcoded backslashes

### ✅ Cleaner Code
- Proper use of `os.path.join()` throughout
- Consistent path handling

### ✅ Automated Process
- No manual find-and-replace needed
- Safe dry-run mode to preview

### ✅ Well-Documented
- Complete migration guide included
- Troubleshooting section for common issues

## What Stays the Same

- Directory structure remains identical
- No changes to logic or algorithms
- All imports remain the same
- Data files don't need modification
- PyTorch model files (.pth) compatible across platforms

## Testing Strategy

After migration, verify:

1. **Imports work**: `python3 -c "from astro_compass.envs.TwoBodyRendezvous_Env import TwoBodyRendezvous_Env"`
2. **Unit tests pass**: `python3 -m pytest tests/`
3. **Data generation runs**: `python3 src/scripts/datagen_Hamiltonian_TBR_controller.py`
4. **Paths resolve correctly**: All data files can be read/written

## Additional Considerations

### Line Endings
Windows uses CRLF (`\r\n`), Linux uses LF (`\n`):
```bash
# Optional but recommended:
find . -name "*.py" -exec dos2unix {} \;
```

### File Permissions
```bash
# Make scripts executable:
chmod +x src/scripts/*.py
```

### PyTorch Installation
Check [pytorch.org](https://pytorch.org/) for Linux-specific installation:
```bash
# Example for CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Support

For detailed information, see:
- **Migration Guide**: `LINUX_MIGRATION_GUIDE.md`
- **Script Help**: `python migrate_to_linux.py --help` (after implementing)

## Safety

- ✅ Dry-run mode available
- ✅ Original files preserved in git
- ✅ Easy to revert with `git checkout .`
- ✅ All changes are deterministic and reversible

## Next Steps

1. **Review this document** to understand the changes
2. **Run the migration script** with `--dry-run` first
3. **Apply the changes** by running without --dry-run
4. **Commit to git** before transferring to Linux
5. **Follow the migration guide** on your Linux system

---

**Note**: This migration preserves all functionality while making the codebase Linux-compatible. The `os.path.join()` approach works correctly on both Windows and Linux, so you can still develop on Windows after these changes if needed.

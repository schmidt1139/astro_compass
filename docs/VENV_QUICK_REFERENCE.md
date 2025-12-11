# Virtual Environment Quick Reference - Shared Linux Systems

## Why Virtual Environments on Shared Systems?

**Problem:** Multiple users, different projects, conflicting dependencies
**Solution:** Each user/project gets isolated Python environment

## Your Dependencies Stay Isolated ✅

```
System Python (shared by all users)
├── minimal packages
└── READ-ONLY for regular users

Your Virtual Environment (only yours!)
├── gymnasium (your version)
├── torch (your version)  
├── stable-baselines3
├── matplotlib
└── ALL your packages - isolated from others
```

## Setup (One Time)

```bash
cd ~/astro_compass
python3 -m venv venv          # Creates isolated environment
source venv/bin/activate      # Enter your environment
pip install -e .              # Install YOUR packages
```

## Daily Workflow

```bash
# Login to server
ssh user@shared-server

# Navigate to project
cd ~/astro_compass

# ALWAYS activate before working
source venv/bin/activate
# Prompt changes: (venv) user@server:~/astro_compass$

# Now run your scripts - uses YOUR packages
python src/scripts/datagen_Hamiltonian_TBR_controller.py

# When done
deactivate
```

## Key Commands

| Command | Purpose |
|---------|---------|
| `source venv/bin/activate` | Enter your isolated environment |
| `deactivate` | Exit back to system Python |
| `which python` | Check which Python you're using |
| `pip list` | See packages in current environment |
| `pip install <package>` | Install to YOUR environment (not system) |
| `pip freeze > requirements.txt` | Save your package list |

## Verify Isolation

```bash
# Check you're using YOUR Python:
which python
# ✅ Should show: /home/youruser/astro_compass/venv/bin/python
# ❌ NOT: /usr/bin/python3

# Check where packages are installed:
pip show torch
# ✅ Location: .../astro_compass/venv/lib/python3.X/site-packages
# ❌ NOT: /usr/lib/python3.X/site-packages
```

## Common Mistakes

### ❌ Forgetting to Activate
```bash
# WITHOUT activating venv:
python script.py
# ERROR: ModuleNotFoundError: No module named 'gymnasium'
# (Uses system Python which doesn't have your packages)

# ✅ CORRECT:
source venv/bin/activate
python script.py
# Works! (Uses venv Python with your packages)
```

### ❌ Using System pip
```bash
# DON'T do this (tries to install system-wide, will fail):
pip3 install torch  # Without venv activated

# ✅ DO this:
source venv/bin/activate
pip install torch   # Installs to YOUR venv only
```

## How Isolation Works

### File Structure
```
~/astro_compass/
├── venv/                          # YOUR virtual environment
│   ├── bin/
│   │   ├── python                 # Your Python (symlink)
│   │   ├── pip                    # Your pip
│   │   └── activate               # Activation script
│   └── lib/python3.X/site-packages/  # YOUR packages live here
│       ├── gymnasium/             # Only in YOUR env
│       ├── torch/                 # Only in YOUR env
│       └── ...
├── src/                           # Your code (normal files)
├── data/                          # Your data (normal files)
└── tests/                         # Your tests (normal files)
```

### What `source venv/bin/activate` Does
1. Temporarily modifies `$PATH` to use `venv/bin/python` first
2. Sets `$VIRTUAL_ENV` environment variable
3. Changes prompt to show `(venv)`
4. **ONLY affects current terminal session**

### What It Doesn't Do
- ❌ Doesn't modify system Python
- ❌ Doesn't affect other users
- ❌ Doesn't persist after you close terminal (need to reactivate)
- ❌ Doesn't isolate data files (just Python packages)

## Multiple Users on Same Server

**User A:**
```bash
# User A's environment
~/astro_compass/venv/
└── torch 2.0.0  # Latest version
```

**User B:**
```bash  
# User B's environment (completely separate!)
~/astro_compass/venv/
└── torch 1.9.0  # Older version for compatibility
```

**No conflicts!** Each user's venv is completely isolated.

## Troubleshooting

### "Command not found" errors
```bash
# Problem: Can't find pytest, python, etc.
# Solution: Activate your venv
source venv/bin/activate
```

### "ModuleNotFoundError"
```bash
# Problem: Can't import gymnasium, torch, etc.
# Cause 1: Forgot to activate venv
source venv/bin/activate

# Cause 2: Package not installed in venv
pip install gymnasium torch
```

### "Permission denied" when installing packages
```bash
# Problem: Trying to install to system Python
# Solution: Make sure venv is activated first
source venv/bin/activate
which python  # Verify it shows venv path
pip install <package>
```

## Advanced: Automation

### Add to ~/.bashrc for convenience
```bash
# Add this to your ~/.bashrc:
alias activate_astro='cd ~/astro_compass && source venv/bin/activate'

# Now just type:
activate_astro
# Automatically goes to project and activates venv!
```

### Create a startup script
```bash
#!/bin/bash
# File: ~/astro_compass/start.sh
cd ~/astro_compass
source venv/bin/activate
echo "✓ astro_compass environment activated"
echo "Python: $(which python)"
echo "Packages: $(pip list | wc -l) installed"
```

```bash
chmod +x ~/astro_compass/start.sh
./start.sh  # One command to setup everything
```

## Summary

**Virtual Environment = Your Private Python Playground**

- ✅ Install any packages you want
- ✅ Any versions you need
- ✅ Won't affect other users
- ✅ Won't affect system Python
- ✅ Easy to delete and recreate
- ✅ Portable via requirements.txt

**Golden Rule:** Always activate before working!
```bash
source venv/bin/activate  # Every. Single. Time.
```

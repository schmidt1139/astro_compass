#!/usr/bin/env python3
"""
Migration script to convert Windows-specific paths to cross-platform paths.
This script updates all hardcoded Windows backslash paths to use os.path.join()
for compatibility with Linux environments.

Usage:
    python migrate_to_linux.py [--dry-run]

Options:
    --dry-run    Show what would be changed without modifying files
"""

import os
import re
import sys
from pathlib import Path

# Files to update with their specific path replacements
# Many files use os.path.join with backslashes inside - we'll fix those with regex
PATH_REPLACEMENTS = {
    "src/scripts/datagen_Hamiltonian_TBR_controller.py": [
        ('"data\\z_script_output\\training_TBR_circular\\"', 
         'os.path.join("data", "z_script_output", "training_TBR_circular")'),
    ],
    "src/scripts/ingest_ephem_and_plot.py": [
        ('"data\\test_data\\"', 
         'os.path.join("data", "test_data")'),
    ],
    "src/scripts/SAC_seeded_training.py": [
        ('"data\\\\neural_networks\\\\"', 
         'os.path.join("data", "neural_networks", "")'),
        ('"data\\\\training_ephems\\\\test_set_bang_bang\\\\"', 
         'os.path.join("data", "training_ephems", "test_set_bang_bang", "")'),
        ('"data\\\\script_output\\\\SAC_seeded_training_"', 
         'os.path.join("data", "script_output", "SAC_seeded_training_"'),
        ('+ "\\\\"', 
         '+ "", "")'),
    ],
    "src/scripts/train_alpha_network.py": [
        ('"..\\\\data\\\\training_ephems\\\\test_set_bang_bang_subset\\\\"', 
         'os.path.join("..", "data", "training_ephems", "test_set_bang_bang_subset", "")'),
        ('"..\\\\data\\\\plots\\\\"', 
         'os.path.join("..", "data", "plots", "")'),
        ('"..\\\\data\\\\neural_networks\\\\"', 
         'os.path.join("..", "data", "neural_networks", "")'),
    ],
    "src/scripts/train_neural_network.py": [
        ('"..\\\\data\\\\training_ephems\\\\test_set_bang_bang_subset\\\\"', 
         'os.path.join("..", "data", "training_ephems", "test_set_bang_bang_subset", "")'),
        ('"..\\\\data\\\\plots\\\\"', 
         'os.path.join("..", "data", "plots", "")'),
        ('"..\\\\data\\\\neural_networks\\\\"', 
         'os.path.join("..", "data", "neural_networks", "")'),
    ],
    "src/scripts/solve_two_body_transfer_and_write_ephem_script.py": [
        ('"data\\\\training_ephems\\\\solution_ephemeris.txt"', 
         'os.path.join("data", "training_ephems", "solution_ephemeris.txt")'),
    ],
    "src/scripts/generate_nn_training_data_parallel.py": [
        ('"\\\\data\\\\training_ephems\\\\test_set3"', 
         'os.path.join("data", "training_ephems", "test_set3")'),
    ],
    "tests/test_write_ephemeris.py": [
        ('"..\\\\..\\\\data\\\\training_ephems\\\\"', 
         'os.path.join("..", "..", "data", "training_ephems", "")'),
    ],
    "tests/test_TBR_env.py": [
        ('"data\\\\test_data\\\\test_TBR\\\\test_traj_"', 
         'os.path.join("data", "test_data", "test_TBR", "test_traj_"'),
        ('"data\\\\test_data\\\\test_TBR\\\\test_traj_ephemeris_"', 
         'os.path.join("data", "test_data", "test_TBR", "test_traj_ephemeris_"'),
    ],
    "tests/test_seeded_SAC_training.py": [
        ('"data\\\\neural_networks\\\\"', 
         'os.path.join("data", "neural_networks", "")'),
        ('"data\\\\training_ephems\\\\test_set_bang_bang\\\\"', 
         'os.path.join("data", "training_ephems", "test_set_bang_bang", "")'),
        ('"data\\\\test_data\\\\test_seeded_SAC_training\\\\"', 
         'os.path.join("data", "test_data", "test_seeded_SAC_training", "")'),
    ],
    "tests/test_SAC_training_TBR.py": [
        ('"data\\\\neural_networks\\\\"', 
         'os.path.join("data", "neural_networks", "")'),
        ('"data\\\\test_data\\\\test_SAC_training_TBR\\\\"', 
         'os.path.join("data", "test_data", "test_SAC_training_TBR", "")'),
    ],
    "tests/test_SAC_training.py": [
        ('"data\\\\neural_networks\\\\"', 
         'os.path.join("data", "neural_networks", "")'),
        ('"data\\\\test_data\\\\test_SAC_training\\\\"', 
         'os.path.join("data", "test_data", "test_SAC_training", "")'),
    ],
    "tests/test_random_env_reset.py": [
        ('"..\\\\..\\\\data\\\\test_data\\\\test_random_TBT_transfer_report.csv"', 
         'os.path.join("..", "..", "data", "test_data", "test_random_TBT_transfer_report.csv")'),
    ],
    "tests/test_Hamiltonian_TBR_controller.py": [
        ('"data\\\\test_data\\\\test_TBR_hamiltonian\\\\"', 
         'os.path.join("data", "test_data", "test_TBR_hamiltonian", "")'),
    ],
    "tests/test_Hamiltonians.py": [
        ('"data\\\\test_data\\\\test_hamiltonians\\\\test_H_ephem.txt"', 
         'os.path.join("data", "test_data", "test_hamiltonians", "test_H_ephem.txt")'),
        ('"data\\\\test_data\\\\test_hamiltonians\\\\test_H_ephem_truth.txt"', 
         'os.path.join("data", "test_data", "test_hamiltonians", "test_H_ephem_truth.txt")'),
    ],
    "tests/test_eom_and_propagation.py": [
        ('"..\\\\..\\\\data\\\\plots\\\\states_nd.pdf"', 
         'os.path.join("..", "..", "data", "plots", "states_nd.pdf")'),
        ('"..\\\\..\\\\data\\\\plots\\\\costates_nd.pdf"', 
         'os.path.join("..", "..", "data", "plots", "costates_nd.pdf")'),
    ],
    "tests/test_env_step_with_nn_action.py": [
        ('"data\\\\test_data\\\\test_env_step_with_nn_action\\\\"', 
         'os.path.join("data", "test_data", "test_env_step_with_nn_action", "")'),
    ],
    "tests/test_env_step_with_action.py": [
        ('"data\\\\test_data\\\\test_env_step_with_action\\\\"', 
         'os.path.join("data", "test_data", "test_env_step_with_action", "")'),
    ],
    "tests/test_env_step_no_action.py": [
        ('"data\\\\test_data\\\\test_env_step_no_action\\\\"', 
         'os.path.join("data", "test_data", "test_env_step_no_action", "")'),
    ],
}


def ensure_os_import(content: str, filepath: str) -> str:
    """Ensure 'import os' exists in the file if os.path.join is used."""
    if 'os.path.join' not in content:
        return content
    
    if re.search(r'^import os\s*$', content, re.MULTILINE):
        return content  # Already has 'import os'
    
    if re.search(r'^from os import', content, re.MULTILINE):
        return content  # Has os imports already
    
    # Add import os at the top after any shebang/encoding lines
    lines = content.split('\n')
    insert_pos = 0
    
    for i, line in enumerate(lines):
        if line.startswith('#!') or 'coding:' in line or 'coding=' in line:
            insert_pos = i + 1
        elif line.strip() and not line.startswith('#'):
            insert_pos = i
            break
    
    # Find where imports start
    for i in range(insert_pos, len(lines)):
        if lines[i].startswith('import ') or lines[i].startswith('from '):
            insert_pos = i
            break
    
    lines.insert(insert_pos, 'import os')
    print(f"  ✓ Added 'import os' to {filepath}")
    return '\n'.join(lines)


def fix_paths_in_os_join(content: str) -> tuple[str, int]:
    """
    Fix Windows backslashes inside os.path.join() calls.
    
    Converts: os.path.join(os.getcwd(), "data\\test\\path\\")
    To:       os.path.join(os.getcwd(), "data", "test", "path")
    
    Returns:
        Tuple of (modified_content, number_of_fixes)
    """
    fixes = 0
    
    # Pattern to match os.path.join with backslashes in string arguments
    # Matches: "data\\path\\to\\file"
    pattern = r'"([^"]*\\[^"]*)"'
    
    def replace_backslashes(match):
        nonlocal fixes
        path_str = match.group(1)
        
        # Split by backslash and create separate arguments
        parts = [p for p in path_str.split('\\') if p]  # Remove empty strings
        
        # If it ends with a backslash (directory), don't add trailing separator
        # Just use separate parts
        if len(parts) > 1:
            fixes += 1
            return '"' + '", "'.join(parts) + '"'
        elif len(parts) == 1 and '\\' in match.group(0):
            # Single part but had backslash (e.g., "data\\")
            fixes += 1
            return '"' + parts[0] + '"'
        else:
            return match.group(0)
    
    # Only replace within os.path.join calls
    lines = content.split('\n')
    modified_lines = []
    
    for line in lines:
        if 'os.path.join' in line and '\\' in line:
            original_line = line
            line = re.sub(pattern, replace_backslashes, line)
            if line != original_line:
                modified_lines.append((original_line, line))
        
    # Apply all changes
    for original, modified in modified_lines:
        content = content.replace(original, modified, 1)
    
    return content, fixes


def migrate_file(filepath: str, replacements: list, dry_run: bool = False) -> bool:
    """
    Migrate a single file with the given replacements.
    
    Args:
        filepath: Path to file to migrate
        replacements: List of (old_pattern, new_pattern) tuples
        dry_run: If True, only show what would change
    
    Returns:
        True if changes were made/would be made, False otherwise
    """
    if not os.path.exists(filepath):
        print(f"  ⚠ File not found: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes_made = False
    
    # First, apply specific replacements
    for old_pattern, new_pattern in replacements:
        # Handle both raw string patterns and regex patterns
        old_str = old_pattern.replace('\\\\', '\\')
        if old_str in content:
            content = content.replace(old_str, new_pattern)
            changes_made = True
            print(f"  ✓ Replaced: {old_str[:60]}...")
    
    # Second, fix any remaining backslashes in os.path.join calls
    content, path_fixes = fix_paths_in_os_join(content)
    if path_fixes > 0:
        changes_made = True
        print(f"  ✓ Fixed {path_fixes} path(s) in os.path.join() calls")
    
    # Third, fix standalone paths with backslashes (not in os.path.join)
    # Look for string literals with backslashes
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # Skip lines that already have os.path.join
        if 'os.path.join' in line:
            continue
        
        # Find string literals with backslashes
        string_pattern = r'"([^"]*\\[^"]*)"'
        matches = re.findall(string_pattern, line)
        
        for match in matches:
            # Only convert if it looks like a path (has data, .., src, tests, etc.)
            path_indicators = ['data\\', '..\\', 'src\\', 'tests\\', '.txt', '.csv', '.pdf', '.png']
            if '\\' in match and any(indicator in match for indicator in path_indicators):
                # Skip if it's just escape sequences like \n
                if match.count('\\') == match.count('\\n') + match.count('\\t') + match.count('\\r'):
                    continue
                    
                parts = [p for p in match.split('\\') if p]
                if len(parts) > 1:
                    old_str = f'"{match}"'
                    new_str = 'os.path.join(' + ', '.join(f'"{p}"' for p in parts) + ')'
                    if old_str in content:
                        content = content.replace(old_str, new_str)
                        changes_made = True
                        print(f"  ✓ Converted standalone path: {old_str[:60]}...")
    
    if changes_made:
        content = ensure_os_import(content, filepath)
        
        if not dry_run:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Updated: {filepath}")
        else:
            print(f"  [DRY RUN] Would update: {filepath}")
    
    return changes_made


def main():
    """Main migration function."""
    dry_run = '--dry-run' in sys.argv
    
    print("=" * 70)
    print("astro_compass: Linux Migration Script")
    print("=" * 70)
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - No files will be modified\n")
    else:
        print("\n⚙️  MIGRATION MODE - Files will be updated\n")
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    total_files = 0
    updated_files = 0
    
    for rel_path, replacements in PATH_REPLACEMENTS.items():
        total_files += 1
        print(f"\n📄 Processing: {rel_path}")
        
        if migrate_file(rel_path, replacements, dry_run):
            updated_files += 1
    
    print("\n" + "=" * 70)
    print(f"✅ Migration {'simulation' if dry_run else 'complete'}!")
    print(f"   Files processed: {total_files}")
    print(f"   Files {'would be ' if dry_run else ''}updated: {updated_files}")
    print("=" * 70)
    
    if dry_run:
        print("\n💡 Run without --dry-run to apply changes:")
        print(f"   python {Path(__file__).name}")
    else:
        print("\n📋 Next steps for Linux migration:")
        print("   1. Commit these changes to git")
        print("   2. Transfer code to Linux environment")
        print("   3. Run: python3 -m venv venv")
        print("   4. Run: source venv/bin/activate")
        print("   5. Run: pip install -e .")
        print("   6. Run tests: python3 -m pytest tests/")
        print("\n   Optional: Convert line endings:")
        print("   find . -name '*.py' -exec dos2unix {} \\;")
    
    print()


if __name__ == "__main__":
    main()

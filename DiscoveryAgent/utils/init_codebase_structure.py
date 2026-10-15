"""init_structure.py

Creates directories and empty files from a PROJECT_STRUCTURE.txt-like file.

Usage:
    python init_structure.py /absolute/path/to/PROJECT_STRUCTURE.txt --datapath /some/base/path

The script will replace the token "{datapath}" in the structure file with the provided datapath value.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple


def parse_structure_lines(lines: List[str], datapath: str) -> Tuple[List[Path], List[Path]]:
    """Parse lines from a PROJECT_STRUCTURE-style file.

    Returns two lists: directories and files (both as Path objects).

    The function treats lines as a tree with prefixes and extracts paths. Lines
    that end with '/' are directories; others are files. The special token
    "{datapath}" will be replaced with the provided datapath string.
    """
    dirs: List[Path] = []
    files: List[Path] = []
    
    # Track the current path at each indentation level
    path_stack: List[str] = []
    prev_indent = -1

    for raw in lines:
        line = raw.rstrip("\n\r")
        if not line.strip():
            continue

        # Replace the token
        line = line.replace("{datapath}", datapath)

        # Calculate indentation level by counting leading characters before the path name
        # Tree characters are: ├──, └──, │, spaces, tabs
        original_line = line
        
        # Remove line number prefix if present (e.g., "     1|")
        if "|" in line:
            parts = line.split("|", 1)
            if parts[0].strip().isdigit():
                line = parts[1]
        
        # Count the indentation by finding where the actual path name starts
        # After removing tree drawing characters (including | which is sometimes used)
        indent_chars = 0
        for char in line:
            if char in " \t│├└─|":
                indent_chars += 1
            else:
                break
        
        # Determine indentation level (roughly every 4 chars is one level)
        indent_level = indent_chars // 4
        
        # Strip leading tree chars and whitespace
        cleaned = line.lstrip(" \t├└│─`|")

        # Remove inline comments starting with '#'
        if "#" in cleaned:
            cleaned = cleaned.split("#", 1)[0]

        cleaned = cleaned.rstrip()
        if not cleaned:
            continue

        # Detect trailing slash meaning directory
        is_dir = cleaned.endswith("/")
        # Remove trailing slash for Path creation
        path_component = cleaned[:-1] if is_dir else cleaned
        path_component = path_component.strip()
        
        if not path_component:
            continue

        # Update the path stack based on indentation
        if indent_level <= prev_indent:
            # Going back up or staying at same level, pop until we're at the right level
            while len(path_stack) > indent_level:
                path_stack.pop()
        
        # Build the full path
        if indent_level == 0:
            # Top level - this is the base path
            path_stack = [path_component]
            full_path = Path(path_component)
        else:
            # Nested - append to parent path
            if len(path_stack) > indent_level:
                path_stack = path_stack[:indent_level]
            path_stack.append(path_component)
            full_path = Path(*path_stack)
        
        prev_indent = indent_level

        # Add to appropriate list
        if is_dir:
            dirs.append(full_path)
        else:
            files.append(full_path)

    return dirs, files


def ensure_path(path: Path, create_as_dir: bool = True) -> None:
    """Create directory or empty file as needed.

    If create_as_dir is True, creates the directory (with parents). If False,
    ensures parent directories exist and touches the file.
    """
    if create_as_dir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        # Create an empty file if it doesn't exist
        path.touch(exist_ok=True)


def init_from_structure_file(structure_file: Path | str, datapath: str) -> str:
    """Read the structure file and create directories/files.

    The datapath parameter will replace the {datapath} token in the structure file.
    If the structure file starts with {datapath}/ as the root, that directory will
    be created along with all subdirectories and files.
    
    Returns:
        The original string content loaded from the structure file.
    """
    if not isinstance(structure_file, Path):
        structure_file = Path(structure_file)
        
    if not structure_file.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_file}")

    content = structure_file.read_text(encoding="utf-8")
    lines = content.splitlines()
    dirs, files = parse_structure_lines(lines, datapath)

    # Create directories first
    # Note: paths already include datapath if {datapath} token was in structure file
    for d in dirs:
        resolved = Path(str(d))
        ensure_path(resolved, create_as_dir=True)

    # Then create files
    for f in files:
        resolved = Path(str(f))
        ensure_path(resolved, create_as_dir=False)
    
    return content


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Init project structure from a PROJECT_STRUCTURE.txt-style file")
    parser.add_argument("structure_file", type=Path, help="Path to the structure file")
    parser.add_argument("--datapath", type=str, default="data", help="Replacement for {datapath} token")
    args = parser.parse_args(argv)

    try:
        init_from_structure_file(args.structure_file, args.datapath)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

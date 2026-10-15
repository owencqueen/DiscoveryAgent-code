"""
Data Directory Manager for Expanding Dataset Mode

This module provides utilities for tracking and describing a growing data directory,
detecting new files between rounds, and generating updated dataset descriptions.
"""

import os
from pathlib import Path
from typing import List, Dict, Set, Optional, Callable
from datetime import datetime
import hashlib


class DataDirectoryManager:
    """
    Manages an expanding data directory with change detection and description generation.
    
    This class tracks files in a data directory, detects additions between scans,
    and can generate updated dataset descriptions that reflect the current state
    of available data files.
    """
    
    def __init__(
        self,
        data_path: str,
        base_description: str = "",
        file_listing_placeholder: str = "{file_listing}",
        ignored_patterns: Optional[List[str]] = None
    ):
        """
        Initialize the DataDirectoryManager.
        
        Args:
            data_path: Path to the data directory to monitor
            base_description: Base dataset description template. Can contain a placeholder
                            for file listing (default: "{file_listing}")
            file_listing_placeholder: The placeholder string to replace with file listing
            ignored_patterns: List of glob patterns to ignore (e.g., ["*.tmp", "__pycache__"])
        """
        self.data_path = Path(data_path)
        self.base_description = base_description
        self.file_listing_placeholder = file_listing_placeholder
        self.ignored_patterns = ignored_patterns or ["__pycache__", "*.pyc", ".DS_Store", "*.tmp"]
        
        # Track file inventory across scans
        self._last_inventory: Set[str] = set()
        self._current_inventory: Set[str] = set()
        self._scan_count = 0
        
        # Track file metadata for more detailed change detection
        self._file_metadata: Dict[str, Dict] = {}
    
    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on patterns."""
        name = path.name
        for pattern in self.ignored_patterns:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif name == pattern:
                return True
        return False
    
    def get_file_inventory(self, include_metadata: bool = False) -> List[Dict]:
        """
        Scan the data directory and return a list of files with metadata.
        
        Args:
            include_metadata: If True, include size, modified_time, and type info
            
        Returns:
            List of dicts with file information:
            [{path, relative_path, name, size?, modified_time?, is_dir?}, ...]
        """
        if not self.data_path.exists():
            return []
        
        inventory = []
        
        for root, dirs, files in os.walk(self.data_path):
            root_path = Path(root)
            
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self._should_ignore(root_path / d)]
            
            for filename in files:
                file_path = root_path / filename
                
                if self._should_ignore(file_path):
                    continue
                
                relative_path = file_path.relative_to(self.data_path)
                
                file_info = {
                    'path': str(file_path),
                    'relative_path': str(relative_path),
                    'name': filename,
                    'is_dir': False
                }
                
                if include_metadata:
                    try:
                        stat = file_path.stat()
                        file_info['size'] = stat.st_size
                        file_info['modified_time'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                    except OSError:
                        file_info['size'] = 0
                        file_info['modified_time'] = None
                
                inventory.append(file_info)
        
        return sorted(inventory, key=lambda x: x['relative_path'])
    
    def scan(self) -> Set[str]:
        """
        Perform a scan of the data directory and update internal state.
        
        Returns:
            Set of relative paths for all current files
        """
        self._last_inventory = self._current_inventory.copy()
        
        inventory = self.get_file_inventory(include_metadata=True)
        self._current_inventory = {f['relative_path'] for f in inventory}
        self._file_metadata = {f['relative_path']: f for f in inventory}
        
        self._scan_count += 1
        return self._current_inventory
    
    def detect_new_files(self) -> List[str]:
        """
        Return files that were added since the last scan.
        
        Returns:
            List of relative paths for newly added files
        """
        if self._scan_count == 0:
            self.scan()
            return []  # First scan, no "new" files
        
        new_files = self._current_inventory - self._last_inventory
        return sorted(list(new_files))
    
    def detect_removed_files(self) -> List[str]:
        """
        Return files that were removed since the last scan.
        
        Returns:
            List of relative paths for removed files
        """
        removed_files = self._last_inventory - self._current_inventory
        return sorted(list(removed_files))
    
    def generate_file_listing(self, max_files: int = 100, include_sizes: bool = True) -> str:
        """
        Generate a formatted markdown listing of files in the data directory.
        
        Args:
            max_files: Maximum number of files to list (to avoid huge descriptions)
            include_sizes: Whether to include file sizes
            
        Returns:
            Formatted markdown string listing the files
        """
        if not self.data_path.exists():
            return "*Data directory does not exist yet*"
        
        inventory = self.get_file_inventory(include_metadata=include_sizes)
        
        if not inventory:
            return "*No data files found*"
        
        lines = []
        truncated = len(inventory) > max_files
        
        for file_info in inventory[:max_files]:
            rel_path = file_info['relative_path']
            if include_sizes and 'size' in file_info:
                size = file_info['size']
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                elif size < 1024 * 1024 * 1024:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{size / (1024 * 1024 * 1024):.2f} GB"
                lines.append(f"- `{rel_path}` ({size_str})")
            else:
                lines.append(f"- `{rel_path}`")
        
        if truncated:
            remaining = len(inventory) - max_files
            lines.append(f"- *... and {remaining} more files*")
        
        return "\n".join(lines)
    
    def generate_tree_listing(self, max_depth: int = 3) -> str:
        """
        Generate a tree-style listing of the data directory structure.
        
        Args:
            max_depth: Maximum directory depth to show
            
        Returns:
            Tree-formatted string showing directory structure
        """
        if not self.data_path.exists():
            return "data/ (not created yet)"
        
        lines = ["data/"]
        
        def add_tree_level(path: Path, prefix: str, depth: int):
            if depth > max_depth:
                return
            
            try:
                entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            except PermissionError:
                return
            
            entries = [e for e in entries if not self._should_ignore(e)]
            
            for i, entry in enumerate(entries):
                is_last = (i == len(entries) - 1)
                connector = "└── " if is_last else "├── "
                
                if entry.is_dir():
                    lines.append(f"{prefix}{connector}{entry.name}/")
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    add_tree_level(entry, new_prefix, depth + 1)
                else:
                    lines.append(f"{prefix}{connector}{entry.name}")
        
        add_tree_level(self.data_path, "", 1)
        return "\n".join(lines)
    
    def update_dataset_description(
        self,
        template: Optional[str] = None,
        include_tree: bool = False,
        max_files: int = 100
    ) -> str:
        """
        Generate an updated dataset description with current file inventory.
        
        Args:
            template: Description template (uses base_description if None).
                     Should contain the file_listing_placeholder.
            include_tree: If True, include a tree-style directory listing
            max_files: Maximum files to list
            
        Returns:
            Updated dataset description string
        """
        template = template or self.base_description
        
        # Generate file listing
        file_listing = self.generate_file_listing(max_files=max_files)
        
        if include_tree:
            tree = self.generate_tree_listing()
            file_listing = f"**Directory Structure:**\n```\n{tree}\n```\n\n**Files:**\n{file_listing}"
        
        # Replace placeholder if present
        if self.file_listing_placeholder in template:
            return template.replace(self.file_listing_placeholder, file_listing)
        else:
            # Append file listing at the end if no placeholder
            return f"{template}\n\n## Available Data Files\n\n{file_listing}"
    
    def get_changes_summary(self) -> str:
        """
        Get a summary of changes since the last scan.
        
        Returns:
            Markdown-formatted summary of file changes
        """
        new_files = self.detect_new_files()
        removed_files = self.detect_removed_files()
        
        if not new_files and not removed_files:
            return ""
        
        lines = ["## Data Directory Changes\n"]
        
        if new_files:
            lines.append("**New files added:**")
            for f in new_files[:20]:  # Limit to 20 files
                lines.append(f"- `{f}`")
            if len(new_files) > 20:
                lines.append(f"- *... and {len(new_files) - 20} more*")
            lines.append("")
        
        if removed_files:
            lines.append("**Files removed:**")
            for f in removed_files[:20]:
                lines.append(f"- `{f}`")
            if len(removed_files) > 20:
                lines.append(f"- *... and {len(removed_files) - 20} more*")
        
        return "\n".join(lines)
    
    def ensure_data_directory_exists(self) -> bool:
        """
        Ensure the data directory exists, creating it if necessary.
        
        Returns:
            True if directory exists or was created, False on error
        """
        try:
            self.data_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False



"""
Git Version Control Utility

This module provides a simple API for git version control operations
to track changes in the DiscoveryAgent workflow working directory.
"""

import os
import datetime
import subprocess
from pathlib import Path
from typing import Optional


class GitVersionControl:
    """
    Simple git version control wrapper for tracking workflow changes.

    Provides methods to initialize repositories, stage files, and create commits
    during the DiscoveryAgent workflow execution.
    """

    GIT_COMMAND_TIMEOUT_SECONDS = 300
    STALE_INDEX_LOCK_SECONDS = 120

    def __init__(self, working_directory: str, logger=None):
        """
        Initialize the GitVersionControl instance.

        Args:
            working_directory (str): Path to the working directory to track
            logger (Logger, optional): Logger instance for logging git operations
        """
        self.working_directory = Path(working_directory)
        self.logger = logger
        # Default: skip auto-staging files larger than 50MB.
        # Can be overridden with DISCOVERYAGENT_GIT_MAX_FILE_SIZE_MB.
        try:
            max_size_mb = float(os.environ.get("DISCOVERYAGENT_GIT_MAX_FILE_SIZE_MB", "50"))
        except ValueError:
            max_size_mb = 50.0
        self.max_stage_file_size_bytes = int(max_size_mb * 1024 * 1024)

        if not self.working_directory.exists():
            raise ValueError(f"Working directory does not exist: {working_directory}")

    def _run_git_command(self, command: list, error_message: str) -> tuple[bool, str]:
        """
        Run a git command and return success status and output.

        Args:
            command (list): Git command as list of strings
            error_message (str): Error message to log on failure

        Returns:
            tuple[bool, str]: (success, output) - success is True if command succeeded,
                             output contains stdout or error message
        """
        try:
            result = subprocess.run(
                command,
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                timeout=self.GIT_COMMAND_TIMEOUT_SECONDS
            )

            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                error_output = result.stderr.strip() or result.stdout.strip()
                if self.logger:
                    self.logger.log_error(f"{error_message}: {error_output}")
                return False, error_output

        except subprocess.TimeoutExpired:
            error = (
                f"{error_message}: Command timed out after "
                f"{self.GIT_COMMAND_TIMEOUT_SECONDS} seconds"
            )
            if self.logger:
                self.logger.log_error(error)
            return False, error
        except Exception as e:
            error = f"{error_message}: {str(e)}"
            if self.logger:
                self.logger.log_error(error)
            return False, error

    @staticmethod
    def _looks_like_index_lock_error(output: str) -> bool:
        """Return True if git output indicates index.lock contention."""
        lowered = output.lower()
        return "index.lock" in lowered and "file exists" in lowered

    def _index_lock_path(self) -> Path:
        """Return the repository index.lock path."""
        return self.working_directory / ".git" / "index.lock"

    def _is_index_lock_held(self, lock_path: Path) -> bool:
        """
        Best-effort check for an actively held lock file.

        Uses `lsof` when available. If lsof is unavailable, we avoid deleting
        very fresh locks and treat older locks as stale.
        """
        if not lock_path.exists():
            return False

        try:
            result = subprocess.run(
                ["lsof", str(lock_path)],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                timeout=10
            )
            # lsof returns 0 if at least one process has the file open.
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            # If we can't check process ownership, be conservative for fresh locks.
            try:
                age_seconds = (
                    datetime.datetime.now().timestamp() - lock_path.stat().st_mtime
                )
                return age_seconds < self.STALE_INDEX_LOCK_SECONDS
            except Exception:
                return True

    def _clear_stale_index_lock(self) -> bool:
        """
        Remove .git/index.lock if it appears stale and unheld.

        Returns:
            bool: True if lock was removed, False otherwise.
        """
        lock_path = self._index_lock_path()
        if not lock_path.exists():
            return False

        if self._is_index_lock_held(lock_path):
            if self.logger:
                self.logger.log_action(
                    "git_index_lock_active",
                    f"index.lock appears active, not removing: {lock_path}"
                )
            return False

        try:
            lock_path.unlink()
            if self.logger:
                self.logger.log_action(
                    "git_index_lock_removed",
                    f"Removed stale index.lock: {lock_path}"
                )
            return True
        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    f"Failed to remove stale index.lock at {lock_path}: {str(e)}"
                )
            return False

    def _run_git_with_stale_lock_recovery(
        self,
        command: list,
        error_message: str
    ) -> tuple[bool, str]:
        """
        Run a git command and retry once after stale lock cleanup.
        """
        success, output = self._run_git_command(command, error_message)
        if success:
            return True, output

        if self._looks_like_index_lock_error(output) and self._clear_stale_index_lock():
            if self.logger:
                self.logger.log_action(
                    "git_retry_after_lock_cleanup",
                    f"Retrying command after stale index.lock cleanup: {' '.join(command)}"
                )
            return self._run_git_command(command, error_message)

        return False, output

    def _list_staged_paths(self) -> list[str]:
        """
        Return all currently staged paths as POSIX-style relative paths.
        """
        success, output = self._run_git_command(
            ["git", "diff", "--cached", "--name-only", "-z"],
            "Failed to list staged files"
        )
        if not success:
            return []
        if not output:
            return []
        return [p for p in output.split("\0") if p]

    def _unstage_path(self, relative_path: str) -> bool:
        """
        Unstage a single path while preserving working tree changes.
        """
        # Prefer modern command first.
        try:
            result = subprocess.run(
                ["git", "restore", "--staged", "--", relative_path],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

        # Fallback for older Git versions.
        try:
            result = subprocess.run(
                ["git", "reset", "HEAD", "--", relative_path],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except Exception:
            return False

    def _filter_large_files_from_index(self) -> tuple[int, list[str]]:
        """
        Unstage staged files above the configured max size threshold.

        Returns:
            tuple[int, list[str]]: (count_skipped, skipped_paths)
        """
        staged_paths = self._list_staged_paths()
        if not staged_paths:
            return 0, []

        skipped_paths: list[str] = []
        for relative_path in staged_paths:
            abs_path = self.working_directory / relative_path

            # If file does not exist in working tree (e.g. deletions), keep staged.
            if not abs_path.exists() or not abs_path.is_file():
                continue

            try:
                if abs_path.stat().st_size > self.max_stage_file_size_bytes:
                    if self._unstage_path(relative_path):
                        skipped_paths.append(relative_path)
                    elif self.logger:
                        self.logger.log_error(
                            f"Failed to unstage oversized file: {relative_path}"
                        )
            except Exception as e:
                if self.logger:
                    self.logger.log_error(
                        f"Failed size check for staged file {relative_path}: {str(e)}"
                    )

        return len(skipped_paths), skipped_paths

    def init_repo(self) -> bool:
        """
        Initialize a git repository in the working directory.

        Returns:
            bool: True if successful, False otherwise
        """
        # Check if already a git repo
        git_dir = self.working_directory / ".git"
        if git_dir.exists():
            if self.logger:
                self.logger.log_action("git_init_skipped",
                    f"Git repository already exists in {self.working_directory}")
            return True

        success, output = self._run_git_command(
            ["git", "init"],
            "Failed to initialize git repository"
        )

        if success:
            if self.logger:
                self.logger.log_action("git_init",
                    f"Initialized git repository in {self.working_directory}")
            print(f"🔧 Initialized git repository in {self.working_directory}")

        return success

    def add_all(self) -> bool:
        """
        Stage changes in the working directory and unstage oversized files.

        Returns:
            bool: True if successful, False otherwise
        """
        success, output = self._run_git_with_stale_lock_recovery(
            ["git", "add", "-A", "."],
            "Failed to stage changes"
        )

        if not success:
            return False

        skipped_count, skipped_paths = self._filter_large_files_from_index()

        if self.logger:
            if skipped_count > 0:
                self.logger.log_action(
                    "git_add_filtered",
                    (
                        f"Staged changes with size filter "
                        f"({self.max_stage_file_size_bytes // (1024 * 1024)}MB max). "
                        f"Skipped {skipped_count} oversized file(s): {skipped_paths[:10]}"
                    )
                )
            else:
                self.logger.log_action(
                    "git_add",
                    (
                        f"Staged all changes "
                        f"({self.max_stage_file_size_bytes // (1024 * 1024)}MB max file size)"
                    )
                )

        return True

    def commit(self, message: str, round_num: Optional[int] = None) -> bool:
        """
        Create a git commit with the given message, optionally including round number and timestamp.

        Args:
            message (str): Commit message
            round_num (Optional[int]): Round number to include in commit message

        Returns:
            bool: True if successful, False otherwise
        """
        
        if not message:
            message = "Automated commit"

        # Build enhanced commit message with round number and timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if round_num is not None:
            full_message = f"[Round {round_num}] {message} | {timestamp}"
        else:
            full_message = f"{message} | {timestamp}"

        # Configure git user if not already configured
        self._ensure_git_config()

        success, output = self._run_git_with_stale_lock_recovery(
            ["git", "commit", "-m", full_message],
            "Failed to create commit"
        )

        if success:
            if self.logger:
                self.logger.log_action("git_commit", f"Created commit: {message}")
            print(f"📝 Git commit: {message}")
        elif "nothing to commit" in output.lower():
            # Not an error - just nothing to commit
            if self.logger:
                self.logger.log_action("git_commit_skipped", "No changes to commit")
            return True

        return success

    def _ensure_git_config(self):
        """Ensure git user.name and user.email are configured."""
        # Check if user.name is set
        result = subprocess.run(
            ["git", "config", "user.name"],
            cwd=self.working_directory,
            capture_output=True,
            text=True
        )

        if result.returncode != 0 or not result.stdout.strip():
            # Set default user.name
            subprocess.run(
                ["git", "config", "user.name", "DiscoveryAgent Workflow"],
                cwd=self.working_directory,
                capture_output=True
            )

        # Check if user.email is set
        result = subprocess.run(
            ["git", "config", "user.email"],
            cwd=self.working_directory,
            capture_output=True,
            text=True
        )

        if result.returncode != 0 or not result.stdout.strip():
            # Set default user.email
            subprocess.run(
                ["git", "config", "user.email", "discoveryagent@example.com"],
                cwd=self.working_directory,
                capture_output=True
            )

    def get_status(self) -> Optional[str]:
        """
        Get the current git status.

        Returns:
            Optional[str]: Git status output, or None if failed
        """
        success, output = self._run_git_command(
            ["git", "status", "--short"],
            "Failed to get git status"
        )

        return output if success else None

    def get_commit_count(self) -> int:
        """
        Get the number of commits in the repository.

        Returns:
            int: Number of commits, or 0 if failed
        """
        success, output = self._run_git_command(
            ["git", "rev-list", "--count", "HEAD"],
            "Failed to count commits"
        )

        if success:
            try:
                return int(output)
            except ValueError:
                return 0
        return 0

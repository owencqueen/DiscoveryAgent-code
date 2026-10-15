"""
Allow running acs_access as a module.

Usage:
    python -m acs_agent.acs_access --help
    python -m acs_agent.acs_access download-table B01001 --state Michigan
"""

from .cli import main

if __name__ == "__main__":
    main()


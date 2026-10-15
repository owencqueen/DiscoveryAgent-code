# Expanding Dataset Mode

This guide explains the **Expanding Dataset Mode** feature, which allows the data directory to grow during workflow execution and automatically updates the dataset description to reflect new files.

## Overview

In standard mode, the `data/` directory is treated as read-only and the dataset description remains static throughout the workflow. **Expanding Dataset Mode** changes this behavior:

- The `data/` directory becomes a **writable, growing directory**
- Agents can **add new files** to `data/` (e.g., downloaded datasets, processed outputs)
- The **dataset description is refreshed after each query** to reflect newly added files
- All agents see an up-to-date view of available data

This is useful for workflows where:
- Data is downloaded or fetched during analysis
- Intermediate processed results should be accessible to future queries
- The dataset evolves as the research progresses

---

## How to Enable

### Command Line

```bash
.venv/bin/python -m scripts.run \
    --working_directory /path/to/your/dataset \
    --analysis_name my_analysis \
    --expanding-dataset
```

### With Custom Data Directory

```bash
.venv/bin/python -m scripts.run \
    --working_directory /path/to/your/dataset \
    --analysis_name my_analysis \
    --expanding-dataset \
    --expanding-data-directory /path/to/custom/data/location
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--expanding-dataset` | flag | `False` | Enable expanding dataset mode |
| `--expanding-data-directory` | string | `data/downloads` | Path to data directory to monitor |

---

## Components

### 1. Codebase Structure Prompt (`DiscoveryAgent/prompts/execution_agent.py`)

Rule 7 of `CODEBASE_STRUCTURE_PROMPT` tells agents they can add files to `data/`:

```
7. **Data Directory Rules (EXPANDING DATASET)**:
   The `data/` directory may contain a growing dataset. You are permitted to:
   - ✅ ADD new data files to `data/`
   - ✅ CREATE subdirectories within `data/`
   - ✅ READ all existing data files
   
   You MUST NOT:
   - ❌ DELETE or MODIFY existing data files
   - ❌ OVERWRITE existing files in `data/`
```

> **Note**: This rule is always present (not conditional), but agents are still instructed not to delete/modify existing files.

### 2. DiscoveryAgentConfig (`DiscoveryAgent/hypothesis_brancher.py`)

Expanding-dataset fields of `DiscoveryAgentConfig`:

```python
@dataclass
class DiscoveryAgentConfig:
    # ... existing fields ...
    
    # Expanding Dataset Mode settings
    expanding_dataset_mode: bool = False
    expanding_data_directory: Optional[str] = None
    file_listing_placeholder: Optional[str] = "{file_listing}"
```

### 3. DataDirectoryManager (`DiscoveryAgent/utils/data_directory_manager.py`)

Utility class that:
- Scans the data directory for files
- Tracks file additions between scans
- Generates formatted file listings (markdown)
- Updates dataset descriptions with current file inventory

Key methods:
- `scan()` - Scan directory and update internal state
- `detect_new_files()` - Return files added since last scan
- `generate_file_listing()` - Create markdown list of files with sizes
- `update_dataset_description()` - Refresh description with current files
- `ensure_data_directory_exists()` - Create data directory if needed

### 4. HypothesisBrancher (`DiscoveryAgent/hypothesis_brancher.py`)

When `expanding_dataset_mode=True`:
- Initializes `DataDirectoryManager` on startup
- Ensures data directory exists at the start of each round
- Passes `data_manager` to workflow instances
- Provides `refresh_dataset_description()` method

### 5. DirectorWorkflow (`DiscoveryAgent/director_workflow.py`)

Method `_refresh_dataset_description()`, which:
- Is called after each successful query
- Scans data directory for changes
- Logs any new files detected
- Updates `self.dataset_description` with current file listing

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Workflow Execution                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ROUND START                                                             │
│  ├── DataDirectoryManager.ensure_data_directory_exists()                │
│  └── DataDirectoryManager.scan() - Initial inventory                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  QUERY N                                                                 │
│  ├── Director generates query with current dataset_description          │
│  ├── ExecutionAgent runs analysis                                       │
│  │   └── Agent may ADD files to data/ (downloads, processed outputs)   │
│  ├── Report updated                                                     │
│  ├── Checkpoint saved                                                   │
│  └── _refresh_dataset_description()                                     │
│      ├── DataDirectoryManager.scan()                                    │
│      ├── DataDirectoryManager.detect_new_files()                        │
│      │   └── Logs: "📁 Detected N new data file(s)"                    │
│      └── DataDirectoryManager.update_dataset_description()              │
│          └── dataset_description now includes new files                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  QUERY N+1                                                               │
│  └── Director sees updated dataset_description with new files           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Dataset Description Template

Your `README.txt` or dataset description can include the `{file_listing}` placeholder:

```markdown
# My Dataset

This dataset contains experimental results from our study.

## Task
Analyze the data and identify key patterns.

## Available Data Files
{file_listing}

## Notes
- Data is collected daily
- New files may be added during analysis
```

When the workflow runs, `{file_listing}` is replaced with a formatted list:

```markdown
## Available Data Files
- `raw_data.csv` (2.3 MB)
- `metadata.json` (15.2 KB)
- `processed/cleaned_data.parquet` (1.1 MB)
- `external/downloaded_reference.csv` (456.7 KB)
```

If no placeholder is found, the file listing is appended at the end of the description.

---

## File Safety

Even with expanding dataset mode enabled, agents are still restricted:

| Action | Allowed? |
|--------|----------|
| Read existing files | ✅ Yes |
| Add new files to `data/` | ✅ Yes |
| Create subdirectories in `data/` | ✅ Yes |
| Modify existing files | ❌ No |
| Delete files | ❌ No |
| Overwrite existing files | ❌ No |

---

## Example Output

When new files are detected, you'll see console output like:

```
📁 Detected 3 new data file(s):
   + downloaded_census_2024.csv
   + processed/normalized_results.parquet
   + external/reference_data.json
```

And in the logs:
```
[data_files_added] Detected 3 new data file(s): ['downloaded_census_2024.csv', ...]
[dataset_description_refreshed] Dataset description updated: 1250 -> 1580 chars
```


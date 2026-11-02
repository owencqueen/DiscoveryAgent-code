# Expanding dataset mode

The economics and GW-CD4 configurations enable this mode so domain tools can
add downloaded data to the workspace. The director receives an updated file
listing after successful queries.

```bash
.venv/bin/python -m scripts.run \
  --working_directory /path/to/dataset \
  --analysis_name example \
  --expanding-dataset \
  --expanding-dataset-dir data/downloads \
  --codebase_structure_prompt_path configs/workspace_expanding.txt
```

The monitored path is relative to the execution workspace unless an absolute
path is supplied. Include `{file_listing}` in the dataset description to place
the listing at that location; otherwise the listing is appended.

The workspace prompt instructs the execution agent to preserve existing data
and add new files separately. File tracking reports missing initial files;
it does not enforce immutability of every existing file. Use read-only mounts
for source data that must not change.

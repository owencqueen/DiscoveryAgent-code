# Setup

Run `bash setup/setup.sh` from any directory to install the core workflow in a
Python 3.11 environment. The script uses `uv.lock` and verifies the import.

Optional environments:

```bash
bash setup/setup.sh --extra acs          # economics Census agent
bash setup/setup.sh --extra opentargets  # functional-genomics data agent
bash setup/setup.sh --extra analysis     # common packages for execution-agent code
bash setup/setup.sh --extra dev          # tests
```

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`,
`PERPLEXITY_API_KEY`, and `DATA_PATH` before running a campaign. The launcher
loads `.env` from the repository root without overriding exported shell values.

Datasets, credentials and domain resources are not part of this repository
(see "What is not included" in the top-level README).

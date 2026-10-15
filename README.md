# DiscoveryAgent

DiscoveryAgent is the research-agent framework described in the accompanying
manuscript. Given a dataset and a short description of it, the agent runs
rounds of computational research: a director agent plans each step and sends
queries to tool agents that write and run code, search the web, and review the
literature, while a reviewer critiques the work and an empirical memory carries
findings from round to round. Rounds run automatically, or a scientist chooses
the hypothesis for each round.

This repository releases the **core agent code** used in the paper's six
campaigns, together with each campaign's launch configuration. It does not
contain the datasets, the code the execution agent wrote during the campaigns,
or the scientist steering web application (see [What is not included](#what-is-not-included)).

> **This is research code.** It is a raw representation of the agent as it was
> implemented for the paper, kept close to the original so the methods can be
> inspected and rerun. For a production version of the agent that is easy to
> install and use, see the DiscoveryAgent App: **[link to come]**.

## How it works

1. **Rounds.** `HypothesisBrancher` (`DiscoveryAgent/hypothesis_brancher.py`) runs one or
   more research rounds. A round explores the data, tests a generated
   hypothesis, or tests a hypothesis a scientist selected.
2. **Director.** Within a round, `DirectorWorkflow` (`DiscoveryAgent/director_workflow.py`)
   holds a multi-turn conversation with the director agent. On each step the
   director sends one query to one tool agent, reads its report, and either
   continues or stops before the query budget runs out.
3. **Tool agents** (`DiscoveryAgent/tool_agents/`):
   - `execution_agent`: a Claude Code agent (via the Claude Agent SDK) that works
     in a sandboxed copy of your dataset directory, with documentation
     sub-agents.
   - `web_search`: Anthropic web search, at most three searches per call.
   - `deep_research`: Perplexity `sonar-deep-research`.
   - Optional domain agents: a US Census agent (economics) and an Open Targets
     agent (functional genomics).
4. **Review.** A review designer writes a task-specific reviewer prompt, and the
   reviewer critiques the director's progress at a fixed query interval
   (`DiscoveryAgent/meta_agents/`).
5. **Memory.** A three-agent empirical memory (`DiscoveryAgent/memory/results_memory.py`)
   summarizes each round's results into a report that seeds hypothesis
   generation for the next round.
6. **Provenance.** Query and round checkpoints record every agent's state and
   report. Git tracking records each change in the execution workspace, and
   `DiscoveryAgent/audit/` reads these records back.

## Requirements

- macOS or Linux, Python 3.11, and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- An Anthropic API key (all agents) and a Perplexity API key (deep research).
- Optional: Census and IPUMS keys for the economics agent, and local Open
  Targets data for the functional-genomics agent.

Runs make many model calls and can take hours. A paper campaign round costs
real API credit, so start with the small smoke run below.

### Run inside a container

The execution agent writes and runs arbitrary code on the machine that hosts
it. This repository does not ship a Docker or Apptainer image: during the
project, the agent ran on Stanford systems inside Stanford-specific sandboxes
that were set up separately from this code. **We recommend running the agent
inside a Docker or Apptainer container** so the models are sandboxed from your
own files and credentials. Mount only the `DATA_PATH` directory and the
repository into the container, and pass API keys as environment variables.
The [execution safeguards](#execution-safeguards) below are a second layer of
defense, not a replacement for a container.

## Installation

From the repository root:

```bash
bash setup/setup.sh          # creates .venv from uv.lock and checks the import
cp .env.example .env
```

Edit `.env`:

```bash
ANTHROPIC_API_KEY=...
PERPLEXITY_API_KEY=...
DATA_PATH=/absolute/path/to/research_workspace   # datasets and run outputs live here
```

The launcher reads `.env` from the repository root and does not override
variables already exported in your shell. The Claude Agent SDK ships the Claude
Code executable, so no separate install is needed.

Optional extras (pass several at once if you like):

```bash
bash setup/setup.sh --extra acs          # economics Census/PUMS/IPUMS agent
bash setup/setup.sh --extra opentargets  # functional-genomics Open Targets agent
bash setup/setup.sh --extra analysis     # common scientific packages for the execution agent
bash setup/setup.sh --extra dev          # pytest
```

## Prepare a dataset

Each dataset is a directory with a `data/` folder and a `data/README.txt` that
describes the files and, if you like, the research task:

```text
$DATA_PATH/my_dataset/
  data/
    README.txt
    measurements.csv
```

The agent reads `README.txt` as its dataset description. Write `{datapath}`
wherever the absolute path of the data should go; the launcher fills it in with
the sandbox's data path. A separate research brief can be passed with
`--task_description_file`; the paper's economics, GW-CD4 and Dwarf campaigns
used one, and the others put the task in the README.

## Run the agent

All runs go through one launcher, `scripts/run.py`. Run it from the
repository root.

### Smoke run

Start from a paper configuration and shrink the budget with command-line
flags, which always override the YAML:

```bash
.venv/bin/python -m scripts.run \
  --config-yaml configs/paper/dms_auto.yaml \
  --working_directory "$DATA_PATH/my_dataset" \
  --analysis_name smoke \
  --num_rounds 1 \
  --num_queries 3
```

`dms_auto.yaml` is the paper's automatic mode: an exploratory round with no
scientist input.

### Paper campaign configurations

`configs/paper/` has one launch file per paper campaign. Each records that
campaign's models, query budget, turn caps, review settings, memory, compaction
and workspace specification. Pass your own data paths on the command line.

| Config | Campaign | Mode | Extra flags needed |
|---|---|---|---|
| `starembed_auto.yaml` | StarEmbed variable-star classification | Automatic | none |
| `dms_auto.yaml` | ProteinGym DMS, DA-Auto | Automatic | none |
| `dms_steer.yaml` | ProteinGym DMS, DA-Steer | Scientist selection | none |
| `dwarf_select.yaml` | Dwarf-galaxy redshift prediction | Scientist selection | `--task_description_file`, `--background_path` |
| `acs_select.yaml` | Economics (American Community Survey) | Scientist selection | `--task_description_file`; `--extra acs` installed |
| `gw_cd4_select.yaml` | GW-CD4 perturbation case study | Scientist selection | `--task_description_file`, `--opentargets_data_path`, `--opentargets_mcp_package_path`; `--extra opentargets` installed |

To run your own data with a campaign's settings:

```bash
.venv/bin/python -m scripts.run \
  --config-yaml configs/paper/gw_cd4_select.yaml \
  --working_directory "$DATA_PATH/my_dataset" \
  --task_description_file /path/to/TASK_DESCRIPTION.md \
  --opentargets_data_path /path/to/opentargets_parquet \
  --opentargets_mcp_package_path /path/to/virtualbiotech_mcp_package
```

The datasets and the dataset and task descriptions for each campaign are not
included in this repository.

### Scientist-selected hypotheses

Configs with `human_selection: true` run one round per invocation and then
stop. Between rounds:

1. Read the round's outputs under `$DATA_PATH/LOGS/<dataset>/<run>/checkpoints/round_N/`.
2. Write the hypothesis to test next, in plain Markdown, to
   `checkpoints/round_N/selected_hypothesis.md`.
3. Resume from that round's config (below). Round N+1 picks up the file
   automatically.

To set the hypothesis for the first round, pass `--selected_hypothesis_file`.
The paper used a separate web application to draft and edit candidate
hypotheses; writing the file by hand is equivalent for the agent.

### Resume a run

Every round saves its full configuration. Resume from it and give the number
of additional rounds:

```bash
.venv/bin/python -m scripts.run \
  --config-file "$DATA_PATH/LOGS/<dataset>/<run>/checkpoints/round_0/config.json" \
  --num_rounds 1
```

A resume continues in the same workspace and log directory, reloads the memory
report, and restarts an unfinished round from its last query checkpoint.

## Outputs

For a run named `<analysis_name>_<timestamp>` on dataset `<dataset>`:

| Path | Contents |
|---|---|
| `$DATA_PATH/<dataset>/<run>/` | Execution workspace (a git repository): code, `data/`, `experiments/`, `models/`, `reports/` |
| `$DATA_PATH/LOGS/<dataset>/<run>/discoveryagent.log` | Every prompt and response, in order |
| `$DATA_PATH/LOGS/<dataset>/<run>/checkpoints/round_N/` | Per-round `config.json`, query checkpoints, reports, memory report, and `selected_hypothesis.md` |

`scripts/utils/audit_run.py` and `DiscoveryAgent/audit/` summarize a run from these
records.

## Useful options

Run `.venv/bin/python -m scripts.run --help` for the full list.

| Option | Meaning |
|---|---|
| `--num_rounds`, `--num_queries` | Rounds per invocation; director queries per round |
| `--num_exploration_rounds` | Leading rounds that explore the data without a hypothesis |
| `--hypothesis_model`, `--director_model`, `--execution_model`, ... | Model for each component |
| `--execution_max_turns` | Turn cap for each execution-agent query |
| `--review_optimization_steps` | Queries between reviewer critiques |
| `--expanding-dataset` | Let tool agents add downloaded data to `data/downloads/` |
| `--use_census_agent`, `--use_opentargets_agent` | Enable the domain agents |
| `--execution_skills_path` | Directory of Claude skills for the execution agent |
| `--no-sandbox-enforcement`, `--no-url-blocker` | Turn off the execution safeguards (below) |

## Execution safeguards

The execution agent works in a per-run workspace inside the dataset directory.
The files present when it starts are recorded, and the run reports any that
go missing. The Open Targets agent's write protection keeps it out of the
source data. This release adds two safeguards the paper campaigns did not
have, both on by default:

- **Sandbox path enforcement** blocks file access outside the run's workspace
  and adds a short note about this to the execution agent's prompt.
- **A URL blocker** denies downloads from ProteinGym test-label sources, which
  would leak held-out labels in the DMS benchmark.

Pass `--no-sandbox-enforcement` and `--no-url-blocker` to run as the paper
campaigns did. None of this replaces a container (see
[Run inside a container](#run-inside-a-container)).

## Differences from the paper's runs

The prompts in this code match those of the DMS-Auto and Dwarf campaigns. The
code was updated while the campaigns were running, so the earlier campaigns
used slightly different director prompt text:

- StarEmbed and ACS lack one director rule ("You may only submit one query to
  one of the helper agents per step…").
- StarEmbed used a shorter description of the execution agent.
- Four campaigns did not repeat the stop reminder in the director's first
  message.

Other differences:

- The two [execution safeguards](#execution-safeguards) were added after the
  paper.
- New runs name query folders `query_01` instead of `query_001`; resuming an
  older checkpoint reads both.
- The pinned dependencies in `uv.lock` are a current environment, not a frozen
  copy of the cluster environment.

## What is not included

- Datasets, API credentials, the HPC container image and Slurm environment.
- Code, models and reports the execution agent produced during the campaigns.
- The scientist steering web application (see above for the file-based
  equivalent).
- Domain resources: Open Targets data and its loader package, the Dwarf
  background PDFs, and the BioMCP skill used in GW-CD4.

## Repository map

| Path | Contents |
|---|---|
| `DiscoveryAgent/hypothesis_brancher.py` | Rounds, scientist-selection handoff, memory integration |
| `DiscoveryAgent/director_workflow.py` | Director loop, tool dispatch, review, reports |
| `DiscoveryAgent/hyp_agents/` | Director and hypothesis-generation agents |
| `DiscoveryAgent/tool_agents/` | Execution, web search, deep research, Census, Open Targets agents; `security/` safeguards |
| `DiscoveryAgent/meta_agents/` | Review designer |
| `DiscoveryAgent/memory/` | Three-agent empirical memory |
| `DiscoveryAgent/prompts/` | Prompt text for all agents |
| `DiscoveryAgent/checkpoint_manager.py`, `DiscoveryAgent/audit/` | Checkpoints and audit readers |
| `scripts/run.py` | Launcher and resume entry point |
| `acs_agent/` | Bundled Census/PUMS/IPUMS helper used by the economics agent |
| `configs/paper/` | Launch configurations of the six paper campaigns |
| `configs/workspace_*.txt` | Workspace layouts given to the execution agent |
| `guides/` | Notes on expanding-dataset mode |
| `tests/` | Tests for the released code |

## Tests

```bash
bash setup/setup.sh --extra dev
.venv/bin/python -m pytest
```

The tests make no API calls.

## Issues

Please report problems with the codebase in this repository's
[Issues](../../issues), for example a missing file, a step that fails, or a
difference from what the paper describes.

## License

MIT (`LICENSE`). The bundled Census helper keeps its own MIT license in
`acs_agent/LICENSE`.

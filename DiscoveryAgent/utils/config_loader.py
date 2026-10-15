import os
from argparse import ArgumentParser
from typing import Any, Dict, Iterable, Optional, Set

import yaml

# Legacy YAML key aliases for backward compatibility.
# Maps old/alternative YAML keys to the canonical argparse dest names.
# New arguments do NOT need to be added here - they are auto-discovered from argparse.
LEGACY_YAML_ALIASES: Dict[str, str] = {
    # Legacy key that maps to the new canonical name
    'expanding_data_directory': 'expanding_dataset_dir',
}


def explicit_cli_destinations(
    parser: ArgumentParser,
    argv: Iterable[str],
) -> Set[str]:
    """Return argparse destinations explicitly present in ``argv``.

    Comparing parsed values with defaults misses explicit options whose chosen
    value equals the default. On resume, explicit CLI choices must override
    the values saved in the checkpoint.
    """
    explicit: Set[str] = set()
    option_actions = parser._option_string_actions
    for token in argv:
        if token == "--":
            break
        option = token.split("=", 1)[0]
        action = option_actions.get(option)
        if action is not None and action.dest != "help":
            explicit.add(action.dest)
    return explicit


def build_arg_mapping_from_parser(parser: ArgumentParser) -> Dict[str, str]:
    """
    Build a YAML key -> argparse dest mapping from an ArgumentParser.

    This auto-discovers all arguments from the parser, so new arguments
    only need to be added to arg_parsers.py - no manual mapping required.

    The mapping includes:
    1. All argparse dest names (identity mapping)
    2. All option strings (--foo-bar -> foo_bar)
    3. Legacy aliases from LEGACY_YAML_ALIASES

    Args:
        parser: The ArgumentParser to extract arguments from

    Returns:
        Dict mapping YAML keys to argparse dest names
    """
    mapping = {}

    for action in parser._actions:
        if action.dest == 'help':
            continue

        dest = action.dest

        # Add identity mapping for the dest name
        mapping[dest] = dest

        # Add mappings for all option strings (e.g., --foo-bar -> foo_bar)
        for opt in action.option_strings:
            # Strip leading dashes and convert to underscore
            key = opt.lstrip('-').replace('-', '_')
            mapping[key] = dest

    # Add legacy aliases
    mapping.update(LEGACY_YAML_ALIASES)

    return mapping


def load_yaml_config(yaml_path: Optional[str]) -> Dict[str, Any]:
    """Load YAML configuration from the given path. Returns {} on failure."""
    if not yaml_path:
        return {}
    if not os.path.exists(yaml_path):
        print(f"⚠️  YAML config file not found: {yaml_path}")
        return {}
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            print(f"⚠️  YAML config at {yaml_path} is not a mapping/object; ignoring.")
            return {}
        return data
    except Exception as e:
        print(f"⚠️  Failed to load YAML config from {yaml_path}: {e}")
        return {}


def apply_updates_with_cli_priority(
    args,
    updates: Dict[str, Any],
    cli_override_keys: Set[str],
    parser: Optional[ArgumentParser] = None,
    mapping: Optional[Dict[str, str]] = None,
    source_label: str = "config",
):
    """
    Apply values from a mapping (YAML or checkpoint config) onto argparse args.

    CLI overrides are respected: if a key was provided via CLI, it will not be
    overwritten. Later calls can override earlier non-CLI values (e.g., YAML can
    override checkpoint config).

    Args:
        args: The argparse Namespace to update
        updates: Dict of config values to apply
        cli_override_keys: Set of keys that were explicitly set via CLI (these won't be overwritten)
        parser: Optional ArgumentParser to auto-build the mapping from.
                If provided, mapping is built automatically from parser's arguments.
        mapping: Optional explicit mapping dict. If neither parser nor mapping provided,
                 uses simple key normalization (replace - with _).
        source_label: Label for logging (e.g., "YAML config", "checkpoint")

    Returns:
        Updated args Namespace
    """
    # Build mapping from parser if provided, otherwise use explicit mapping or simple normalization
    if parser is not None:
        effective_mapping = build_arg_mapping_from_parser(parser)
    elif mapping is not None:
        effective_mapping = mapping
    else:
        # Fallback: simple key normalization, no explicit mapping
        effective_mapping = {}

    applied_count = 0
    skipped_count = 0

    for raw_key, value in updates.items():
        # Normalize the key: check mapping first, then just normalize dashes to underscores
        target_key = effective_mapping.get(raw_key, raw_key).replace("-", "_")

        if target_key in cli_override_keys:
            skipped_count += 1
            continue
        if not hasattr(args, target_key):
            skipped_count += 1
            continue
        if value is None:
            skipped_count += 1
            continue

        setattr(args, target_key, value)
        applied_count += 1

    print(f"✓ Applied {applied_count} values from {source_label}")
    if skipped_count > 0:
        print(f"   - Skipped {skipped_count} values (CLI override, missing arg, or None)")

    return args


def snapshot_args_to_yaml(args, output_path: str):
    """Persist resolved argparse Namespace to YAML for future reuse."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Underscore-prefixed values are process-local coordination state, not
        # reproducible user configuration (for example resume authority flags).
        serializable = {
            key: value
            for key, value in vars(args).items()
            if not key.startswith("_")
        }
        with open(output_path, 'w') as f:
            yaml.safe_dump(serializable, f, sort_keys=True)
        print(f"✓ Saved resolved arguments to YAML: {output_path}")
    except Exception as e:
        print(f"⚠️  Could not write args YAML to {output_path}: {e}")

#!/usr/bin/env python3
"""
DiscoveryAgent Workflow Audit Script

This script audits a DiscoveryAgent workflow run by analyzing checkpoints and generating
a human-readable summary report using Claude Haiku.

Usage:
    python scripts/utils/audit_run.py <checkpoint_path> <output_path>

Example:
    python scripts/utils/audit_run.py logs/checkpoints audit_report.md
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from DiscoveryAgent.audit import (
    CheckpointDiscovery,
    DataExtractor,
    AuditSummarizer,
    AuditReportGenerator
)


def _read_selected_hypotheses(checkpoint_base: Path, rounds_to_audit: list) -> dict:
    """
    Read selected_hypothesis.md files to build a mapping of which hypothesis
    drove each round's research.

    For round N (N > 0), the driving hypothesis is stored in
    checkpoints/round_{N-1}/selected_hypothesis.md (the output of round N-1).

    For round 0, if a selected_hypothesis.md exists there, it represents the
    initial exploration directive (not from a previous round).

    Args:
        checkpoint_base: Path to the checkpoints/ directory
        rounds_to_audit: List of round numbers being audited

    Returns:
        Dict mapping round_num -> hypothesis text that drove that round
    """
    per_round_hypotheses = {}

    for round_num in rounds_to_audit:
        if round_num == 0:
            # Round 0 has no previous round; skip
            continue

        # The hypothesis that drives round N is selected at the end of round N-1
        prev_round_dir = checkpoint_base / f"round_{round_num - 1}"
        hyp_file = prev_round_dir / "selected_hypothesis.md"

        if hyp_file.exists():
            try:
                with open(hyp_file, 'r', encoding='utf-8') as f:
                    per_round_hypotheses[round_num] = f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read {hyp_file}: {e}")

    return per_round_hypotheses


def main():
    """Main audit script."""
    parser = argparse.ArgumentParser(
        description='Audit a DiscoveryAgent workflow run and generate a summary report.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Audit all rounds (parallel mode, fast)
  python scripts/utils/audit_run.py logs/checkpoints audit_report.md

  # Audit a specific round
  python scripts/utils/audit_run.py logs/checkpoints audit_report.md --round 1

  # Audit with hypotheses included (for hypothesis branching workflows)
  python scripts/utils/audit_run.py logs/checkpoints audit_report.md --hb_mode

  # Audit with reviewer feedback included
  python scripts/utils/audit_run.py logs/checkpoints audit_report.md --audit_reviews

  # Full audit with hypotheses and reviews
  python scripts/utils/audit_run.py logs/checkpoints audit_report.md --hb_mode --audit_reviews

  # Audit with custom concurrency limit
  python scripts/utils/audit_run.py logs/checkpoints audit_report.md --max-concurrent 20

  # Audit in sequential mode (slower but more conservative)
  python scripts/utils/audit_run.py logs/checkpoints audit_report.md --no-parallel

  # Audit a specific run with full path
  python scripts/utils/audit_run.py /path/to/run/logs/checkpoints /path/to/output/report.md

Note: By default, parallel processing is enabled for faster summarization.
Use --no-parallel if you encounter rate limit issues.
Use --hb_mode to include generated hypotheses in the report.
Use --audit_reviews to include reviewer feedback in the report.
        """
    )

    parser.add_argument(
        'checkpoint_path',
        type=str,
        help='Path to the checkpoints directory or logs directory'
    )

    parser.add_argument(
        'output_path',
        type=str,
        help='Path where the audit report should be saved (e.g., audit_report.md)'
    )

    parser.add_argument(
        '--round',
        type=int,
        default=None,
        help='Audit only a specific round (e.g., --round 1). If not set, all rounds are audited.'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='claude-haiku-4-5-20251001',
        help='Claude model to use for summarization (default: claude-haiku-4-5-20251001)'
    )

    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=10,
        help='Maximum number of concurrent API requests (default: 10)'
    )

    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing (slower but more conservative)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output (only show errors)'
    )

    parser.add_argument(
        '--hb_mode',
        action='store_true',
        help='Include generated hypotheses in the audit report (useful for hypothesis branching workflows)'
    )

    parser.add_argument(
        '--audit_reviews',
        action='store_true',
        help='Include reviewer feedback in the audit report'
    )

    args = parser.parse_args()

    # Validate paths
    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)

    output_path = Path(args.output_path)

    verbose = not args.quiet

    # Print header
    if verbose:
        print("=" * 80)
        print("DiscoveryAgent Workflow Audit")
        print("=" * 80)
        print(f"Checkpoint Path: {checkpoint_path}")
        print(f"Output Path: {output_path}")
        print(f"Model: {args.model}")
        if args.round is not None:
            print(f"Round Filter: {args.round}")
        if not args.no_parallel:
            print(f"Parallel Processing: Enabled (max concurrent: {args.max_concurrent})")
        else:
            print(f"Parallel Processing: Disabled (sequential mode)")
        if args.hb_mode:
            print(f"Hypothesis Mode: Enabled (will include generated hypotheses)")
        if args.audit_reviews:
            print(f"Review Audit: Enabled (will include reviewer feedback)")
        print("=" * 80)
        print()

    try:
        # Step 1: Discover checkpoints
        if verbose:
            print("Step 1/4: Discovering checkpoints...")

        discovery = CheckpointDiscovery(str(checkpoint_path))

        # Validate --round argument if provided
        if args.round is not None:
            available_rounds = discovery.get_available_rounds()
            if available_rounds and args.round not in available_rounds:
                print(f"Error: Round {args.round} not found. Available rounds: {available_rounds}")
                sys.exit(1)

        checkpoints = discovery.discover_checkpoints(round_filter=args.round)

        if not checkpoints:
            print(f"Error: No checkpoints found in {checkpoint_path}")
            if args.round is not None:
                print(f"   (filtered to round {args.round})")
            print("   Make sure the path points to a directory containing checkpoint files.")
            sys.exit(1)

        statistics = discovery.get_statistics(checkpoints)

        if verbose:
            print(f"   Found {statistics['total_checkpoints']} checkpoints")
            if statistics['has_rounds']:
                print(f"   Multi-round workflow: {statistics['total_rounds']} round(s)")
                for round_num, count in sorted(statistics['queries_per_round'].items()):
                    print(f"     - Round {round_num}: {count} queries")
            else:
                print(f"   Single-round workflow")
            print()

        # Determine which rounds are being audited and read selected hypotheses
        if statistics['has_rounds']:
            rounds_to_audit = sorted(statistics['queries_per_round'].keys())
        else:
            rounds_to_audit = []

        per_round_hypotheses = _read_selected_hypotheses(
            discovery.checkpoint_base, rounds_to_audit
        )

        if verbose and per_round_hypotheses:
            for rn in sorted(per_round_hypotheses.keys()):
                # Show first 80 chars of hypothesis
                preview = per_round_hypotheses[rn][:80].replace('\n', ' ')
                print(f"   Selected hypothesis for round {rn}: {preview}...")
            print()

        # Step 2: Extract data
        if verbose:
            print("Step 2/4: Extracting query and result data...")
            if args.hb_mode:
                print("   Including hypothesis data...")
            if args.audit_reviews:
                print("   Including review data...")

        extractor = DataExtractor(
            extract_hypotheses=args.hb_mode,
            extract_reviews=args.audit_reviews
        )
        pairs = extractor.extract_all(
            checkpoints,
            per_round_hypotheses=per_round_hypotheses
        )

        if not pairs:
            print(f"Error: Could not extract any data from checkpoints")
            sys.exit(1)

        if verbose:
            print(f"   Extracted {len(pairs)} query-result pairs")
            print()

        # Step 3: Summarize with LLM
        if verbose:
            print(f"Step 3/4: Summarizing with {args.model}...")
            if not args.no_parallel:
                print(f"   Using parallel processing for faster summarization...")
            else:
                print(f"   Using sequential processing...")
            print()

        summarizer = AuditSummarizer(model_name=args.model, max_concurrent=args.max_concurrent)

        # Use parallel or sequential processing
        if args.no_parallel:
            summaries = summarizer.summarize_all(pairs, verbose=verbose)
        else:
            summaries = summarizer.summarize_all_parallel(pairs, verbose=verbose)

        if verbose:
            cost_estimate = summarizer.get_total_cost_estimate()
            print(f"   Generated {len(summaries)} summaries")
            print(f"   Total tokens: {summarizer.total_tokens:,}")
            print(f"   Estimated cost: ${cost_estimate:.4f}")
            print()

        # Step 4: Generate report
        if verbose:
            print("Step 4/4: Generating audit report...")

        generator = AuditReportGenerator(
            include_hypotheses=args.hb_mode,
            include_reviews=args.audit_reviews,
            per_round_hypotheses=per_round_hypotheses
        )
        report = generator.generate_report(
            summaries=summaries,
            checkpoint_path=str(checkpoint_path),
            output_path=str(output_path),
            statistics=statistics
        )

        # Save report
        saved_path = generator.save_report(report, str(output_path))

        if verbose:
            print(f"   Report generated ({len(report):,} characters)")
            print()

        # Success message
        print("=" * 80)
        print("Audit Complete!")
        print("=" * 80)
        print(f"Report saved to: {saved_path}")
        print(f"Total queries audited: {len(summaries)}")
        if statistics['has_rounds']:
            print(f"Round(s): {', '.join(str(r) for r in sorted(statistics['queries_per_round'].keys()))}")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nAudit cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during audit: {str(e)}")
        import traceback
        if verbose:
            print("\nFull traceback:")
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

"""
Command-line interface for ACS Access.

Provides a CLI for common ACS data operations:
- Downloading tables
- Searching for tables/variables
- Listing available data
- Getting table information

Usage:
    python -m acs_agent.acs_access download-table B01001 --state Michigan
    python -m acs_agent.acs_access search-tables "median income"
    python -m acs_agent.acs_access list-tables --year 2022
"""

import asyncio
import argparse

from rich.console import Console

from .client import ACSClient
from .config import ACSDataset

console = Console()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download American Community Survey data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a table for a state
  python -m acs_agent.acs_access download-table B01001 --state Michigan --year 2022
  
  # Download for a specific county
  python -m acs_agent.acs_access download-table B19013 --state MI --county "Wayne County"
  
  # Download for all states
  python -m acs_agent.acs_access download-table B01001 --all-states --year 2022
  
  # Search for tables
  python -m acs_agent.acs_access search-tables "median income" --year 2022
  
  # List available tables
  python -m acs_agent.acs_access list-tables --dataset acs5 --year 2022
  
  # Download time series
  python -m acs_agent.acs_access download-timeseries B01001 --state California --years 2015-2022
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Download table command
    dl_table = subparsers.add_parser("download-table", help="Download a full ACS table")
    dl_table.add_argument("table_id", help="Table ID (e.g., B01001)")
    dl_table.add_argument("--state", help="State name or abbreviation")
    dl_table.add_argument("--county", help="County name")
    dl_table.add_argument("--place", help="Place/city name")
    dl_table.add_argument("--year", type=int, default=2022, help="ACS year")
    dl_table.add_argument("--dataset", default="acs/acs5", help="ACS dataset")
    dl_table.add_argument("--output", "-o", default=".", help="Output directory")
    dl_table.add_argument("--all-states", action="store_true", help="Download for all states")
    dl_table.add_argument("--all-counties", action="store_true", help="Download for all counties")
    dl_table.add_argument("--no-moe", action="store_true", help="Exclude margin of error")
    
    # Download variables command
    dl_vars = subparsers.add_parser("download-variables", help="Download specific variables")
    dl_vars.add_argument("variables", nargs="+", help="Variable names")
    dl_vars.add_argument("--state", required=True, help="State")
    dl_vars.add_argument("--county", help="County")
    dl_vars.add_argument("--year", type=int, default=2022)
    dl_vars.add_argument("--output", "-o", default=".")
    
    # Download time series command
    dl_ts = subparsers.add_parser("download-timeseries", help="Download table across years")
    dl_ts.add_argument("table_id", help="Table ID")
    dl_ts.add_argument("--state", required=True)
    dl_ts.add_argument("--years", required=True, help="Year range (e.g., 2015-2022)")
    dl_ts.add_argument("--output", "-o", default=".")
    
    # Search tables command
    search = subparsers.add_parser("search-tables", help="Search for tables")
    search.add_argument("query", help="Search query")
    search.add_argument("--year", type=int, default=2022)
    search.add_argument("--dataset", default="acs/acs5")
    search.add_argument("--max", type=int, default=20)
    
    # Search variables command
    search_vars = subparsers.add_parser("search-variables", help="Search for variables")
    search_vars.add_argument("query", help="Search query")
    search_vars.add_argument("--year", type=int, default=2022)
    search_vars.add_argument("--dataset", default="acs/acs5")
    search_vars.add_argument("--max", type=int, default=30)
    
    # List tables command
    list_tables = subparsers.add_parser("list-tables", help="List available tables")
    list_tables.add_argument("--year", type=int, default=2022)
    list_tables.add_argument("--dataset", default="acs/acs5")
    
    # Get table info command
    table_info = subparsers.add_parser("table-info", help="Get table details")
    table_info.add_argument("table_id", help="Table ID")
    table_info.add_argument("--year", type=int, default=2022)
    table_info.add_argument("--dataset", default="acs/acs5")
    
    # List datasets command
    list_ds = subparsers.add_parser("list-datasets", help="List available datasets")
    list_ds.add_argument("--year", type=int, default=2022)
    
    # Clear cache command
    clear_cache = subparsers.add_parser("clear-cache", help="Clear metadata cache")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run the appropriate async function
    asyncio.run(run_command(args))


async def run_command(args):
    """Execute the CLI command."""
    async with ACSClient() as client:
        
        if args.command == "download-table":
            await cmd_download_table(client, args)
        
        elif args.command == "download-variables":
            await cmd_download_variables(client, args)
        
        elif args.command == "download-timeseries":
            await cmd_download_timeseries(client, args)
        
        elif args.command == "search-tables":
            await cmd_search_tables(client, args)
        
        elif args.command == "search-variables":
            await cmd_search_variables(client, args)
        
        elif args.command == "list-tables":
            await cmd_list_tables(client, args)
        
        elif args.command == "table-info":
            await cmd_table_info(client, args)
        
        elif args.command == "list-datasets":
            await cmd_list_datasets(client, args)
        
        elif args.command == "clear-cache":
            client.clear_cache()
            console.print("[green]✓ Cache cleared[/green]")


async def cmd_download_table(client: ACSClient, args):
    """Handle download-table command."""
    if args.all_states:
        result = await client.download_all_states(
            args.table_id, args.year, args.dataset, args.output, not args.no_moe
        )
    elif args.all_counties:
        result = await client.download_all_counties(
            args.table_id, args.state, args.year, args.dataset, args.output, not args.no_moe
        )
    else:
        geography = {}
        if args.state:
            geography["state"] = args.state
        if args.county:
            geography["county"] = args.county
        if args.place:
            geography["place"] = args.place
        
        if not geography:
            console.print("[red]Error: Specify --state, --all-states, or --all-counties[/red]")
            return
        
        result = await client.download_table(
            args.table_id, geography, args.year, args.dataset,
            args.output, not args.no_moe
        )
    
    if result.success:
        console.print(f"[green]✓ Downloaded to {result.file_path}[/green]")
        console.print(f"  Rows: {result.rows}, Columns: {result.columns}")
    else:
        console.print(f"[red]✗ Error: {result.error}[/red]")


async def cmd_download_variables(client: ACSClient, args):
    """Handle download-variables command."""
    geography = {"state": args.state}
    if args.county:
        geography["county"] = args.county
    
    result = await client.download_variables(
        args.variables, geography, args.year, output_dir=args.output
    )
    
    if result.success:
        console.print(f"[green]✓ Downloaded to {result.file_path}[/green]")
    else:
        console.print(f"[red]✗ Error: {result.error}[/red]")


async def cmd_download_timeseries(client: ACSClient, args):
    """Handle download-timeseries command."""
    # Parse year range
    if "-" in args.years:
        start, end = map(int, args.years.split("-"))
        years = list(range(start, end + 1))
    else:
        years = [int(y) for y in args.years.split(",")]
    
    result = await client.download_table_time_series(
        args.table_id,
        {"state": args.state},
        years,
        output_dir=args.output
    )
    
    if result.success:
        console.print(f"[green]✓ Downloaded to {result.file_path}[/green]")
    else:
        console.print(f"[red]✗ Error: {result.error}[/red]")


async def cmd_search_tables(client: ACSClient, args):
    """Handle search-tables command."""
    tables = await client.search_tables(args.query, args.year, args.dataset, args.max)
    
    console.print(f"\n[bold]Found {len(tables)} tables matching '{args.query}':[/bold]\n")
    for table in tables:
        console.print(f"  [cyan]{table.table_id}[/cyan]: {table.title}")
        if table.universe:
            console.print(f"    Universe: {table.universe}")


async def cmd_search_variables(client: ACSClient, args):
    """Handle search-variables command."""
    variables = await client.search_variables(args.query, args.year, args.dataset, args.max)
    
    console.print(f"\n[bold]Found {len(variables)} variables matching '{args.query}':[/bold]\n")
    for var in variables:
        console.print(f"  [cyan]{var.name}[/cyan]: {var.label[:60]}...")


async def cmd_list_tables(client: ACSClient, args):
    """Handle list-tables command."""
    tables = await client.list_tables(args.year, args.dataset)
    
    console.print(f"\n[bold]Available tables ({len(tables)}):[/bold]\n")
    for table in tables[:50]:
        title = table.title[:70] + "..." if len(table.title) > 70 else table.title
        console.print(f"  [cyan]{table.table_id}[/cyan]: {title}")
    
    if len(tables) > 50:
        console.print(f"\n  ... and {len(tables) - 50} more")


async def cmd_table_info(client: ACSClient, args):
    """Handle table-info command."""
    info = await client.get_table_info(args.table_id, args.year, args.dataset)
    
    if info:
        console.print(f"\n[bold]{info.table_id}: {info.title}[/bold]")
        console.print(f"Universe: {info.universe}")
        console.print(f"Variables: {len(info.variables)}\n")
        
        for var in info.variables[:20]:
            console.print(f"  [cyan]{var.name}[/cyan]: {var.label}")
        
        if len(info.variables) > 20:
            console.print(f"\n  ... and {len(info.variables) - 20} more variables")
    else:
        console.print(f"[red]Table {args.table_id} not found[/red]")


async def cmd_list_datasets(client: ACSClient, args):
    """Handle list-datasets command."""
    datasets = await client.list_datasets(args.year)
    
    console.print(f"\n[bold]Available ACS datasets for {args.year}:[/bold]\n")
    for ds in datasets:
        console.print(f"  [cyan]{ds['identifier']}[/cyan]")
        console.print(f"    {ds['title']}")


if __name__ == "__main__":
    main()


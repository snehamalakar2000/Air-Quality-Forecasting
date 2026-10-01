"""Run from the repository root: python -m air_quality --help."""

import argparse
import json

from .pipeline import evaluate, read_config, select, smoke, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["validate", "select", "evaluate"]:
        command = commands.add_parser(name)
        command.add_argument("--config", default="config.json")
        if name == "evaluate":
            command.add_argument("--selection", required=True)
    commands.add_parser("smoke", help="Offline, deterministic synthetic end-to-end check")
    args = parser.parse_args()
    try:
        if args.command == "smoke":
            result = smoke()
        else:
            config = read_config(args.config)
            if args.command == "validate":
                quality = validate(config)
                result = {"status": "valid", "raw_rows": quality["raw_rows"],
                          "absent_hours": quality["absent_hours"], "partitions": quality["partitions"]}
            elif args.command == "select":
                manifest = select(config)
                result = {"selected_learned_model": manifest["selected_learned_model"],
                          "recommended_method": manifest["recommended_method"]}
            else:
                result = evaluate(config, args.selection)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()

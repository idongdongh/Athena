"""Tracer Evaluation 命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.pipeline import build_report, evaluate_run, wash_run
from evaluation.runner import run_manifest
from evaluation.judge_pipeline import judge_plan, judge_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="athena-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("evaluate", "wash", "report"):
        item = subparsers.add_parser(command)
        item.add_argument("run", type=Path)
        item.add_argument("--output", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--repetitions", type=int, default=1)
    run.add_argument("--no-resume", action="store_true")
    judge = subparsers.add_parser("judge")
    judge.add_argument("run", type=Path)
    judge.add_argument("--output", type=Path)
    judge.add_argument("--model")
    judge.add_argument("--max-concurrency", type=int, default=2)
    judge.add_argument("--rpm", type=int, default=20)
    judge.add_argument("--execute", action="store_true")
    judge.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        destination = run_manifest(
            args.manifest,
            args.output,
            repetitions=args.repetitions,
            resume=not args.no_resume,
        )
        print(destination)
        return 0
    if args.command == "judge":
        if not args.execute:
            print(json.dumps(judge_plan(args.run, args.output), ensure_ascii=False, indent=2))
            return 0
        result = judge_run(
            args.run,
            model=args.model,
            output=args.output,
            max_concurrency=args.max_concurrency,
            requests_per_minute=args.rpm,
            force=args.force,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    evaluated = evaluate_run(args.run, args.output)
    if args.command == "evaluate":
        print(evaluated.output / "evaluations.jsonl")
    elif args.command == "wash":
        wash_run(evaluated)
        print(evaluated.output / "wash_manifest.json")
    else:
        build_report(evaluated)
        print(evaluated.output / "report.md")
    return 0

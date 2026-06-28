import argparse
from dataclasses import replace

from configs import RUNTIMES, Runtime
from networks import ViT


def build_model(runtime: Runtime) -> ViT:
    match runtime.model_kind:
        case "vit":
            return ViT(runtime.model)


def apply_overrides(runtime: Runtime, args: argparse.Namespace) -> Runtime:
    if args.epochs is not None:
        runtime = replace(runtime, train=replace(runtime.train, epochs=args.epochs))
    if args.pruning_ratio is not None:
        runtime = replace(runtime, model=replace(runtime.model, pruning_ratio=args.pruning_ratio))
    return runtime


def resolve_runtime(args: argparse.Namespace) -> Runtime:
    if args.runtime:
        return RUNTIMES[args.runtime]
    if not args.dataset:
        raise SystemExit("ds error")

    kind = args.model_kind
    suffix = {"vit": "vit"}[kind]
    key = f"{args.dataset}_{suffix}_{args.backend}"
    if key not in RUNTIMES:
        raise SystemExit("key error")
    return RUNTIMES[key]

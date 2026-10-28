"""Shared command-line options, so every script exposes the same estimator settings."""

import argparse

from .estimator import DEFAULT_INPUT_SIZE, find_default_model


def add_estimator_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=find_default_model(),
                        help=".pth checkpoint (default: first .pth in models/)")
    parser.add_argument("--device", default="auto",
                        help="auto (GPU if available), cpu, cuda, cuda:1, mps")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE,
                        help=f"resolution the model runs at (default {DEFAULT_INPUT_SIZE}; lower = faster)")


def estimator_kwargs(args: argparse.Namespace) -> dict:
    return {"checkpoint": args.model, "device": args.device, "input_size": args.input_size}


def describe_values(kind: str) -> str:
    """One-line human meaning of the depth values, for printing."""
    if kind == "metric_depth":
        return "metres from the camera (bigger = farther)"
    return "relative 0..1 per frame (0 = nearest, 1 = farthest; ordering only, not metres)"

#!/usr/bin/env python

from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt-path",
        "-K",
        type=Path,
        default=Path("out/ckpt/hab_bl/imgnav"),
    )
    parser.add_argument(
        "--out-path",
        "-o",
        type=Path,
        default=Path("weights/nle/checkpoints/croco-rpve"),
    )
    parser.add_argument("--force", "-f", action="store_true", default=False)
    return parser.parse_args()


def main(args: Namespace) -> int:
    PREFIX = "actor_critic.net.binocular_encoder."
    args.out_path.mkdir(parents=True, exist_ok=args.force)
    for path in (
        args.ckpt_path.glob("*.pth") if args.ckpt_path.is_dir() else (args.ckpt_path,)
    ):
        ckpt = torch.load(path, map_location="cpu")
        binoc_weights = {
            name[len(PREFIX) :]: param
            for name, param in ckpt["state_dict"].items()
            if name.startswith(PREFIX)
        }
        torch.save({"model": binoc_weights}, args.out_path / path.name)
    return 0


if __name__ == "__main__":
    exit(main(parse_args()))

from __future__ import annotations

import os
from pathlib import Path
import argparse
import json

from rzd_main.model import RZDModel


class Predictor:
    """Class for your model's predictions.

    You are free to add your own properties and methods
    or modify existing ones, but the output submission
    structure must be identical to the one presented.

    Examples:
        >>> python -m get_submission --src input_dir --dst output_dir
    """

    def __init__(self):
        self.model = RZDModel()

    def __call__(self, audio_path: str):
        prediction = self.model(audio_path)
        return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get submission.")
    parser.add_argument(
        "--src",
        type=str,
        help="Path to the source audio files.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        help="Path to the output submission.",
    )
    args = parser.parse_args()
    predictor = Predictor()

    results = []
    for audio_path in os.listdir(args.src):
        result = predictor(os.path.join(args.src, audio_path))
        results.append(result)

    if not Path(args.dst).exists():
        os.mkdir(args.dst)

    with open(
        os.path.join(args.dst, "submission.json"), "w", encoding="utf-8"
    ) as outfile:
        json.dump(results, outfile)

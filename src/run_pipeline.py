"""
MAIN ORCHESTRATOR: runs the full pipeline end to end --

    data_download -> preprocess -> train -> evaluate

This does NOT call set_active_model.py or start the API -- promoting a
trained model to "live" and serving it are explicit, on-demand steps by
design (see docs/ARCHITECTURE_NOTES.md). Each stage is also independently runnable
(e.g. `python src/train.py --sample-fraction 0.05`) for debugging one step
at a time.
"""

import argparse

from src import data_download, evaluate, preprocess, train


def main(skip_download: bool = False, sample_fraction: float | None = None) -> None:
    if not skip_download:
        print("\n=== 1/4: data_download ===")
        data_download.main()
    else:
        print("\n=== 1/4: data_download (skipped) ===")

    print("\n=== 2/4: preprocess ===")
    preprocess.main()

    print("\n=== 3/4: train ===")
    train.main(sample_fraction=sample_fraction)

    print("\n=== 4/4: evaluate ===")
    evaluate.main()

    print(
        "\nPipeline complete. Run `mlflow ui` to browse results, "
        "`python src/evaluate.py --version N` to compare specific versions on the test set, "
        "then `python src/set_active_model.py --version N [--reload-api]` to activate one for serving."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data_download.py (use if data/raw is already populated).",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Passed through to train.py for fast iteration.",
    )
    args = parser.parse_args()
    main(skip_download=args.skip_download, sample_fraction=args.sample_fraction)

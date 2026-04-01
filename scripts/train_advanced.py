#!/usr/bin/env python3
"""Advanced training pipeline CLI.

Usage:
    python scripts/train_advanced.py --data-path data/session.npz
    python scripts/train_advanced.py --data-path data/session.npz --augmentation 0.8
    python scripts/train_advanced.py --data-path data/session.npz --skip-hyperopt
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced 5-phase EEG training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to recorded .npz session file",
    )
    parser.add_argument(
        "--augmentation",
        type=float,
        default=0.5,
        help="Augmentation strength 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--hyperopt",
        action="store_true",
        default=False,
        help="Enable Optuna hyperparameter optimization (default: off)",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save results (default: models/)",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings.yaml (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model types to train (default: all). "
        "Options: csp_lda eegnet riemannian neural_sde",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    # Run pipeline
    from src.training.advanced_pipeline import AdvancedTrainingPipeline

    report = AdvancedTrainingPipeline.from_npz(
        data_path=str(data_path),
        config=config,
        augmentation=args.augmentation,
        skip_hyperopt=not args.hyperopt,
        output_dir=args.output_dir,
        model_types=args.models,
    )

    # Print report (pipeline already saves artifacts internally)
    print(report.format())
    print(f"\nArtifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

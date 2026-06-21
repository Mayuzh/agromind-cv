import argparse
import hashlib
import shutil
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from .config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TARGET_CLASSES,
    VALID_IMAGE_EXTENSIONS,
)


def find_class_dir(root: Path, class_name: str) -> Path:
    matches = sorted(path for path in root.rglob(class_name) if path.is_dir())
    if not matches:
        raise FileNotFoundError(
            f"Could not find {class_name!r} below {root}. "
            "See README.md for the expected dataset layout."
        )
    if len(matches) > 1:
        raise ValueError(f"Found multiple folders for {class_name}: {matches}")
    return matches[0]


def valid_images(class_dir: Path) -> list[Path]:
    good: list[Path] = []
    for path in sorted(class_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            good.append(path)
        except (OSError, UnidentifiedImageError):
            print(f"Skipping unreadable image: {path}")
    return good


def copy_files(files: Iterable[Path], split: str, class_name: str, output: Path) -> None:
    destination = output / split / class_name
    destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        # The suffix avoids silent overwrites if a dataset has duplicate basenames.
        digest = hashlib.sha1(str(source).encode()).hexdigest()[:8]
        target = destination / f"{source.stem}_{digest}{source.suffix.lower()}"
        shutil.copy2(source, target)


def prepare_dataset(raw: Path, output: Path, seed: int = 42) -> dict[str, dict[str, int]]:
    if not raw.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {raw}")
    if output.exists():
        shutil.rmtree(output)

    summary: dict[str, dict[str, int]] = {}
    for class_name in TARGET_CLASSES:
        files = valid_images(find_class_dir(raw, class_name))
        if len(files) < 10:
            raise ValueError(f"Need at least 10 valid images for {class_name}; found {len(files)}")

        train, remainder = train_test_split(files, test_size=0.30, random_state=seed)
        validation, test = train_test_split(remainder, test_size=0.50, random_state=seed)
        splits = {"train": train, "val": validation, "test": test}
        for split, split_files in splits.items():
            copy_files(split_files, split, class_name, output)

        summary[class_name] = {name: len(items) for name, items in splits.items()}
        counts = summary[class_name]
        print(f"{class_name}: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic 70/15/15 PlantVillage splits."
    )
    parser.add_argument("--raw", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_dataset(args.raw.resolve(), args.output.resolve(), args.seed)


if __name__ == "__main__":
    main()

from pathlib import Path

from PIL import Image

from agromind_cv.config import TARGET_CLASSES
from agromind_cv.prepare import prepare_dataset


def test_prepare_dataset_creates_disjoint_splits(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "processed"
    for class_name in TARGET_CLASSES:
        folder = raw / class_name
        folder.mkdir(parents=True)
        for index in range(20):
            Image.new("RGB", (8, 8), (index, 20, 30)).save(folder / f"{index}.jpg")

    summary = prepare_dataset(raw, output)

    for class_name in TARGET_CLASSES:
        assert summary[class_name] == {"train": 14, "val": 3, "test": 3}
        names = [
            {path.name for path in (output / split / class_name).iterdir()}
            for split in ("train", "val", "test")
        ]
        assert names[0].isdisjoint(names[1])
        assert names[0].isdisjoint(names[2])
        assert names[1].isdisjoint(names[2])


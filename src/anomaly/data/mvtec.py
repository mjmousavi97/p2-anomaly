"""Read the MVTec AD folder layout into a list of samples."""

from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "test")


@dataclass(frozen=True)
class Sample:
    path: Path
    split: str
    defect_type: str
    label: int
    mask_path: Path | None


def list_samples(cat_dir: Path) -> list[Sample]:
    """Return every image of one category, in a fixed order."""
    cat_dir = Path(cat_dir)
    samples = []

    for split in SPLITS:
        split_dir = cat_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing folder: {split_dir}")

        for defect_dir in sorted(
            p for p in split_dir.iterdir() if p.is_dir()
        ):
            defect = defect_dir.name

            for img in sorted(defect_dir.glob("*.png")):
                if defect == "good":
                    samples.append(
                        Sample(img, split, defect, 0, None)
                    )
                    continue

                mask = (
                    cat_dir
                    / "ground_truth"
                    / defect
                    / f"{img.stem}_mask.png"
                )

                if not mask.exists():
                    raise FileNotFoundError(f"missing mask for {img}")

                samples.append(
                    Sample(img, split, defect, 1, mask)
                )

    return samples

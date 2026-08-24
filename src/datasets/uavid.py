from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

from .common import CommonClass, CommonSegmentationDataset, PairedTransform, RGB


# Official UAVid palette and its explicit semantic mapping.
UAVID_PALETTE: dict[str, RGB] = {
    "Clutter": (0, 0, 0),
    "Building": (128, 0, 0),
    "Road": (128, 64, 128),
    "Static_Car": (192, 0, 192),
    "Tree": (0, 128, 0),
    "LowVegetation": (128, 128, 0),
    "Human": (64, 64, 0),
    "Moving_Car": (64, 0, 128),
}
UAVID_CLASS_TO_COMMON: dict[str, CommonClass] = {
    "Clutter": CommonClass.OTHER,
    "Building": CommonClass.BUILDING,
    "Road": CommonClass.ROAD,
    "Static_Car": CommonClass.VEHICLE,
    "Tree": CommonClass.VEGETATION,
    "LowVegetation": CommonClass.VEGETATION,
    "Human": CommonClass.OTHER,
    "Moving_Car": CommonClass.VEHICLE,
}
UAVID_TO_COMMON: dict[RGB, CommonClass] = {
    UAVID_PALETTE[name]: common_id
    for name, common_id in UAVID_CLASS_TO_COMMON.items()
}


class UAVidDataset(CommonSegmentationDataset):

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        *,
        transforms: PairedTransform | None = None,
        image_transform: Callable[[object], object] | None = None,
        target_transform: Callable[[object], object] | None = None,
    ) -> None:
        super().__init__(
            transforms=transforms,
            image_transform=image_transform,
            target_transform=target_transform,
        )
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")

        root = Path(root).expanduser()
        split_dir = root / f"uavid_{split}"
        if not split_dir.is_dir():
            raise FileNotFoundError(f"UAVid split directory not found: {split_dir}")

        self.root = root
        self.split = split
        self._samples = self._find_samples(split_dir, labelled=split != "test")
        if not self._samples:
            raise RuntimeError(f"No UAVid images found below {split_dir}")

    @staticmethod
    def _find_samples(
        split_dir: Path,
        *,
        labelled: bool,
    ) -> list[tuple[Path, Path | None]]:
        samples: list[tuple[Path, Path | None]] = []
        for image_path in sorted(split_dir.glob("*/Images/*")):
            if not image_path.is_file():
                continue
            mask_path = image_path.parent.parent / "Labels" / image_path.name
            if labelled and not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")
            samples.append((image_path, mask_path if labelled else None))
        return samples

    @property
    def samples(self) -> list[tuple[Path, Path | None]]:
        return self._samples

    @property
    def colour_mapping(self) -> dict[RGB, CommonClass]:
        return UAVID_TO_COMMON

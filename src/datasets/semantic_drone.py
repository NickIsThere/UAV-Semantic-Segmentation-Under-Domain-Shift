from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from .common import CommonClass, CommonSegmentationDataset, PairedTransform, RGB


SEMANTIC_DRONE_PALETTE: dict[str, RGB] = {
    "unlabeled": (0, 0, 0),
    "paved-area": (128, 64, 128),
    "dirt": (130, 76, 0),
    "grass": (0, 102, 0),
    "gravel": (112, 103, 87),
    "water": (28, 42, 168),
    "rocks": (48, 41, 30),
    "pool": (0, 50, 89),
    "vegetation": (107, 142, 35),
    "roof": (70, 70, 70),
    "wall": (102, 102, 156),
    "window": (254, 228, 12),
    "door": (254, 148, 12),
    "fence": (190, 153, 153),
    "fence-pole": (153, 153, 153),
    "person": (255, 22, 96),
    "dog": (102, 51, 0),
    "car": (9, 143, 150),
    "bicycle": (119, 11, 32),
    "tree": (51, 51, 0),
    "bald-tree": (190, 250, 190),
    "ar-marker": (112, 150, 146),
    "obstacle": (2, 135, 115),
    "conflicting": (255, 0, 0),
}
SEMANTIC_DRONE_CLASS_TO_COMMON: dict[str, CommonClass] = {
    name: CommonClass.OTHER for name in SEMANTIC_DRONE_PALETTE
}
SEMANTIC_DRONE_CLASS_TO_COMMON.update(
    {
        "paved-area": CommonClass.ROAD,
        "grass": CommonClass.VEGETATION,
        "vegetation": CommonClass.VEGETATION,
        "roof": CommonClass.BUILDING,
        "wall": CommonClass.BUILDING,
        "car": CommonClass.VEHICLE,
        "tree": CommonClass.VEGETATION,
    }
)
SEMANTIC_DRONE_TO_COMMON: dict[RGB, CommonClass] = {
    SEMANTIC_DRONE_PALETTE[name]: common_id
    for name, common_id in SEMANTIC_DRONE_CLASS_TO_COMMON.items()
}


class SemanticDroneDataset(CommonSegmentationDataset):

    def __init__(
        self,
        root: str | Path,
        *,
        sample_ids: Sequence[str] | None = None,
        transforms: PairedTransform | None = None,
        image_transform: Callable[[object], object] | None = None,
        target_transform: Callable[[object], object] | None = None,
    ) -> None:
        super().__init__(
            transforms=transforms,
            image_transform=image_transform,
            target_transform=target_transform,
        )
        self.root = Path(root).expanduser()
        image_dir = self.root / "images"
        mask_dir = self.root / "labels" / "png"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(
                f"Expected 'images' and 'labels/png' below {self.root}"
            )

        image_by_stem = {
            path.stem: path for path in sorted(image_dir.iterdir()) if path.is_file()
        }
        ids = sorted(image_by_stem) if sample_ids is None else list(sample_ids)
        self._samples: list[tuple[Path, Path | None]] = []
        for sample_id in ids:
            sample_id = str(sample_id)
            image_path = image_by_stem.get(sample_id)
            if image_path is None:
                raise FileNotFoundError(f"Image for sample ID {sample_id!r} not found")
            mask_path = mask_dir / f"{sample_id}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Mask for sample ID {sample_id!r} not found")
            self._samples.append((image_path, mask_path))

        if not self._samples:
            raise RuntimeError(f"No Semantic Drone samples found below {self.root}")

    @property
    def samples(self) -> list[tuple[Path, Path | None]]:
        return self._samples

    @property
    def colour_mapping(self) -> dict[RGB, CommonClass]:
        return SEMANTIC_DRONE_TO_COMMON

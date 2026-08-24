from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CommonClass(IntEnum):
    """Class IDs shared by UAVid and Semantic Drone."""

    BUILDING = 0
    ROAD = 1
    VEGETATION = 2
    VEHICLE = 3
    OTHER = 4


COMMON_CLASS_NAMES: tuple[str, ...] = (
    "Building",
    "Road",
    "Vegetation",
    "Vehicle",
    "Other",
)

NUM_COMMON_CLASSES = len(COMMON_CLASS_NAMES)

RGB = tuple[int, int, int]
PairedTransform = Callable[[Image.Image, Image.Image], tuple[object, object]]


def _packed_rgb(rgb: RGB) -> int:
    return (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]


def remap_rgb_mask(
    mask: Image.Image,
    colour_mapping: Mapping[RGB, int | CommonClass],
) -> Image.Image:
    """Map an RGB colour mask into the common integer label space.

    Any colour absent from ``colour_mapping`` becomes ``CommonClass.OTHER``.
    Returning a PIL ``L`` image keeps the target compatible with paired PIL
    transforms such as crops and flips.
    """

    rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
    packed = (
        (rgb[..., 0].astype(np.uint32) << 16)
        | (rgb[..., 1].astype(np.uint32) << 8)
        | rgb[..., 2].astype(np.uint32)
    )
    remapped = np.full(packed.shape, CommonClass.OTHER, dtype=np.uint8)
    for colour, common_id in colour_mapping.items():
        remapped[packed == _packed_rgb(colour)] = int(common_id)
    return Image.fromarray(remapped, mode="L")


def _image_to_tensor(image: object) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image
        if tensor.ndim != 3:
            raise ValueError(f"Expected a 3D image tensor, got {tuple(tensor.shape)}")
        if tensor.shape[0] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        if tensor.dtype == torch.uint8:
            return tensor.float().div(255)
        return tensor.float()

    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError(f"Expected an HWC image, got shape {array.shape}")
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(255)


def _target_to_tensor(target: object) -> torch.Tensor:
    if isinstance(target, torch.Tensor):
        tensor = target
    else:
        tensor = torch.from_numpy(np.asarray(target).copy())

    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D target mask, got {tuple(tensor.shape)}")
    return tensor.long()


class CommonSegmentationDataset(Dataset):
    """Base class enforcing a common output contract for labelled images."""

    class_names = COMMON_CLASS_NAMES
    num_classes = NUM_COMMON_CLASSES
    ignore_index: int | None = None

    def __init__(
        self,
        *,
        transforms: PairedTransform | None = None,
        image_transform: Callable[[object], object] | None = None,
        target_transform: Callable[[object], object] | None = None,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.image_transform = image_transform
        self.target_transform = target_transform

    @property
    def samples(self) -> list[tuple[Path, Path | None]]:
        raise NotImplementedError

    @property
    def colour_mapping(self) -> Mapping[RGB, int | CommonClass]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        # UAVid's official test set has no public annotations. Returning only
        # the image keeps it usable for inference without inventing a target.
        if mask_path is None:
            if self.image_transform is not None:
                image = self.image_transform(image)
            return _image_to_tensor(image)

        with Image.open(mask_path) as source_mask:
            mask = remap_rgb_mask(source_mask, self.colour_mapping)
        if self.transforms is not None:
            transformed = self.transforms(image, mask)
            if not isinstance(transformed, (tuple, list)) or len(transformed) != 2:
                raise TypeError("transforms must return an (image, mask) pair")
            image, mask = transformed
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return _image_to_tensor(image), _target_to_tensor(mask)

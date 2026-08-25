from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Callable, Mapping

import jax.numpy as jnp
import torch
from PIL import Image
from torch.utils.data import Dataset


class CommonClass(IntEnum):
    """Class IDs shared by UAVid and Semantic Drone."""

    BUILDING = 0
    ROAD = 1
    VEGETATION = 2
    VEHICLE = 3
    HUMAN = 4
    OTHER = 5


COMMON_CLASS_NAMES: tuple[str, ...] = (
    "Building",
    "Road",
    "Vegetation",
    "Vehicle",
    "Human",
    "Other",
)

NUM_COMMON_CLASSES = len(COMMON_CLASS_NAMES)

RGB = tuple[int, int, int]
PairedTransform = Callable[[Image.Image, Image.Image], tuple[object, object]]


def _packed_rgb(rgb: RGB) -> int:
    return (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]


def _pil_to_jax(image: Image.Image) -> jnp.ndarray:
    """Decode a PIL image into a JAX array."""

    width, height = image.size
    channels = len(image.getbands())
    array = jnp.frombuffer(image.tobytes(), dtype=jnp.uint8)
    if channels == 1:
        return array.reshape(height, width)
    return array.reshape(height, width, channels)


def _jax_to_torch(array: jnp.ndarray) -> torch.Tensor:
    """Share a JAX array with PyTorch through DLPack."""

    return torch.utils.dlpack.from_dlpack(array)


def remap_rgb_mask(
    mask: Image.Image,
    colour_mapping: Mapping[RGB, int | CommonClass],
) -> Image.Image:
    """Map an RGB colour mask into the common integer label space.

    Any colour absent from ``colour_mapping`` becomes ``CommonClass.OTHER``.
    Returning a PIL ``L`` image keeps the target compatible with paired PIL
    transforms such as crops and flips.
    """

    rgb = _pil_to_jax(mask.convert("RGB"))
    packed = (
        (rgb[..., 0].astype(jnp.uint32) << 16)
        | (rgb[..., 1].astype(jnp.uint32) << 8)
        | rgb[..., 2].astype(jnp.uint32)
    )
    remapped = jnp.full(
        packed.shape,
        int(CommonClass.OTHER),
        dtype=jnp.uint8,
    )
    for colour, common_id in colour_mapping.items():
        if common_id == CommonClass.OTHER:
            continue
        remapped = jnp.where(
            packed == _packed_rgb(colour),
            jnp.uint8(common_id),
            remapped,
        )
    return Image.frombytes("L", mask.size, remapped.tobytes())


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

    if isinstance(image, Image.Image):
        array = _pil_to_jax(image)
    else:
        array = jnp.asarray(image)
    if array.ndim == 2:
        array = jnp.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError(f"Expected an HWC image, got shape {array.shape}")
    tensor = _jax_to_torch(array).permute(2, 0, 1).float()
    if array.dtype == jnp.uint8:
        tensor = tensor.div(255)
    return tensor


def _target_to_tensor(target: object) -> torch.Tensor:
    if isinstance(target, torch.Tensor):
        tensor = target
    else:
        if isinstance(target, Image.Image):
            array = _pil_to_jax(target)
        else:
            array = jnp.asarray(target)
        tensor = _jax_to_torch(array)

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

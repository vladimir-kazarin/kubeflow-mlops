"""Base64 PNG <-> uint8 tensor conversion shared by the predictor and the client script."""
from __future__ import annotations

import base64
import binascii
import io

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

CIFAR_SIDE = 32
MAX_IMAGE_B64_CHARS = 2_000_000  # ~1.5 MB decoded; far above any CIFAR-sized image
MAX_IMAGE_PIXELS = 4096 * 4096  # small PNGs can declare huge sizes (decompression bombs)


def encode_png_b64(image_hwc: np.ndarray) -> str:
    """Encode an (H, W, 3) uint8 array as a base64 PNG string."""
    buffer = io.BytesIO()
    Image.fromarray(image_hwc).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image_b64(data: str) -> torch.Tensor:
    """Decode a base64 image of any mode/size into a uint8 (3, 32, 32) RGB tensor."""
    if len(data) > MAX_IMAGE_B64_CHARS:
        raise ValueError(f"image_b64 is too large ({len(data)} chars; max {MAX_IMAGE_B64_CHARS})")
    try:
        # Image.open only reads the header; pixels are decoded later by load().
        image = Image.open(io.BytesIO(base64.b64decode(data, validate=True)))
    except (binascii.Error, ValueError, UnidentifiedImageError, Image.DecompressionBombError, OSError) as err:
        raise ValueError("image_b64 is not a valid base64-encoded image") from err
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError(f"image_b64 is too large ({image.width}x{image.height}; max {MAX_IMAGE_PIXELS} pixels)")
    try:
        image.load()
    except (ValueError, Image.DecompressionBombError, OSError) as err:
        raise ValueError("image_b64 is not a valid base64-encoded image") from err
    rgb = image.convert("RGB").resize((CIFAR_SIDE, CIFAR_SIDE), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(rgb, dtype=np.uint8)).permute(2, 0, 1).contiguous()

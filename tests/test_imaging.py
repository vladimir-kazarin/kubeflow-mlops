import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from cifar_mlops.imaging import MAX_IMAGE_B64_CHARS, decode_image_b64, encode_png_b64


def _pil_to_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_round_trip_preserves_a_cifar_image_exactly():
    image = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)

    decoded = decode_image_b64(encode_png_b64(image))

    assert torch.equal(decoded, torch.from_numpy(image).permute(2, 0, 1))


def test_decode_converts_grayscale_of_any_size_to_rgb_32x32():
    decoded = decode_image_b64(_pil_to_b64(Image.new("L", (64, 48), color=128)))

    assert decoded.shape == (3, 32, 32)
    assert decoded.dtype == torch.uint8


def test_decode_drops_alpha_channel():
    decoded = decode_image_b64(_pil_to_b64(Image.new("RGBA", (32, 32), color=(10, 20, 30, 40))))

    assert decoded.shape == (3, 32, 32)
    assert decoded[:, 0, 0].tolist() == [10, 20, 30]


@pytest.mark.parametrize("data", ["not base64!!", base64.b64encode(b"hello").decode("ascii"), ""])
def test_decode_rejects_data_that_is_not_an_image(data):
    with pytest.raises(ValueError, match="image_b64"):
        decode_image_b64(data)


def test_decode_rejects_small_payload_with_huge_pixel_dimensions():
    # A 1-bit 15000x15000 PNG is only ~36k base64 chars but would decode to ~225M pixels.
    bomb = _pil_to_b64(Image.new("1", (15_000, 15_000)))

    with pytest.raises(ValueError, match="image_b64"):
        decode_image_b64(bomb)


def test_decode_rejects_images_over_pixel_cap_below_pil_bomb_limit():
    # 5000x5000 = 25M pixels: PIL would decode it, but it is far beyond any useful input.
    bomb = _pil_to_b64(Image.new("1", (5_000, 5_000)))

    with pytest.raises(ValueError, match="image_b64"):
        decode_image_b64(bomb)


def test_decode_rejects_oversized_payload():
    with pytest.raises(ValueError, match="too large"):
        decode_image_b64("A" * (MAX_IMAGE_B64_CHARS + 4))

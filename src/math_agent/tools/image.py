"""图像辅助工具：尺寸/DPI 检查 + base64 编码（供多模态 LLM 输入）。"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class ImageInfo:
    width: int
    height: int
    dpi: tuple[float, float]


def inspect_image(path: str | Path) -> ImageInfo:
    with Image.open(path) as im:
        dpi = im.info.get("dpi", (72.0, 72.0))
        return ImageInfo(width=im.width, height=im.height, dpi=tuple(dpi))


def encode_image_to_data_url(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")

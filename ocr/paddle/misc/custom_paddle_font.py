from pathlib import Path

import PIL
from PIL import ImageFont

import os
from .. import logging
from ..cache import CACHE_DIR
from ..download import download
from ..flags import LOCAL_FONT_FILE_PATH

print(Path(CACHE_DIR) / "fonts")
print(os.listdir((Path(CACHE_DIR) / "fonts")))

def get_font_file_path(file_name: str) -> str:
    """
    Get the path of the font file.

    Returns:
    str: The path to the font file.
    """
    font_path = (Path(CACHE_DIR) / "fonts" / file_name).resolve().as_posix()
    if not Path(font_path).is_file():
        download(
            url=f"https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/{file_name}",
            save_path=font_path,
        )

    return font_path


def create_font(txt: str, sz: tuple, font_path: str) -> ImageFont:
    """
    Create a font object with specified size and path, adjusted to fit within the given image region.

    Parameters:
    txt (str): The text to be rendered with the font.
    sz (tuple): A tuple containing the height and width of an image region, used for font size.
    font_path (str): The path to the font file.

    Returns:
    ImageFont: An ImageFont object adjusted to fit within the given image region.
    """

    font_size = int(sz[1] * 0.8)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    if int(PIL.__version__.split(".")[0]) < 10:
        length = font.getsize(txt)[0]
    else:
        length = font.getlength(txt)

    if length > sz[0]:
        font_size = int(font_size * sz[0] / length)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font


def create_font_vertical(
    txt: str, sz: tuple, font_path: str, scale=1.2
) -> ImageFont.FreeTypeFont:
    n = len(txt) if len(txt) > 0 else 1
    base_font_size = int(sz[1] / n * 0.8 * scale)
    base_font_size = max(base_font_size, 10)
    font = ImageFont.truetype(font_path, base_font_size, encoding="utf-8")

    if int(PIL.__version__.split(".")[0]) < 10:
        max_char_width = max([font.getsize(c)[0] for c in txt])
    else:
        max_char_width = max([font.getlength(c) for c in txt])

    if max_char_width > sz[0]:
        new_size = int(base_font_size * sz[0] / max_char_width)
        new_size = max(new_size, 10)
        font = ImageFont.truetype(font_path, new_size, encoding="utf-8")

    return font


if Path(str(LOCAL_FONT_FILE_PATH)).is_file():
    logging.warning(
        f"Using the local font file(`{LOCAL_FONT_FILE_PATH}`) specified by `LOCAL_FONT_FILE_PATH`!"
    )
    PINGFANG_FONT_FILE_PATH = LOCAL_FONT_FILE_PATH
    SIMFANG_FONT_FILE_PATH = LOCAL_FONT_FILE_PATH
else:
    PINGFANG_FONT_FILE_PATH = get_font_file_path("PingFang-SC-Regular.ttf")
    SIMFANG_FONT_FILE_PATH = get_font_file_path("simfang.ttf")
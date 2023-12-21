#!/usr/bin/env python3

import freetype
import numpy

from dataclasses import dataclass
from argparse import ArgumentParser

SAMPLE_WIDTH = 12

CHAR_LOOKUP = [" ", ".", ":", "a", "@"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image-path", required=True)
    return parser.parse_args()


@dataclass
class CachedGlyph:
    bitmap: numpy.ndarray
    advance: float


class GlyphRenderer:
    def __init__(self):
        self.glyph_cache = {}
        self.face = freetype.Face("Hack-Regular.ttf")
        self.face.set_char_size(48 * 64)

    def render_char(self, char):
        cached = self.glyph_cache.get(char, None)
        if cached is None:
            self.face.load_char(char)
            bitmap = self.face.glyph.bitmap

            if bitmap.pixel_mode != freetype.FT_PIXEL_MODE_GRAY:
                raise RuntimeError("Unsupported pixel mode")

            if bitmap.num_grays != 256:
                raise RuntimeError("Unsupported num_grays")

            cached = CachedGlyph(
                bitmap=numpy.array(bitmap.buffer, dtype=numpy.uint8).reshape(
                    (bitmap.rows, bitmap.width)
                ),
                advance=self.face.glyph.advance.x / 64.0,
            )

            self.glyph_cache[char] = cached

        return cached

    def row_height(self):
        return freetype.FT_MulFix(self.face.units_per_EM, self.face.size.y_scale) / 64.0

    def char_width(self):
        return self.render_char("@").advance

    def char_aspect(self):
        return self.char_width() / self.row_height()

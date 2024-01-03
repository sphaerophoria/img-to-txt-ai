#!/usr/bin/env python3

import freetype
import numpy

from dataclasses import dataclass

SAMPLE_WIDTH = 12


@dataclass
class CachedGlyph:
    bitmap: numpy.ndarray
    x_bearing: float
    y_bearing: float
    advance: float


class GlyphRenderer:
    def __init__(self):
        self.glyph_cache = {}
        self.face = freetype.Face("Hack-Regular.ttf")
        self.face.set_char_size(36 * 64)

        char_code, index = self.face.get_first_char()
        while index != 0:
            try:
                self.render_char(char_code)
            except RuntimeError:
                print("Failed to render", char_code)

            char_code, index = self.face.get_next_char(char_code, index)

    def render_char(self, char):
        if isinstance(char, str):
            char = ord(char)
        cached = self.glyph_cache.get(char, None)
        if cached is None:
            self.face.load_char(char)
            bitmap = self.face.glyph.bitmap
            # FIXME: Sometimes this might not be an error?
            if len(bitmap.buffer) == 0 or self.face.glyph.advance.x == 0:
                raise RuntimeError("Cannot render")

            if bitmap.pixel_mode != freetype.FT_PIXEL_MODE_GRAY:
                raise RuntimeError("Unsupported pixel mode")

            if bitmap.num_grays != 256:
                raise RuntimeError("Unsupported num_grays")

            cached = CachedGlyph(
                bitmap=numpy.array(bitmap.buffer, dtype=numpy.uint8).reshape(
                    (bitmap.rows, bitmap.width)
                ),
                x_bearing=self.face.glyph.metrics.horiBearingX / 64.0,
                y_bearing=self.face.glyph.metrics.horiBearingY / 64.0,
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

#!/usr/bin/env python3

import freetype
import numpy
import torch

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class FreetypeGlyph:
    bitmap: numpy.ndarray
    x_bearing: float
    y_bearing: float
    advance: float


def _render_char(face, char):
    face.load_char(char)
    bitmap = face.glyph.bitmap
    # FIXME: Sometimes this might not be an error?
    if len(bitmap.buffer) == 0 or face.glyph.advance.x == 0:
        raise RuntimeError("Cannot render")

    if bitmap.pixel_mode != freetype.FT_PIXEL_MODE_GRAY:
        raise RuntimeError("Unsupported pixel mode")

    if bitmap.num_grays != 256:
        raise RuntimeError("Unsupported num_grays")

    return FreetypeGlyph(
        bitmap=numpy.array(bitmap.buffer, dtype=numpy.uint8).reshape(
            (bitmap.rows, bitmap.width)
        ),
        x_bearing=face.glyph.metrics.horiBearingX / 64.0,
        y_bearing=face.glyph.metrics.horiBearingY / 64.0,
        advance=face.glyph.advance.x / 64.0,
    )


def _get_row_height(face):
    return freetype.FT_MulFix(face.units_per_EM, face.size.y_scale) / 64.0


def _get_char_width(face):
    return _render_char(face, "@").advance


@dataclass
class _Point:
    x: int
    y: int


@dataclass
class _IndexMapping:
    source_start: _Point
    source_end: _Point
    dest_start: _Point
    dest_end: _Point


def _get_index_mapping_for_glyph(
    item: FreetypeGlyph, dest_height: int, dest_width: int
) -> _IndexMapping:
    """
    Find a mapping from bitmap[y1:y2, x1:x2] into dest[y3:y4, x3:x4] that...
    * Positions the glyph into the appropriate spot in dest
    * Ensures that we do not try to write outside of dest
    * Ensures that x2-x1 == x3-x4 (and the same for y)
    """
    source_height = item.bitmap.shape[0]
    source_width = item.bitmap.shape[1]

    # Destination is offset by the bearing
    dest_start_x = int(item.x_bearing)

    # Y bearing is from the baseline to the top of the image, which means it's
    # inverted from the top left
    # We also have to set the baseline as somewhere towards the middle of the
    # output as tails of characters can drift below the basleline

    # FIXME: hardcoded baseline is a problem, if the font size changes the
    # baseline offset needs to change as well
    BASELINE = dest_height / 6
    dest_start_y = int(dest_height - item.y_bearing - BASELINE)

    # Assuming no clipping, the end destination should completely be a function
    # of the dest start + source width
    dest_end_x = dest_start_x + source_width
    dest_end_y = dest_start_y + source_height

    source_start_x = 0
    source_start_y = 0
    source_end_x = source_width
    source_end_y = source_height

    # Clamp the destination to be in [0, width]/[0, height], and adjust the
    # source mappings to follow
    if dest_start_x < 0:
        source_start_x += -dest_start_x
        dest_start_x = 0

    if dest_start_y < 0:
        source_start_y += -dest_start_y
        dest_start_y = 0

    if dest_end_x < 0:
        source_end_x += -dest_end_x
        dest_end_x = 0

    if dest_end_y < 0:
        source_end_y += -dest_end_y
        dest_end_y = 0

    if dest_end_x > dest_width:
        source_end_x -= dest_end_x - dest_width
        dest_end_x = dest_width

    if dest_end_y > dest_height:
        source_end_y -= dest_end_y - dest_height
        dest_end_y = dest_height

    source_start = _Point(x=source_start_x, y=source_start_y)
    source_end = _Point(x=source_end_x, y=source_end_y)
    dest_start = _Point(x=dest_start_x, y=dest_start_y)
    dest_end = _Point(x=dest_end_x, y=dest_end_y)

    return _IndexMapping(
        source_start=source_start,
        source_end=source_end,
        dest_start=dest_start,
        dest_end=dest_end,
    )


def _make_font_face():
    face = freetype.Face("Hack-Regular.ttf")
    face.set_char_size(36 * 64)
    return face


def _render_glyphs(face) -> Tuple[List[int], List[FreetypeGlyph]]:
    glyphs = []
    char_codes = []
    char_code, index = face.get_first_char()
    while index != 0:
        try:
            rendered = _render_char(face, char_code)
            char_codes.append(char_code)
            glyphs.append(rendered)
        except RuntimeError:
            print("Failed to render", char_code)

        char_code, index = face.get_next_char(char_code, index)

    return char_codes, glyphs


def _insert_glyph_into_cache(glyph: FreetypeGlyph, glyph_slot: torch.Tensor):
    # FIXME: if character is too big, we need to handle that more gracefully
    index_mapping = _get_index_mapping_for_glyph(
        glyph, glyph_slot.shape[0], glyph_slot.shape[1]
    )

    if index_mapping.dest_end.x == 0 or index_mapping.dest_end.y == 0:
        print("0 sized glyph after index mapping")
        return

    glyph_slot[
        index_mapping.dest_start.y : index_mapping.dest_end.y,
        index_mapping.dest_start.x : index_mapping.dest_end.x,
    ] = torch.tensor(
        glyph.bitmap[
            index_mapping.source_start.y : index_mapping.source_end.y,
            index_mapping.source_start.x : index_mapping.source_end.x,
        ],
        device=glyph_slot.device,
    )


def _glyph_cache_from_glyphs(
    glyphs: List[FreetypeGlyph], width: int, height: int, device="cpu"
) -> torch.Tensor:
    glyph_cache = torch.zeros(
        (len(glyphs), height, width), dtype=torch.uint8, device=device
    )

    for idx, glyph in enumerate(glyphs):
        try:
            _insert_glyph_into_cache(glyph, glyph_cache[idx])
        except Exception as e:
            print("failure to cache glyph: {}".format(e))
            pass

    return glyph_cache


def generate_glyph_cache(device="cpu") -> Tuple[List[int], torch.Tensor]:
    """
    returns tensor of shape (c, h, w) where c is the number of glyphs, h is height, and w is width of the generated bitmaps
    """
    face = _make_font_face()
    char_codes, glyphs = _render_glyphs(face)
    char_height = int(_get_row_height(face))
    char_width = int(_get_char_width(face))

    glyph_cache = _glyph_cache_from_glyphs(
        glyphs, char_width, char_height, device=device
    )
    return char_codes, glyph_cache


class TextRenderer:
    def __init__(self, glyph_cache):
        self.glyph_cache = glyph_cache

    def render(self, labels, chars_per_row):
        chars_per_row = int(chars_per_row)
        chars_per_col = int(len(labels) / chars_per_row)

        _, glyph_height, glyph_width = self.glyph_cache.shape

        output = torch.zeros(
            (glyph_height * chars_per_col, glyph_width * chars_per_row),
            device=self.glyph_cache.device,
            dtype=self.glyph_cache.dtype,
        )

        for i, label in enumerate(labels):
            char_x = i % chars_per_row
            char_y = int(i / chars_per_row)
            start_x = char_x * glyph_width
            end_x = start_x + glyph_width
            start_y = char_y * glyph_height
            end_y = start_y + glyph_height
            output[start_y:end_y, start_x:end_x] = self.glyph_cache[label]

        return output

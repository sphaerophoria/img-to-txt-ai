import math
import numpy
import torch

from lib.img_to_text import GlyphRenderer, CachedGlyph
from PIL import Image
from typing import Tuple
from dataclasses import dataclass


def num_samples_for_img(
    img: torch.Tensor, sample_width: int, sample_height: int
) -> Tuple[int, int]:
    """
    Returns how many samples there are in each dimension for the given img tensor

    returns (y, x) number of samples
    """
    img_height = img.shape[0]
    img_width = img.shape[1]
    num_y_samples = int(img_height / sample_height)
    num_x_samples = int(img_width / sample_width)
    return (num_y_samples, num_x_samples)


def extract_w_h_samples(
    sample_width: int, sample_height: int, img: torch.Tensor
) -> torch.Tensor:
    """
    Extracts samples in a grid for the entire image. Samples are arranged in row major order

    img: (h, w)
    return: (n, h, w) where n is h/sample_height * w/sample_width number of samples
    """
    num_y_samples, num_x_samples = num_samples_for_img(img, sample_width, sample_height)

    y_indices = torch.arange(
        0, num_y_samples * sample_height, dtype=torch.int64
    ).reshape((num_y_samples, sample_height))
    x_indices = torch.arange(
        0, num_x_samples * sample_width, dtype=torch.int64
    ).reshape((num_x_samples, sample_width))

    samples = img[y_indices][:, :, x_indices]
    samples = samples.transpose(1, 2)
    samples = samples.reshape((-1, samples.shape[2], samples.shape[3]))
    return samples


def sample_to_sample_for_comparison(x, y, sample, img, img_for_comparison):
    img_for_comparison_x = int(img_for_comparison.shape[1] / img.shape[1] * x)
    img_for_comparison_y = int(img_for_comparison.shape[0] / img.shape[0] * y)
    img_for_comparison_width = int(
        img_for_comparison.shape[1] / img.shape[1] * sample.shape[1]
    )
    img_for_comparison_height = int(
        img_for_comparison.shape[0] / img.shape[0] * sample.shape[0]
    )

    return img_for_comparison[
        img_for_comparison_y : img_for_comparison_y + img_for_comparison_height,
        img_for_comparison_x : img_for_comparison_x + img_for_comparison_width,
    ]


def get_samples_and_labels_for_img(
    sample_width: int,
    sample_height: int,
    img: torch.Tensor,
    img_for_comparison: torch.Tensor,
    glyph_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts samples + labels for the given img/img_for_comparison pair
    """
    data = extract_w_h_samples(sample_width, sample_height, img)

    num_y_samples, num_x_samples = num_samples_for_img(img, sample_width, sample_height)

    img_for_comp_sample_height = int(img_for_comparison.shape[0] / num_y_samples)
    img_for_comp_sample_width = int(img_for_comparison.shape[1] / num_x_samples)
    data_for_comparison = extract_w_h_samples(
        img_for_comp_sample_width, img_for_comp_sample_height, img_for_comparison
    )

    assert (
        data.shape[0] == data_for_comparison.shape[0]
    ), "Data and upsampled data did not end with same shape"

    labels = get_labels_for_samples(data_for_comparison, glyph_cache)

    return data, labels


@dataclass
class Point:
    x: int
    y: int


@dataclass
class IndexMapping:
    source_start: Point
    source_end: Point
    dest_start: Point
    dest_end: Point


def get_index_mapping_for_glyph(
    item: CachedGlyph, dest_height: int, dest_width: int
) -> IndexMapping:
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
    BASELINE = 4
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

    source_start = Point(x=source_start_x, y=source_start_y)
    source_end = Point(x=source_end_x, y=source_end_y)
    dest_start = Point(x=dest_start_x, y=dest_start_y)
    dest_end = Point(x=dest_end_x, y=dest_end_y)

    return IndexMapping(
        source_start=source_start,
        source_end=source_end,
        dest_start=dest_start,
        dest_end=dest_end,
    )


def generate_glyph_cache(glyph_renderer: GlyphRenderer, device="cpu") -> torch.Tensor:
    """
    returns tensor of shape (c, h, w) where c is the number of glyphs, h is height, and w is width of the generated bitmaps
    """
    char_height = int(glyph_renderer.row_height())
    char_width = int(glyph_renderer.char_width())
    item_tensor = torch.zeros(
        (len(glyph_renderer.glyph_cache), char_height, char_width), device=device
    )

    for idx, char in enumerate(glyph_renderer.glyph_cache):
        try:
            item = glyph_renderer.render_char(char)
            # FIXME: if character is too big, we need to handle that more gracefully
            index_mapping = get_index_mapping_for_glyph(
                item, item_tensor.shape[1], item_tensor.shape[2]
            )

            if index_mapping.dest_end.x == 0 or index_mapping.dest_end.y == 0:
                print("{} has 0 sized glyph after index mapping".format(chr(char)))

            item_tensor[
                idx,
                index_mapping.dest_start.y : index_mapping.dest_end.y,
                index_mapping.dest_start.x : index_mapping.dest_end.x,
            ] = torch.tensor(
                item.bitmap[
                    index_mapping.source_start.y : index_mapping.source_end.y,
                    index_mapping.source_start.x : index_mapping.source_end.x,
                ],
                device=device,
            )
        except Exception as e:
            print("failure to cache glyph: {}".format(e))
            pass

    # Normalize for sane comparison with sample images
    return item_tensor / 255.0


def compute_glyph_diff_scores_for_samples(
    samples_for_comparison: torch.Tensor, glyph_cache: torch.Tensor
) -> torch.Tensor:
    """
    Where n = number of samples
          c = number of possible characters
          h = height
          w = width
          s = score

    Returns (n, c, s), scores are the difference between a glyph and the sample

    samples_for_comparison: tensor of (n, h, w)
    glyph_cache: tensor of (c, h, w)
    """

    n = samples_for_comparison.shape[0]
    c = glyph_cache.shape[0]
    h = glyph_cache.shape[1]
    w = glyph_cache.shape[2]

    # (n, c, h, w)
    samples_for_comparison = samples_for_comparison.repeat_interleave(c, dim=0)
    # (n * c, h, w)
    samples_for_comparison = samples_for_comparison.reshape((n, c, h, w))
    # (n, c, h, w)
    scores = (samples_for_comparison - glyph_cache).abs()
    # (n, c, s)
    scores = scores.sum(dim=(2, 3))
    return scores


def compute_brightness_scores_for_samples(
    samples_for_comparison: torch.Tensor, glyph_cache: torch.Tensor
) -> torch.Tensor:
    """
    Where n = number of samples
          c = number of possible characters
          h = height
          w = width
          s = score

    Returns (n, c, s)
    samples_for_comparison: tensor of (n, h, w)
    glyph_cache: tensor of (c, h, w)
    """

    n = samples_for_comparison.shape[0]
    c = glyph_cache.shape[0]

    sample_sums = samples_for_comparison.sum(dim=(1, 2))
    sample_sums = sample_sums.reshape((n, 1)).repeat((1, c))
    glyph_sums = glyph_cache.sum(dim=(1, 2))
    return (sample_sums - glyph_sums).abs()


def get_labels_for_samples(
    samples_for_comparison: torch.Tensor,
    glyph_cache: torch.Tensor,
    scoring_fn=compute_brightness_scores_for_samples,
) -> torch.Tensor:
    """
    Returns which cached index is the best for each sample

    Where n = number of samples
          c = number of possible characters
          h = height
          w = width
    samples_for_comparison: tensor of (n, h, w)
    glyph_cache: tensor of (c, h, w)
    """

    labels = torch.zeros(
        samples_for_comparison.shape[0],
        device=samples_for_comparison.device,
        dtype=torch.int64,
    )

    # Batch the score calculation to avoid OOMing
    BATCH_SIZE = 100
    for i in range(0, labels.shape[0], BATCH_SIZE):
        scores = scoring_fn(samples_for_comparison[i : i + BATCH_SIZE], glyph_cache)
        labels[i : i + BATCH_SIZE] = scores.min(dim=1)[1]

    return labels


class ImgSampler:
    def __init__(self, w, h, glyph_renderer, img_path, device="cpu"):
        self.sample_width = w
        self.sample_height = h
        self.glyph_renderer = glyph_renderer
        img_pil = Image.open(img_path).convert(mode="L")
        self.img = torch.tensor(numpy.array(img_pil), device=device) / 255.0

        img_for_comparison = img_pil.resize(
            (
                int(
                    math.ceil(self.img.shape[1] / w * self.glyph_renderer.char_width())
                ),
                int(
                    math.ceil(self.img.shape[0] / h * self.glyph_renderer.row_height())
                ),
            )
        )
        self.img_for_comparison = (
            torch.tensor(numpy.array(img_for_comparison), device=device) / 255.0
        )
        self.glyph_cache = generate_glyph_cache(self.glyph_renderer, device)
        self.char_width = int(glyph_renderer.char_width())
        self.char_height = int(glyph_renderer.row_height())
        self.glyph_renderer = glyph_renderer

    def get_samples(self):
        return get_samples_and_labels_for_img(
            self.sample_width,
            self.sample_height,
            self.img,
            self.img_for_comparison,
            self.glyph_cache,
        )

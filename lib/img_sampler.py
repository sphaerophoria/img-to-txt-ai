import math
import numpy
import torch

from lib.img_to_text import GlyphRenderer, CHAR_LOOKUP
from PIL import Image
from typing import Tuple


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


def generate_glyph_cache(glyph_renderer: GlyphRenderer) -> torch.Tensor:
    """
    returns tensor of shape (c, h, w) where c is the number of glyphs, h is height, and w is width of the generated bitmaps
    """
    char_height = int(glyph_renderer.row_height())
    char_width = int(glyph_renderer.char_width())
    item_tensor = torch.zeros((len(CHAR_LOOKUP), char_height, char_width))
    for idx, char in enumerate(CHAR_LOOKUP):
        item = glyph_renderer.render_char(char)
        item_tensor[
            idx, 0 : item.bitmap.shape[0], 0 : item.bitmap.shape[1]
        ] = torch.tensor(item.bitmap)

    return item_tensor


def get_labels_for_samples(
    samples_for_comparison: torch.Tensor, glyph_cache: torch.Tensor
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

    c = glyph_cache.shape[0]
    h = glyph_cache.shape[1]
    w = glyph_cache.shape[2]

    # (n, sum)
    summed = samples_for_comparison.sum(dim=(1, 2))
    # (n, label)
    labels = (summed / (h * w + 1) * c).to(torch.int64)
    return labels


class ImgSampler:
    def __init__(self, w, h, glyph_renderer, img_path):
        self.sample_width = w
        self.sample_height = h
        self.glyph_renderer = glyph_renderer
        img_pil = Image.open(img_path).convert(mode="L")
        self.img = torch.tensor(numpy.array(img_pil)) / 255.0

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
        self.img_for_comparison = torch.tensor(numpy.array(img_for_comparison)) / 255.0
        self.glyph_cache = generate_glyph_cache(self.glyph_renderer)
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

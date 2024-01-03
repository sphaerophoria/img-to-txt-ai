import numpy
import torch
import torchvision.transforms.functional as transforms

from PIL import Image
from typing import Tuple
from pathlib import Path
from torch.nn.functional import conv2d


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
    glyph_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts samples + labels for the given img/img_for_comparison pair
    """
    assert img.dtype == torch.uint8

    data = extract_w_h_samples(sample_width, sample_height, img)

    _, h, w = glyph_cache.shape

    data_for_comparison = transforms.resize(data, [h, w])

    assert (
        data.shape[0] == data_for_comparison.shape[0]
    ), "Data and upsampled data did not end with same shape"

    labels = get_labels_for_samples(data_for_comparison, glyph_cache)

    return data / 255.0, labels


def blur_glyph_cache(glyph_cache: torch.Tensor) -> torch.Tensor:
    # FIXME: Matching old behavior
    glyph_cache = glyph_cache / 255.0

    c = glyph_cache.shape[0]
    h = glyph_cache.shape[1]
    w = glyph_cache.shape[2]

    # 3x3 gaussian blur with steep falloff
    kernel = torch.tensor(
        [
            [1, 5, 1],
            [5, 25, 5],
            [1, 5, 1],
        ],
        dtype=torch.float32,
        device=glyph_cache.device,
    )
    kernel /= kernel.sum()
    kernel = kernel.reshape(1, 1, 3, 3)

    blurred_glyph_cache = conv2d(
        glyph_cache.reshape((c, 1, h, w)), kernel, padding=1
    ).reshape((c, h, w))
    return (blurred_glyph_cache * 255.0).to(torch.uint8)


def compute_glyph_diff_scores_for_samples(
    samples_for_comparison: torch.Tensor, glyph_cache: torch.Tensor
) -> torch.Tensor:
    """
    Where n = number of samples
          c = number of possible characters
          h = height
          w = width

    Returns (n, c), scores are the difference between a glyph and the sample

    samples_for_comparison: tensor of (n, h, w)
    glyph_cache: tensor of (c, h, w)
    """

    assert samples_for_comparison.dtype == torch.uint8
    assert glyph_cache.dtype == torch.uint8

    n = samples_for_comparison.shape[0]
    c = glyph_cache.shape[0]
    h = glyph_cache.shape[1]
    w = glyph_cache.shape[2]

    h = int(h / 2)
    w = int(w / 2)
    samples_for_comparison = transforms.resize(
        samples_for_comparison, [h, w], antialias=True
    )
    glyph_cache = transforms.resize(glyph_cache, [h, w], antialias=True)

    samples_for_comparison = samples_for_comparison.reshape((n, 1, h, w))
    glyph_cache = glyph_cache.reshape((1, c, h, w))

    # (n, c, h, w)
    scores = samples_for_comparison.to(torch.int16) - glyph_cache

    scores.abs_()
    assert int(scores.max()) < 256
    RIGHT_SHIFT = 1
    scores.bitwise_right_shift_(RIGHT_SHIFT)
    # Make sure that the max sum can fit into our output
    assert ((255 >> RIGHT_SHIFT) * h * w) < (1 << 15)
    # (n, c)
    scores = scores.sum(dim=(2, 3), dtype=torch.int16)
    return scores


def compute_brightness_scores_for_samples(
    samples_for_comparison: torch.Tensor, glyph_cache: torch.Tensor
) -> torch.Tensor:
    """
    Where n = number of samples
          c = number of possible characters
          h = height
          w = width

    Returns (n, c), scores are the difference between the overall brightness
    of the glyph, and the overall brightness of the sample

    samples_for_comparison: tensor of (n, h, w)
    glyph_cache: tensor of (c, h, w)
    """

    n = samples_for_comparison.shape[0]
    c = glyph_cache.shape[0]

    sample_sums = samples_for_comparison.sum(dim=(1, 2))
    sample_sums = sample_sums.reshape((-1, 1, 1))
    glyph_sums = glyph_cache.sum(dim=(1, 2)).reshape((1, -1, 1))
    ret = (sample_sums - glyph_sums).abs().reshape((n, c))
    return ret


def compute_glyph_diff_with_brightness_scores_for_samples(
    samples_for_comparison: torch.Tensor, glyph_cache: torch.Tensor
) -> torch.Tensor:
    """
    Where n = number of samples
          c = number of possible characters
          h = height
          w = width

    Returns (n, c), scores are a combination of a gaussian blurred glyph
    diffed with the source image, and the difference between the brightness of
    the glyph and the sample

    samples_for_comparison: tensor of (n, h, w)
    glyph_cache: tensor of (c, h, w)
    """
    scores = compute_glyph_diff_scores_for_samples(
        samples_for_comparison, glyph_cache
    ).to(torch.float32)
    scores += compute_brightness_scores_for_samples(
        samples_for_comparison, glyph_cache
    ) * torch.tensor(0.07, dtype=torch.float32)
    return scores


def get_labels_for_samples(
    samples_for_comparison: torch.Tensor,
    glyph_cache: torch.Tensor,
    scoring_fn=compute_glyph_diff_with_brightness_scores_for_samples,
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

    # Batch the score calculation to avoid OOMing
    scores = scoring_fn(samples_for_comparison, glyph_cache)
    labels = scores.min(dim=1)[1]

    return labels


class NestedDirImageLoader:
    def __init__(
        self,
        image_dir,
        sample_width: int,
        sample_height: int,
        glyph_cache: torch.Tensor,
        device="cpu",
    ):
        image_dir = Path(image_dir)
        self.images = list(image_dir.glob("**/*.jpg"))
        self.sample_width = sample_width
        self.sample_height = sample_height
        self.blurred_glyph_cache = blur_glyph_cache(glyph_cache)
        self.device = device

    def num_images(self):
        return len(self.images)

    def get_sample(self, i):
        img = Image.open(self.images[i]).convert(mode="L")

        img = torch.tensor(numpy.array(img), device=self.device)
        ret = get_samples_and_labels_for_img(
            self.sample_width,
            self.sample_height,
            img,
            self.blurred_glyph_cache,
        )

        return ret

    def get_samples(self, n):
        assert n > 0
        sample_ids = torch.randint(0, len(self.images), (n,))

        datas = []
        labels = []

        for i in sample_ids:
            # (n_1, h, w), (n_1,s)
            data, sample_labels = self.get_sample(i)
            datas.append(data)
            labels.append(sample_labels)

        return torch.cat(datas), torch.cat(labels)

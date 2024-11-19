import pytest
from pathlib import Path
from Postdam.tiff_dataset import TiffDataset
from source.dataset_classes import load_classes_as_numpy
import torchvision.transforms as transforms
import numpy as np


@pytest.fixture
def raw_image_dir():
    return Path("/home/armolina/Qsync/images/multispectral_segmentation/Potsdam/2_Ortho_RGB")


@pytest.fixture
def image_label_dir():
    return Path("/home/armolina/Qsync/images/multispectral_segmentation/Potsdam/5_Labels_all")


@pytest.fixture
def label_classes_path():
    return Path(
        "/home/armolina/aaMain/workspace/multispectral_segmentation/SegNet_PyTorch/Postdam/postdam_classes.json")


@pytest.fixture
def labeled_image():
    return np.array([[[255, 255, 255],      [0, 0, 255],  [0, 255, 255], [0, 0, 255]],
                     [[0, 255, 0],    [255, 255, 0],  [255, 0, 0], [255, 255, 0]],
                     [[255, 255, 255],      [0, 0, 255],  [0, 255, 255], [0, 0, 255]]])


@pytest.fixture
def label_tensor():
    return np.array([
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
        [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0]],
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
    ])


def test_get_label(raw_image_dir: Path, image_label_dir: Path):
    dataset = TiffDataset(raw_image_dir, image_label_dir)
    labeled_image = dataset._get_labeled_image(raw_image_dir.joinpath("top_potsdam_2_13_RGB.tif"))
    assert labeled_image == image_label_dir.joinpath("top_potsdam_2_13_label.tif")


def test_label_to_one_hot(
        raw_image_dir: Path, image_label_dir: Path, label_classes_path: Path,
        labeled_image: np.ndarray, label_tensor: np.ndarray):
    numpy_classes = load_classes_as_numpy(label_classes_path)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = TiffDataset(raw_image_dir, image_label_dir, transform, numpy_classes)
    labeled_output = dataset.label_to_one_hot_encoder(labeled_image)
    assert np.array_equal(labeled_output, label_tensor)

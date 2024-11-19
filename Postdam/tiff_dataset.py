import torch
import torchvision.transforms as transforms
import numpy as np
import csv

from torch.utils.data import Dataset
from torchmetrics.functional import jaccard_index, precision, recall, stat_scores
from skimage import io
from pathlib import Path
from typing import Optional, Tuple

import os
import time
import re


# Read the CSV and return a list of paths (ignoring the first row)
def get_paths_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        paths = [Path(row[0]) for row in reader]
    return paths


class TiffDataset(Dataset):

    def __init__(self, raw_dir: Path, label_dir: Path,
                 transform: Optional[transforms.Compose] = None, label_classes: Optional[np.ndarray] = None):

        # raw_dir: (directory) Folder directory of raw input image files
        # lbl_dir: (directory) Folder directory of labeled image files

        self.image_dirs = raw_dir
        self.labels_dirs = label_dir
        self.transform = transform
        self.label_classes = label_classes
        self.list_img = get_paths_from_csv(self.image_dirs)[:100]
        self.list_label = get_paths_from_csv(self.labels_dirs)[:100]

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_raw_path = self.list_img[idx]
        label_path = self.list_label[idx]

        image_raw = io.imread(img_raw_path)
        image_label = io.imread(label_path)
        try:
            label = self.label_to_one_hot_encoder(image_label)
        except Exception as e:
            print(img_raw_path, label_path)
            print(image_label.shape)
            print(e)
            raise e

        if self.transform:
            image_raw = self.transform(image_raw)
            label = self.transform(label)

        return image_raw, label

    def label_to_one_hot_encoder(self, labeled_image):
        output = np.zeros((labeled_image.shape[0], labeled_image.shape[1], self.label_classes.shape[0]))

        for c, label_class in enumerate(self.label_classes):
            label = np.nanmin(label_class == labeled_image, axis=2)
            output[:, :, c] = label

        return output

    def classify(self, image):
        output = np.zeros_like(image, dtype=np.int)

        # Threshold pixels such that (<= threshold is pavement surface) & (> threshold is pavement crack)
        output[image <= self.pixel_value_threshold] = 0
        output[image > self.pixel_value_threshold] = 1

        return output

    def compute_pavement_crack_area(self, pred, as_ratio=False):
        crack_pixels = torch.where(pred == 1.0)[0].shape[0]
        if as_ratio:
            total_pixels = pred.nelement()
            return crack_pixels / total_pixels

        return crack_pixels

    def compute_precision(self, pred, target, threshold=0.5):
        # Precision: TP / (TP + FP)

        return precision(pred, target, average='none', mdmc_average='samplewise', ignore_index=None,
                         num_classes=2, threshold=0.5, top_k=None, multiclass=None)

    def compute_recall(self, pred, target, threshold=0.5):
        # Recall: TP / (TP + FN)

        return recall(pred, target, average='none', mdmc_average='samplewise', ignore_index=None,
                      num_classes=2, threshold=0.5, top_k=None, multiclass=None)

    def compute_m_iou(self, pred, target, threshold=0.5):
        # Mean Intersection over Union (mIoU) a.k.a. Jaccard Index

        return jaccard_index(pred, target, 2, ignore_index=None, absent_score=0.0,
                             threshold=threshold, average='none')

    def compute_balanced_class_accuracy(self, pred, target):
        """
          Balanced class accuracy = (Sensitivity + Specificity) / 2
                                  = ((TP / (TP + FN)) + TN / (TN + FP)) / 2
        """
        scores = stat_scores(pred, target, reduce='macro', num_classes=2,
                             mdmc_reduce='samplewise')  # [[[tp, fp, tn, fn, sup]]]

        tp = scores[:, :, 0]
        fp = scores[:, :, 1]
        tn = scores[:, :, 2]
        fn = scores[:, :, 3]
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        return torch.mean((sensitivity + specificity) / 2, dim=0)[0]

import SegNet
from Pavements import Pavements
from tiff_dataset import TiffDataset

import os

import json
import numpy as np
import torch
import torchvision.transforms as transforms

from pathlib import Path
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Union
from torchvision.utils import save_image

from itertools import product


def load_json(json_path: Union[str, Path]) -> list:
    with open(json_path, 'r') as f:
        class_dict = OrderedDict(json.load(f))
        return list(class_dict.values())


def build_color_map(labels_class_file: Path):
    # Load the colors in alphabetical order and transform them to a list.
    color_list = load_json(labels_class_file)
    color_map = torch.tensor(color_list)

    print("Map of class to color: ")
    for class_ind, color in enumerate(color_map):
        print(f"Class: {class_ind + 1}, RGB Color: {color}")

    print("************************")

    return color_map


def load_model_json():
    # batch_size: Training batch-size
    # epochs: No. of epochs to run
    # lr: Optimizer learning rate
    # momentum: SGD momentum
    # no_cuda: Disables CUDA training (**To be implemented)
    # seed: Random seed
    # in-chn: Input image channels (3 for RGB, 4 for RGB-A)
    # out-chn: Output channels/semantic classes (2 for Pavements dataset)

    with open('./model.json') as f:
        model_json = json.load(f)

    return model_json


def load(model, weight_fn: Path):
    assert weight_fn.is_file(), f"{weight_fn} is not a file."

    checkpoint = torch.load(weight_fn)
    epoch = checkpoint['epoch']
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    print(f"Checkpoint is loaded at {weight_fn} | Epochs: {epoch}")


def check_input_arguments(args: Namespace):
    image_paths = Path(args.image_paths)
    if not image_paths.is_file():
        raise FileNotFoundError(f"{image_paths} does not exist or its invalid")

    labels_paths = Path(args.labels_paths)
    if not labels_paths.is_file():
        raise FileNotFoundError(f"{labels_paths} does not exist or its invalid")

    label_color_path = Path(args.colors_path)
    if not label_color_path.exists():
        raise FileNotFoundError(f"{label_color_path} does not exist")

    output_folder = Path(args.output_dir)
    if not output_folder.exists():
        output_folder.mkdir()
    
    return image_paths, labels_paths, label_color_path, output_folder


def main(args):
    cuda_available = torch.cuda.is_available()
    model_json = load_model_json()
    image_paths, labels_paths, labels_colors, output_folder = check_input_arguments(args)
    class_labels = np.array(load_json(args.labels_class))

    # initialize model in evaluation mode
    model = SegNet.SegNet(in_chn=model_json['in_chn'], out_chn=model_json['out_chn'],
                          BN_momentum=model_json['bn_momentum'])
    if cuda_available:
        model.cuda()
    model.eval()

    # load pretrained weights
    load(model, Path(args.weight_fn))

    # load test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = TiffDataset(image_paths, labels_paths, transform, class_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)

    # init metrics aggregation
    num_images = 0
    sum_precision = torch.zeros(2)
    sum_recall = torch.zeros(2)
    sum_m_iou = torch.zeros(2)
    sum_balanced_class_accuracy = 0.0

    # run evaluation
    for i, data in enumerate(dataloader):
        images = data[0]

        if cuda_available:
            images = images.cuda()

        res = model(images)
        res = torch.argmax(res, dim=1).type(torch.long)  # pixel-wise probs squashed to pixel-wise labels
        lbl = data[1].type(torch.long)

        if cuda_available:
            lbl = lbl.cuda()

        for n in range(res.shape[0]):  # loop over each image
            image_name = "img_{}_{}.png".format(i, n)
            input_image = images[n]
            lbl_image = class_labels[lbl[n]].permute(2, 0, 1).to(torch.float).div(255.0)
            res_image = class_labels[res[n]].permute(2, 0, 1).to(torch.float).div(
                255.0)  # transpose back to C, H, W, normalize to (0.0, 1.0)
            if cuda_available:
                input_image = input_image.cuda()
                lbl_image = lbl_image.cuda()
                res_image = res_image.cuda()

            compare_image = torch.cat((input_image, lbl_image, res_image), dim=2)

            if cuda_available:
                compare_image = compare_image.cuda()
            save_image(compare_image, os.path.join(args.res_dir, image_name))

            # Compute metrics per image & accumulate
            precision = dataset.compute_precision(res, lbl).to('cpu')
            recall = dataset.compute_recall(res, lbl).to('cpu')
            m_iou = dataset.compute_m_iou(res, lbl).to('cpu')
            balanced_class_accuracy = dataset.compute_balanced_class_accuracy(res, lbl).to('cpu')
            pavement_crack_area = dataset.compute_pavement_crack_area(res, as_ratio=True) * 100.0
            print("{} | Precision: {} | Recall: {} | IoU: {} | Balanced Class Accuracy: {} | Crack Area: {:.6f}%"
                  .format(image_name, precision, recall, m_iou, balanced_class_accuracy, pavement_crack_area))

            num_images += 1
            sum_precision += precision
            sum_recall += recall
            sum_m_iou += m_iou
            sum_balanced_class_accuracy += balanced_class_accuracy

    print("\nEvaluation complete. {} segmented images saved at {}\n".format(num_images, args.res_dir))

    # Compute global metrics & present
    print("Averaged metrics | Precision: {} | Recall: {} | IoU: {} | Balanced Class Accuracy: {}"
          .format(*[x / num_images for x in [sum_precision, sum_recall, sum_m_iou, sum_balanced_class_accuracy]]))


if __name__ == "__main__":
    parser = ArgumentParser()

    # FORMAT DIRECTORIES
    parser.add_argument("-i", "--image_paths", type=str, help="Path to the list of raw images")
    parser.add_argument("-l", "--labels_paths", type=str, help="Path to the list of labels")
    parser.add_argument("--weight_fn", required=True, type=str, help="Path: Trained weights")
    parser.add_argument("--output_dir", type=str,
                        help="Path to the folder in which the images will be saved")
    parser.add_argument("-lc", "--colors_path", type=str,
                        help="Path to the file with the json that stores the name-color pairs")

    arguments = parser.parse_args()

    main(arguments)

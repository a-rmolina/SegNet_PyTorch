import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
import numpy as np
import json

from collections import OrderedDict
from pathlib import Path
from typing import Union
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser, Namespace

from SegNet import SegNet
from tiff_dataset import TiffDataset


def load_json(json_path: Union[str, Path]) -> list:
    with open(json_path, 'r') as f:
        class_dict = OrderedDict(json.load(f))
        return list(class_dict.values())


def parse_arguments() -> Namespace:
    parser = ArgumentParser()

    #FORMAT DIRECTORIES
    parser.add_argument("-i", "--input_raw_dir", type=str, help="Path to the list of raw images")
    parser.add_argument("-l", "--input_label_dir", type=str, help="Path to the list of labels")
    parser.add_argument("-lc", "--labels_class", type=str, help="Path to labels class")
    parser.add_argument("-w", "--weight-fn", type=str, help="Path: Trained weights", default=None)
    parser.add_argument("--tb_logs_dir", type=str, help="Directory: Logs for tensorboard")

    return parser.parse_args()


def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"Checkpoint saved at {path}")


def load_model_json():
    # batch_size: Training batch-size
    # epochs: No. of epochs to run
    # lr: Optimizer learning rate
    # momentum: SGD momentum
    # no_cuda: Disables CUDA training (**To be implemented)
    # seed: Random seed
    # in-chn: Input image channels (3 for RGB, 4 for RGB-A)
    # out-chn: Output channels/semantic classes (2 for Pavements dataset)

    with open(os.path.join(os.getcwd(), 'Postdam/model.json')) as f:
        model_json = json.load(f)

    return model_json


def main(args):
    cuda_available = torch.cuda.is_available()
    writer = SummaryWriter(args.tb_logs_dir)

    weight_fn = Path(args.weight_fn) if args.weight_fn is not None else None
    model_json = load_model_json()

    assert len(model_json['cross_entropy_loss_weights']) == model_json[
        'out_chn'], "CrossEntropyLoss class weights must be same as no. of output channels"

    class_labels = np.array(load_json(args.labels_class))
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = TiffDataset(Path(args.input_raw_dir), Path(args.input_label_dir), transform, class_labels)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=model_json['batch_size'], shuffle=True,
                                               num_workers=4)

    model = SegNet(in_chn=model_json['in_chn'], out_chn=model_json['out_chn'], BN_momentum=model_json['bn_momentum'])
    optimizer = optim.SGD(model.parameters(), lr=model_json['learning_rate'], momentum=model_json['sgd_momentum'])
    # [0.25, 3.55, 1.32, 0.21, 0.27, 0.40],
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(model_json["cross_entropy_loss_weights"]))
    # loss_fn = nn.CrossEntropyLoss()
    if cuda_available:
        model.cuda()
        loss_fn.cuda()

    run_epoch = model_json['epochs']
    epoch = None
    if weight_fn is None:
        print("Starting new checkpoint.")
        weight_fn = os.path.join(
            os.getcwd(),
            f'weights/checkpoint_pavements_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth.tar',
        )
    elif weight_fn.is_file():
        print(f"Loading checkpoint '{weight_fn}'")
        checkpoint = torch.load(weight_fn)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{weight_fn}' (epoch {checkpoint['epoch']})")
    else:
        print(f"No checkpoint found at '{weight_fn}'. Will create new checkpoint.")

    for i in range(epoch + 1 if epoch is not None else 1, run_epoch + 1):
        print(f'Epoch {i}:')
        sum_loss = 0.0

        for j, data in enumerate(train_loader, 1):
            images, labels = data
            labels = labels.argmax(dim=1)
            if cuda_available:
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss', loss.item() / train_loader.batch_size, j)
            sum_loss += loss.item()

            #print(f'Loss at {j} mini-batch: {loss.item() / train_loader.batch_size}')

        print(f'Average loss @ epoch: {(sum_loss / (j * train_loader.batch_size))}')

    print("Training complete. Saving checkpoint...")
    save_checkpoint({'epoch': run_epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                    weight_fn)


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)

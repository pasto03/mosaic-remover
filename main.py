"""
try the model here
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

import os
import pickle
import shutil

import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as transforms

from generator import Generator
from config import device, IMAGE_SIZE, mosaic_window_size
from utils import load_checkpoint

from mosaic import add_mosaic


class MosaicRemover:
    def __init__(self, device='cpu', pretrained=True):
        self.device = 'cpu' if not torch.cuda.is_available() else device
        self.generator = Generator(in_channels=3, features=64).to(self.device)
        if pretrained:
            checkpoint_file = "train-data-20231009T101553Z-001/gen checkpoint epoch--5 timestep--500.pth.tar"
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.generator.load_state_dict(checkpoint["state_dict"])
            print("=> Pretrained generator loaded")
        self.generator.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        ])

    def __process_image(self, image_path):
        """
        add mosaic to an original image
        :param image_path: str, string path of image
        :return: mosaic_image, t_image
        """
        image = Image.open(image_path)
        # t_image = transforms.ToTensor()(image)
        t_image = self.transform(image)
        mosaic_image = torch.tensor(add_mosaic(t_image, window_size=mosaic_window_size))
        return mosaic_image, t_image

    def remove_mosaic(self, image_path, output_folder="example outputs", apply_mosaic=True):
        image_name = ''.join(os.path.basename(image_path).split('.')[:-1])
        mosaic_image, t_image = self.__process_image(image_path)
        processed_image = mosaic_image if apply_mosaic else t_image
        pred = self.generator(processed_image.unsqueeze(0).to(self.device))
        print("=> Prediction generated")

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        save_image(pred[0], f"{output_folder}/{image_name} -- pred.png")
        save_image(mosaic_image, f"{output_folder}/{image_name} -- mosaic.png")
        if apply_mosaic:
            save_image(t_image, f"{output_folder}/{image_name} -- original.png")
        print("=> Prediction image saved")

    def __call__(self, *args, **kwargs):
        # self.remove_mosaic(kwargs["image_path"])
        raise NotImplementedError
    

if __name__ == '__main__':
    remover = MosaicRemover(device='cuda')

    # 1. apply and remove mosaic
    image_path = "ORIGINAL_IMAGE_PATH"
    remover(image_path=image_path)

    # 2. pass mosaic image path directly -- set apply_mosaic as False
    # image_path = "MOSAIC_IMAGE_PATH"
    # remover.remove_mosaic(image_path, apply_mosaic=False)


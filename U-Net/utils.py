import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch import nn
from torch import optim
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.utils.data as data
import pandas as pd
import os
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
import math
import random
import cv2
import numpy as np
from cv2.cv2 import imread
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL.Image as PIL
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError as err:
        print("OS error: {0}".format(err))

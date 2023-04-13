import os

import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from pathlib import Path

from tqdm import tqdm

from definitions import ROOT_DIR, MODEL_PATH
from training.custom_image_dataset_from_directory import CustomImageDatasetFromDirectory
from training.captcha_coder import Coding, CaptchaCoder

from training.models.resnet_101_mse import ResNet101Mse
from training.models.resnet_152_mse import ResNet152Mse
from training.models.resnet_50_mse import ResNet50Mse
from training.models.resnext_101_32x8d_mse import ResNext10132x8d

from training.utils.training_utils import get_default_device, DeviceDataLoader, to_device, evaluate, fit

CAPTCHA_IMAGE_PATH = ROOT_DIR / Path("training_data/tmp")

# Problem parameters
ALPHABET_SIZE = 33
CAPTCHA_CHARACTERS = 5

# Model hyperparameters
BATCH_SIZE = 8
EPOCHS = 100
LR = 0.001
OPT_FUNC = torch.optim.Adam
# OPT_FUNC = torch.optim.SGD

# Prepare dataset and set up pipeline
transformations = torchvision.transforms.Compose([
    # Converts the PIL image with a pixel range of [0, 255] to a PyTorch FloatTensor of shape (C, H, W) with a range
    # [0.0, 1.0]
    torchvision.transforms.ToTensor(),
    # This normalizes the tensor image with mean and standard deviation
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Resize image to 224 x 224 as required by most vision models
    torchvision.transforms.Resize(size=(224, 224))
])


def train():
    dataset = CustomImageDatasetFromDirectory(str(CAPTCHA_IMAGE_PATH), encoding=Coding.full_one_hot,
                                                transform=transformations)
    print("Number of CAPTCHAs in training set: ", dataset.num_of_samples)


    val_size = 1191 # 取15%作为验证集
    train_size = dataset.num_of_samples - val_size

    # 打印数据星系
    print("Num of classes:", dataset.get_num_of_classes())
    print("Class names:", dataset.get_class_names())
    print("Class occurrences:", dataset.get_labels_occurrences_map())

    # 对数据集进行切分
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print("Train set size:", len(train_ds))
    print("Validation set size:", len(val_ds))

    # 加载数据
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, BATCH_SIZE)

    # 打印train_loader中的数据
    for images, labels in train_loader:
        print("images.shape:", images.shape)
        print("labels.shape:", labels.shape)
        print("labels:", labels)
        break

    # 选择GPU
    device = get_default_device()
    print("Actual running device:", device)

    # 将数据加载到GPU
    train_loader_cuda = DeviceDataLoader(train_loader, device)
    val_loader_cuda = DeviceDataLoader(val_loader, device)

    # 验证数据集
    for images, _ in train_loader:
        print('CAPTCHA images.shape:', images.shape)
        plt.axis('off')
        plt.imshow(images[0].permute(1, 2, 0))
        break
    plt.show()

    # 选择模型
    # model = ResNet101Mse(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    # model = ResNet50Mse(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    # model = ResNet152Mse(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    model = ResNext10132x8d(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    to_device(model, device)

    # 验证模型输出
    for images, _ in train_loader_cuda:
        out = model(images[0].view(1, 3, 224, 224))
        print(out.size())
        break

    # 训练模型
    history = [evaluate(model, val_loader_cuda)]
    print(history)
    history = fit(EPOCHS, LR, model, train_loader_cuda, val_loader_cuda, OPT_FUNC)
    # TODO - Train model - Transfer learning


def test():
    dataset = CustomImageDatasetFromDirectory(str(ROOT_DIR / Path("training_data/captchy_test")),
                                                encoding=Coding.full_one_hot, transform=transformations)
    test_ds = dataset
    print("Number of CAPTCHAs in test set: ", dataset.num_of_samples)

    # batch_size为1，因为测试集中每个样本都是独立的
    batch_size = 1
    test_loader = DataLoader(test_ds, batch_size)

    device = get_default_device()
    print("Actual running device:", device)

    test_loader_cuda = DeviceDataLoader(test_loader, device)

    # 验证数据集
    for images, _ in test_loader:
        print('CAPTCHA images.shape:', images.shape)
        plt.axis('off')
        plt.imshow(images[0].permute(1, 2, 0))
        break
    plt.show()

    # 选择保存的模型
    model_name = "ResNext10132x8d_acc=0.9773489832878113.pt"

    model = ResNext10132x8d(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    # model = ResNet50Mse(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    # model = ResNet152Mse(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    # model = ResNet101Mse(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    model.load_state_dict(torch.load(MODEL_PATH / model_name))
    print("Model", model_name, "have been loaded")
    model.eval()

    to_device(model, device)

    correct = 0
    error_str = ""
    coder = CaptchaCoder(Coding.full_one_hot)
    with torch.no_grad():
        for batch in tqdm(test_loader_cuda):
            images, labels = batch

            # 预测
            out = model(images)

            # 计算完全解码的验证码，便于调试
            prediction = coder.decode_raw_output(out.cpu())
            ground_truth = coder.decode(labels.cpu())
            if prediction == ground_truth:
                correct += 1
                print("Correct: ", prediction, " Ground truth: ", ground_truth)
            else:
                print("Mistake, predicted:", prediction, " ,but correct is:", ground_truth)

    # 预测结果
    wrong = dataset.num_of_samples - correct
    precision = correct / dataset.num_of_samples
    print("Correct:", correct, "Wrong:", wrong)
    print("Precision:", precision)

def test_of_upload_image(img):
    # 选择保存的模型
    model_name = "ResNext10132x8d_acc=0.9773489832878113.pt"

    model = ResNext10132x8d(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    # model = ResNet50Mse(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    # model = ResNet152Mse(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    # model = ResNet101Mse(ALPHABET_SIZE * CAPTCHA_CHARACTERS, False)
    model.load_state_dict(torch.load(MODEL_PATH / model_name))
    print("Model", model_name, "have been loaded")
    model.eval()

    device = get_default_device()
    print("Actual running device:", device)
    to_device(model, device)

    coder = CaptchaCoder(Coding.full_one_hot)

    img = Image.open(img).convert('RGB')
    img = transformations(img)
    img = img.view(1, 3, 224, 224)
    img = img.to(device)

    with torch.no_grad():
        # 预测
        out = model(img)

        # 计算完全解码的验证码，便于调试
        prediction = coder.decode_raw_output(out.cpu())
        return prediction


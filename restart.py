import argparse
import os
import string
from collections import Counter, defaultdict
from pathlib import Path

import torchvision
from tqdm import tqdm

from definitions import ROOT_DIR, MODEL_PATH

file_root = ROOT_DIR / Path("training_data/captcha_images_v2")
model_root = ROOT_DIR / Path("model_save")

# 自己生成验证码
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = [chr(each) for each in range(97, 123)]
alphabet.remove("i")
alphabet.remove("q")
alphabet.remove("o")

# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number + alphabet, captcha_size=5):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

# 生成字符对应的验证码
def gen_captcha_text_and_image():
    # 指定图片大小
    image = ImageCaptcha(width=200, height=50)

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image

def creat(num, file_root):
    for i in range(num):
        text, image = gen_captcha_text_and_image()
        file_path = file_root / Path(text + ".png")
        if not file_path.exists():
            plt.imsave(file_path, image)
        if i % 100 == 0:
            print(f"已生成{i}张验证码")

def creat_captcha():
    # 生成1k个验证码到file_root目录下
    creat(1000, file_root / Path('train'))
    creat(100, file_root / Path('test'))
    creat(100, file_root / Path('val'))

    # 统计训练集验证码的字符出现次数
    file_train_list = list(file_root.glob('train/*.png'))

    dict = defaultdict(int)

    for file in file_train_list:
        file_name = file.stem
        for char in file_name:
            dict[char] += 1

    print(len(dict))

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(20, 8), dpi=80)
    plt.bar(dict.keys(), dict.values())
    plt.show()

# 使用CRNN + CTC进行对验证码进行识别

# 1. 首先对验证码进行预处理，将验证码转换为灰度图，然后进行二值化处理，最后将图片转换为Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


class CaptchaDataset(Dataset):
    def __init__(self, dataset_dir, img_size=(64, 128)):
        self.vocab = string.ascii_lowercase + string.digits
        self.vocab_size = len(self.vocab)
        self.char2num = {char: i + 1 for i, char in enumerate(self.vocab)}
        self.num2char = {label: char for char, label in self.char2num.items()}
        self.dataset_dir = dataset_dir
        self.all_imgs = os.listdir(dataset_dir)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image_path = os.path.join(self.dataset_dir, self.all_imgs[index])
        pil_image = Image.open(image_path).convert('RGB')
        X = self.transform(pil_image)

        label = self.all_imgs[index].split(".")[0]
        encode_label = [self.char2num[c] for c in label]

        y = torch.LongTensor(encode_label)

        return X, y

    def __len__(self):
        return len(self.all_imgs)



# 2. 定义CRNN模型

class CRNN(nn.Module):
    def __init__(self, img_size, vocab_size, seq_dim=64, hidden_dim=256, n_rnn_layers=24):
        super(CRNN, self).__init__()

        self.hidden_dim = hidden_dim

        c, h, w = img_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),  # batch normalization layer
            nn.ReLU(),  # relu layers
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),  # batch normalization layer
            nn.ReLU(),  # relu layers
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # max pooling layer (64, )
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),  # batch normalization layer
            nn.ReLU(),  # relu layers
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),  # batch normalization layer
            nn.ReLU(),  # relu layers
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # max pooling layer
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),  # batch normalization layer
            nn.ReLU(),  # relu layers
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),  # batch normalization layer
            nn.ReLU(),  # relu layers
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),  # batch normalization layer
            nn.ReLU(),  # relu layers
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 2))  # max pooling layer
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),  # batch normalization layer
            nn.ReLU(),  # relu layers
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),  # batch normalization layer
            nn.ReLU(),  # relu layers
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),  # batch normalization layer
            nn.ReLU(),  # relu layers
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 2))  # max pooling layer
        )

        self.sequence_features = nn.Linear(512 * (h // 16), seq_dim)
        self.rnn = nn.GRU(seq_dim, hidden_dim, n_rnn_layers, bidirectional=True)  # recurrent layers
        self.fc1 = nn.Linear(hidden_dim * 2, 32)  # fully connected layers
        self.fc2 = nn.Linear(32, vocab_size)  # fully connected layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        batch_size, c, h, w = x.size()

        x = x.view(batch_size, c * h, w)
        x = x.permute(2, 0, 1)
        x = self.sequence_features(x)

        x, _ = self.rnn(x)

        x = self.fc1(x)
        x = self.fc2(x)

        return x

class GreedyCTCDecoder(nn.Module):
    def __init__(self, vocab, blank=0):
        super(GreedyCTCDecoder, self).__init__()

        self.vocab = string.ascii_lowercase + string.digits
        self.blank = blank

    def forward(self, x):
        indices = torch.argmax(x, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.vocab[i] for i in indices])

        return joined.replace("|", " ").strip().split()

parser = argparse.ArgumentParser(description="parameters for training")

parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--epochs", type=int, default=100, help="epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument('--dataset_dir', default=file_root, type=str)
parser.add_argument('--model_dir', default=model_root, type=str)

args = parser.parse_args()

DATASET_DIR = args.dataset_dir
MODEL_DIR = args.model_dir
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
IMG_SIZE = (3, 64, 128)
MOMENTUM = 0.9
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
VAL_DIR = os.path.join(DATASET_DIR, 'val')

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CaptchaDataset(TRAIN_DIR, img_size = IMG_SIZE[1:])
    val_dataset = CaptchaDataset(VAL_DIR, img_size = IMG_SIZE[1:])

    # 打印数据集大小
    for i, v in enumerate(train_dataset):
        print(i, v)
        break

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CRNN(IMG_SIZE, vocab_size=train_dataset.vocab_size).to(device)

    criterion = nn.CTCLoss(blank=train_dataset.vocab_size - 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    min_loss = float('inf')

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            outputs_softmax = nn.functional.log_softmax(outputs, dim=2)

            pred_length = torch.LongTensor([outputs_softmax.size(0)] * outputs_softmax.size(1))
            target_length = torch.tensor([len(arr) for arr in labels])

            loss = criterion(outputs_softmax, labels, pred_length, target_length)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % BATCH_SIZE == 0:
                print()
                print(f"Epoch: {epoch + 1}/{EPOCHS}, Step: {i}/{len(train_loader)}, Loss: {loss.item()}")

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for index, (imgs, labels) in enumerate(val_loader):
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                outputs_softmax = nn.functional.log_softmax(outputs, dim=2)
                pred_length = torch.LongTensor([outputs_softmax.size(0)] * outputs_softmax.size(1))
                target_length = torch.tensor([len(arr) for arr in labels])
                loss = criterion(outputs_softmax, labels, pred_length, target_length, )
                test_loss += loss.item()

            print(f'\n[EPOCH {epoch + 1}] test loss: {loss / len(val_loader)}\n')

        if test_loss < min_loss:
            min_loss = test_loss
            model_name = f'crnn_min_loss:{min_loss:.4f}.pth'
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_name))




def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = CaptchaDataset(TEST_DIR, IMG_SIZE = IMG_SIZE[1:])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_path = 'sadasd'

    decoder = GreedyCTCDecoder()

    crnn = CRNN()
    crnn.load_state_dict(torch.load(model_path))
    crnn.eval()

    acc = 0

    for i, (imgs, labels) in enumerate(test_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = crnn(imgs)
        outputs_softmax = nn.functional.softmax(outputs, dim=2)

        pred = decoder(outputs_softmax)

        acc += 1 if pred == labels else 0

    print(f"Accuracy: {acc / len(test_loader)}")

def predict():
    img_path = 'sadasd'
    model_path = 'sadasd'

    decoder = GreedyCTCDecoder()

    crnn = CRNN()
    crnn.load_state_dict(torch.load(model_path))
    crnn.eval()

    img = Image(img_path)

    transformers = transforms.Compose([
        transforms.Resize(IMG_SIZE[1:]),
        transforms.ToTensor(),
    ])

    img = transformers(img)
    img = img.unsqueeze(0)
    pred = crnn(img)
    pred = decoder(pred)

    print(pred)



if __name__ == '__main__':
    # create_captcha()
    train()
    # test()
    # predict()









import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional

from training.captcha_coder import CaptchaCoder, Coding
from training.trainer import train, test, test_of_upload_image


def test_captcha_coder():
    coder = CaptchaCoder(Coding.full_one_hot)
    print(coder._encode_full_one_hot_vector("1"))
    print(coder._encode_full_one_hot_vector("2"))
    print(coder._encode_full_one_hot_vector("3"))

    print(torch.zeros(33 * 5).dtype)

    print(coder.encode("12ack"))
    print(coder.decode(coder.encode("12ack")))


# test_captcha_coder() # 这个已经运行过了

# train() # 训练，保留了模型


test() # 测试

# 这个左边的照片就是测试的照片
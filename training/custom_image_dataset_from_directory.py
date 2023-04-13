import os
from PIL import Image, ImageOps
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import glob
import cv2


from training.captcha_coder import Coding, CaptchaCoder


class CustomImageDatasetFromDirectory(Dataset):
    """
        A dataset example where the class is embedded in the file names
        Args:
            folder_path (string): path to image folder
    """
    def __init__(self, folder_path, encoding, transform=None):
        # Get image list
        self.image_list = glob.glob(os.path.join(folder_path, '*'))
        # Calculate len
        self.num_of_samples = len(self.image_list)
        self.encoding = encoding
        self.transform = transform
        # Meant only for classification
        self.num_of_classes = 0
        self.labels_occurrences_map = dict()
        self.coder = CaptchaCoder(encoding)

    def __getitem__(self, index):
        """
        Returns preprocessed image - label pair on given index, image must be prepared to be fed to the network,
        and label must be encoded to be used directly by loss function to compute the gradient.
        """
        single_image_path = self.image_list[index]
        # Open image
        cv2_image = cv2.imread(single_image_path)
        # Clean up the image by removing noise
        #image = hard_noise_removing_filter(image)
        # Transfer image type from cv to PIL
        pil_image = Image.fromarray(cv2_image)

        # Proceed with custom transformations
        if self.transform is not None:
            tensor_image = self.transform(pil_image)
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            # Transform the pIL image to tensor
            tensor_image = transform(pil_image)

        # Get label(class) of the image based on the file name
        filename = os.path.basename(single_image_path)
        captcha_string = os.path.splitext(filename)[0]

        if self.encoding == Coding.simple_single_char:
            captcha_string = captcha_string[:1]
            label = self.coder.encode(captcha_string)
        elif self.encoding == Coding.full_one_hot:
            label = self.coder.encode(captcha_string)
            # Check for wrong data label
            if label.size(dim=0) != 165:
                print("Wrong label shape:", label.size(), " in CAPTCHA file:", captcha_string)

        return tensor_image, label

    def __len__(self):
        return self.num_of_samples

    def _count_classes(self):
        for image_file in self.image_list:
            filename = os.path.basename(image_file)
            label = os.path.splitext(filename)[0]
            label = label[:1]
            if label not in self.labels_occurrences_map:
                self.labels_occurrences_map[label] = 1
                self.num_of_classes += 1
            else:
                self.labels_occurrences_map[label] += 1

    def get_num_of_classes(self):
        if self.num_of_classes == 0:
            self._count_classes()
        return self.num_of_classes

    def get_class_names(self):
        if self.num_of_classes == 0:
            self._count_classes()
        return sorted(self.labels_occurrences_map.keys())

    def get_labels_occurrences_map(self):
        return self.labels_occurrences_map

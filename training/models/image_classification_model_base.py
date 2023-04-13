import os

import torch
import torch.nn.functional as F
from torch import nn


from definitions import MODEL_PATH


class ImageClassificationModelBase(nn.Module):
    """
        Base class providing implementation of common methods for image classification models.
    """
    def __init__(self):
        super().__init__()
        self.best_val_accuracy = float('-inf')

    # def training_step(self, batch):
    #     images, labels = batch
    #     out = self(images)  # Generate predictions
    #     loss = F.cross_entropy(out, labels)  # Calculate loss
    #     return loss

    # def accuracy(self, outputs, labels):
    #     _, preds = torch.max(outputs, dim=1)
    #     return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    # def validation_step(self, batch):
    #     images, labels = batch
    #     out = self(images)  # Generate predictions
    #     loss = F.cross_entropy(out, labels)  # Calculate loss
    #     acc = self.accuracy(out, labels)  # Calculate accuracy
    #     return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        info = ""
        if result['val_acc'] > self.best_val_accuracy:
            info = "saving new best model"
            self.remove_old_save()
            self.best_val_accuracy = result['val_acc']
            self.save()

        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, {}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc'], info))
    
    def save(self):
        path = MODEL_PATH
        name = path / (self.__class__.__name__ + "_acc=" + str(self.best_val_accuracy) + ".pt")
        torch.save(self.state_dict(), name)

    def remove_old_save(self):
        path = MODEL_PATH
        name = path / (self.__class__.__name__ + "_acc=" + str(self.best_val_accuracy) + ".pt")

        if os.path.exists(name):
            os.remove(name)

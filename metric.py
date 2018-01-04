from copy import deepcopy

import torch
import scipy.stats as scis


class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pearson(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        return torch.mean(torch.mul(x, y))

    def mse(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        return torch.mean((x - y) ** 2)

    def spearman(self, predictions, labels):
        return scis.spearmanr(predictions.tolist(), labels.tolist())[0]
    
    def pearson2(self, predictions, labels):
        return scis.pearsonr(predictions.tolist(), labels.tolist())[0]
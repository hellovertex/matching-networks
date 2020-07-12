import matplotlib.pyplot as plt
import torch

class SubsetSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices
        self.start = 0
    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    def __len__(self):
        return len(self.indices)
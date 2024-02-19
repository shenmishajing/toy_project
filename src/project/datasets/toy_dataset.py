import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    SAMPLE_NUM = {"train": 1000, "valid": 200, "test": 100}

    def __init__(self, subset, dim=1024) -> None:
        super().__init__()
        self.subset = subset
        self.dim = dim
        self.length = self.SAMPLE_NUM[subset]
        self.data = torch.randn(self.length, dim)
        self.label = torch.randint(0, 2, (self.length,))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {"data": self.data[index], "label": self.label[index], "index": index}

from torch.utils.data import Dataset

class PpoDataset(Dataset):
  def __init__(self, data):
    super(PpoDataset, self).__init__()
    self.data = []
    for d in data:
      self.data.extend(d)

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self):
    return len(self.data)

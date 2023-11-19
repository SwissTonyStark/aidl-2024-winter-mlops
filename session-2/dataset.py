import os.path
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms

class MyDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        super().__init__()
        self.images_path = images_path
        self.labels_df = pd.read_csv(labels_path)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # Standard normalization for images on greyscale
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        suite_id, sample_id, code, value, character = self.labels_df.loc[idx, :]
        path = os.path.join(self.images_path, f"input_{suite_id}_{sample_id}_{code}.jpg")
        # Convert to greyscale if needed
        sample = Image.open(path).convert('L')
        sample = self.transform(sample)

        return sample, code - 1  # code goes from 1 to 15

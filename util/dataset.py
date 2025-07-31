from torch.utils.data import Dataset
import os
from PIL import Image


class TerrainDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Sort to ensure alignment between h, t, i files
        self.height_paths = sorted(
            [os.path.join(data_dir, f)
             for f in os.listdir(data_dir) if '_h' in f]
        )
        self.terrain_paths = sorted(
            [os.path.join(data_dir, f)
             for f in os.listdir(data_dir) if '_t' in f]
        )
        self.segmentation_paths = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(
                data_dir) if '_i' in f or '_i2' in f]
        )

        assert len(self.height_paths) == len(self.terrain_paths) == len(self.segmentation_paths), \
            "Mismatch in dataset triplet lengths"

        print(f"Found {len(self.height_paths)} triplets in {data_dir}")

    def __len__(self):
        return len(self.height_paths)

    def __getitem__(self, idx):
        # Load heightmap, terrain, segmentation
        paths = [self.height_paths[idx], self.terrain_paths[idx],
                 self.segmentation_paths[idx]]
        images = []
        for path in paths:
            # image = Image.open(path).convert('RGB')
            image = Image.open(path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return tuple(images)  # (heightmap, terrain, segmentation)

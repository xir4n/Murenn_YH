import os
import random
from torch.utils.data import Dataset
import glob
import librosa
import torch
import re


CLASS_MAP = {
    'A': 0,
    'D': 1,
    'Da': 1,
    'Db': 1,
    'E': 2,
    'F': 3,
    'G': 4,
    'H': 5,
}

class TrainingDataset(Dataset):
    def __init__(self, data_folder, sample_rate=20480):
        self.data_folder = data_folder
        self.sample_rate = sample_rate
        self.file_paths = glob.glob(os.path.join(data_folder, '*.wav'))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, _ = librosa.load(file_path, sr=self.sample_rate)
        waveform = torch.from_numpy(waveform.astype("float32"))
        waveform = waveform / waveform.std()

        aug, file_type = self.parse_filename(os.path.basename(file_path))
        return {
            'waveform': waveform,
            'aug': aug,
            'type': CLASS_MAP[file_type],
            'file_name': os.path.basename(file_path)
        }
    
    def parse_filename(self, filename):
        aug_match = re.search(r'aug\{(-?\d+)\}', filename)
        aug = int(aug_match.group(1)) if aug_match else 0
        type_match = re.search(r'_([A-Z](?:[a-z])?)_', filename)
        file_type = type_match.group(1) if type_match else 'unknown'

        return aug, file_type


class TestDataset(Dataset):
    def __init__(self, data_folder, sample_rate=20480):
        self.data_folder = data_folder
        self.sample_rate = sample_rate
        self.file_paths = glob.glob(os.path.join(data_folder, '*.wav'))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, _ = librosa.load(file_path, sr=self.sample_rate)
        waveform = torch.from_numpy(waveform.astype("float32"))
        waveform = waveform / waveform.std()

        type, distance = self.parse_filename(os.path.basename(file_path))
        return {
            'waveform': waveform,
            "type": CLASS_MAP[type],
            'distance': torch.tensor(distance, dtype=torch.float32),
            'file_name': os.path.basename(file_path)
        }
    
    def parse_filename(self, name):
        type_match = re.search(r'_([A-Z](?:[a-z])?)_', name)
        label = type_match.group(1) if type_match else 'unknown'

        dist_match = re.search(r'_(\d+)(?:_(\d+))?m_', name)
        if dist_match:
            if dist_match.group(2):
                distance = f"{dist_match.group(1)}.{dist_match.group(2)}"
            else:
                distance = f"{dist_match.group(1)}"
            # Convert distance to float
            distance = float(distance)
        else:
            distance = None

        return label, distance


if __name__ == "__main__":
    data_dir = '/Users/zhang/MuReNN/YH_data_with_aug/train'
    dataset = TrainingDataset(data_dir)
    print(f"Number of samples in the dataset: {len(dataset)}")
    sample_waveform = dataset[random.randint(0, len(dataset)-1)]
    print(f"Sample waveform shape: {sample_waveform['waveform'].shape}")
    print(f"sample waveform dtype: {sample_waveform['waveform'].dtype}")
    print(f"Sample augmentation: {sample_waveform['aug']}")
    print(f"Sample type: {sample_waveform['type']}")
    print(f"Sample file name: {sample_waveform['file_name']}")

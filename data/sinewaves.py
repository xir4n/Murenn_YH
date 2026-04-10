import torch
import numpy as np
import torchaudio
import math
import pytorch_lightning as pl

class SineData(torch.utils.data.Dataset):
    def __init__(self, f_min=3000, f_max=10000, n_samples=1000, seg_length=1024, sr=20480):
        super().__init__()
        self.fmax = f_max
        self.fmin = f_min
        self.freqs = np.logspace(
            np.log2(self.fmin),
            np.log2(self.fmax),
            n_samples,
            base=2.0,
            endpoint=False,
        )
        self.seg_length = seg_length
        self.sr = sr
        self.closure = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.sr,
            n_fft = 512,
            win_length = 101,
            hop_length = 64,
            # pad = 256,
            f_min = f_min,
            f_max = f_max,
            n_mels = 32,
            power = 1.0,
            center = False,
            normalized = True,
        )

    
    def __getitem__(self, idx):
        freq = self.freqs[idx]
        t = torch.arange(self.seg_length)/self.sr
        x = torch.sin(2*math.pi*freq*t)
        y = self.closure(x)
        return freq, x, y

    def __len__(self):
        return len(self.freqs)


class SineDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=1, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    def setup(self, stage=None):
        dataset = SineData(**self.kwargs)
        n_train = int(0.8*len(dataset))
        n_val = int(0.1*len(dataset))
        n_test = len(dataset) - n_train - n_val
        split = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
        self.train_dataset, self.val_dataset, self.test_dataset = split

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


if __name__ == "__main__":
    # dataset = TrainingDataset(data_dir)
    import random
    dataset = SineData()
    print(f"Number of samples in the dataset: {len(dataset)}")
    sample_freq, sample_waveform, sample_spectrogram = dataset[random.randint(0, len(dataset)-1)]
    print(f"Sample waveform shape: {sample_waveform.shape}")
    print(f"Sample spectrogram shape: {sample_spectrogram.shape}")
    print(f"Sample frequency: {sample_freq}")
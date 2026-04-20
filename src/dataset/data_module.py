from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from src.dataset.dataset import MegaScenesDataset

from src.dataset.utils import collate_viewsets

class DataModule(LightningDataModule):

    def __init__(self, cfg, global_rank=0):
        super().__init__()
        self.dataset_cfg = cfg.dataset
        self.train_cfg = cfg.data_loader.train
        self.test_cfg = cfg.data_loader.test
        self.val_cfg = cfg.data_loader.val

    def train_dataloader(self):
        if self.dataset_cfg.train_all:
            self.dataset_cfg.val_ratio=0.
            
        train_dataset = MegaScenesDataset(
            root=self.dataset_cfg.dataset_path,
            mode="train",
            val_ratio=self.dataset_cfg.val_ratio,
            n_views=self.dataset_cfg.n_views,
            shuffle_views=True,
            image_size=self.dataset_cfg.image_size,
        )
        data_loader = DataLoader(
            train_dataset,
            batch_size=self.train_cfg.batch_size,
            num_workers=self.train_cfg.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_viewsets,
        )
        return data_loader

    def val_dataloader(self):
        val_dataset = MegaScenesDataset(
            root=self.dataset_cfg.dataset_path,
            mode="val",
            val_ratio=self.dataset_cfg.val_ratio,
            n_views=self.dataset_cfg.n_views,
            shuffle_views=False,
            image_size=self.dataset_cfg.image_size,
        )
        data_loader = DataLoader(
            val_dataset,
            batch_size=self.val_cfg.batch_size,
            num_workers=self.val_cfg.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_viewsets,
        )
        return data_loader

    def test_dataloader(self):
        val_dataset = MegaScenesDataset(
            root=self.dataset_cfg.dataset_path,
            mode="val",
            val_ratio=self.dataset_cfg.val_ratio,
            n_views=self.dataset_cfg.n_views,
            shuffle_views=False,
            image_size=self.dataset_cfg.image_size,
        )
        data_loader = DataLoader(
            val_dataset,
            batch_size=self.val_cfg.batch_size,
            num_workers=self.val_cfg.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_viewsets,
        )
        return data_loader
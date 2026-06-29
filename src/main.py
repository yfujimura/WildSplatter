import sys
sys.path.append("src/Depth-Anything-3/src")

import os
import hydra
import wandb
from omegaconf import OmegaConf

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from src.models.model import WildSplatterModel
from src.models.model_wrapper import ModelWrapper
from src.dataset.data_module import DataModule

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="main",
)
def main(cfg):
    torch.set_float32_matmul_precision("medium")

    output_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]

    callbacks = []
    if cfg.wandb.mode != "disabled" and cfg.mode == "train":
        logger = WandbLogger(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name=cfg.wandb.name + " (" + "/".join(output_dir.split("/")[-2:]) + ")",
            tags=None,
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg),
        )
        callbacks.append(LearningRateMonitor("step", True))
    else:
        logger = None

    callbacks.append(
        ModelCheckpoint(
            os.path.join(output_dir, "ckpts"),
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor="info/global_step",
            mode="max",
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy="ddp_find_unused_parameters_true",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        limit_val_batches=cfg.trainer.limit_val_batches,
        enable_progress_bar=False if cfg.mode=="train" else True,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        precision="bf16-mixed",
        max_steps=cfg.trainer.max_steps,
        inference_mode=False if cfg.mode == "test" else True,
        num_sanity_val_steps=20,
    )
    torch.manual_seed(cfg.seed + trainer.global_rank)

    wild_splatter_model = WildSplatterModel(cfg.model)
    model_wrapper = ModelWrapper(
        cfg,
        wild_splatter_model,
    )

    data_module = DataModule(
        cfg,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module)
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    main()



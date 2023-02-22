from typing import Optional
from pathlib import Path
import tempfile
import multiprocessing


import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .models import SimpleNet, LitSimpleNet
from .dataloaders import DataDist

def get_new_model_and_trainer(
    save_dir: Optional[Path] = None,
    lr: float = 3e-4,
    n_epochs: int = 50,
    log_every: int = 20,
    input_size: int = 2,
) -> tuple[LitSimpleNet, pl.Trainer]:
    """Initialize LitSimpleNet model and pl trainer.
    The trainer uses best model path to return best model score.
    When resume_from_saved == True, use the last saved ckpt to resume the training.

    Args:
        ckpt_path (os.PathLike | str): directory path to save model ckpt
        lr (float, optional): learning rate to initialize LitRankNet model. Defaults to 3e-4.
        n_epochs (int, optional): max_epoch to initialize pl.Trainer. Defaults to 50.
        log_every (int, optional): log_every_n_steps to initialize pl.Trainer. Defaults to 20.

        input_size (int, optional): input size of given featurizer, default to 2
    Returns:
        Tuple[LitRankNet, pl.Trainer]: Return model and trainer
    """

    # use tempdir if unavailable
    if save_dir is None:
        save_dir = tempfile.mkdtemp()

    save_dir.mkdir(exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=save_dir,
        name="default",
    )
    model = LitSimpleNet(
        input_size=input_size,
        lr=lr,
    )

    ckpt_callback = ModelCheckpoint(
        monitor="train/loss",
        dirpath=save_dir / "checkpoints",
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=n_epochs,
        logger=logger,
        log_every_n_steps=log_every,
        callbacks=[ckpt_callback],
    )

    return model, trainer


def get_dataloader(
    dist, 
    target,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: Optional[int] = None,
    ):
    data = DataDist(dist, target)
    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2
    return DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
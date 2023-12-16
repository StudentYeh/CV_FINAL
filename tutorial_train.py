from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import CocoDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
import time

class EpochTimerCallback(Callback):
    def on_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_epoch_end(self, trainer, pl_module):
        end_time = time.time()
        epoch_time = end_time - self.start_time
        trainer.logger.experiment.add_scalar('epoch_time', epoch_time, trainer.current_epoch)

def main():
    # Configs
    resume_path = './models/control_sd15_ini.ckpt'
    batch_size = 3 # yeh 4 to 3
    logger_freq = 1000
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    train_dataset = CocoDataset()
    train_data = DataLoader(train_dataset, num_workers=2, batch_size=batch_size, shuffle=True) # yeh 0 to 2
    # modified a val dataset in tutorial_dataset.py
    val_dataset = CocoDataset()
    val_data = DataLoader(val_dataset, num_workers=2, batch_size=batch_size, shuffle=False)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], max_epochs=100) # yeh

    # Train!
    trainer.fit(model, train_data, val_data)
    
    ## joanna
    # Create a TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="model_control_sd15")

    # Create a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='control_sd15-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    # Create the trainer
    trainer = pl.Trainer(
        gpus=1,
        precision=32,
        callbacks=[logger, checkpoint_callback, EpochTimerCallback()],
        max_epochs=100,
        logger=logger,
    )

    # Log the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.experiment.add_scalar('total_params', total_params)

    # Train and time the training process
    start_time = time.time()
    trainer.fit(model, train_data, val_data)
    end_time = time.time()

    # Log the training time
    training_time = end_time - start_time
    logger.experiment.add_scalar('training_time', training_time)


if __name__ == "__main__":
    main()
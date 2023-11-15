from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import CocoDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

def main():
    # Configs
    resume_path = './models/control_sd15_ini.ckpt'
    batch_size = 3 # yeh 4 to 3
    logger_freq = 300
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
    dataset = CocoDataset()
    dataloader = DataLoader(dataset, num_workers=2, batch_size=batch_size, shuffle=True) # yeh 0 to 2
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], max_epochs=100) # yeh


    # Train!
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
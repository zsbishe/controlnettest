from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint  # 新增：导入ModelCheckpoint模块


# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
max_epochs = 10  # 新增：设置训练的最大epoch数
checkpoint_dir = './save'  # 新增：设置模型权重保存的路径

# ModelCheckpoint回调设置
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # 根据需要选择监控指标
    dirpath=checkpoint_dir,  # 指定模型权重保存的目录
    filename='{epoch}-{val_loss:.2f}',  # 设置模型权重文件的名称格式
    save_top_k=2,  # 保存最佳的k个模型，这里是1
    mode='min',  # 如果monitor是损失，则选择最小值
)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

# 更新Trainer实例
trainer = pl.Trainer(
    gpus=1, 
    precision=32, 
    callbacks=[logger, checkpoint_callback],  # 在callbacks列表中加入ModelCheckpoint
    max_epochs=max_epochs,
)

# Train!
trainer.fit(model, dataloader)
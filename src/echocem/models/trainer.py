import torch
import torchmetrics
from pytorch_lightning import LightningModule
from hydra.utils import instantiate

class SegmTrainer(LightningModule):
    def __init__(self, model, loss_fn, optimizer_cfg, scheduler_cfg, metrics):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

        # Store optimizer and scheduler configs
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        # Store metrics as a dictionary
        self.metrics = torch.nn.ModuleDict(metrics)

        self.save_hyperparameters(ignore=['model', 'loss_fn', 'metrics'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        preds = torch.argmax(y_hat, dim=1)
        for name, metric in self.metrics.items():
            self.log(f"train_{name}", metric(preds, y), prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        preds = torch.argmax(y_hat, dim=1)
        for name, metric in self.metrics.items():
            self.log(f"val_{name}", metric(preds, y), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        self.optimizer_cfg._target_ = self.optimizer_cfg.target
        self.scheduler_cfg._target_ = self.scheduler_cfg.target
        self.optimizer_cfg.pop('target', None)
        self.scheduler_cfg.pop('target', None)
        
        optimizer = instantiate(self.optimizer_cfg, params=self.model.parameters())

        if self.scheduler_cfg is not None:
            scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
            return {
                "optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler, 
                    "monitor": "val_loss"
                }
            }
        
        return optimizer

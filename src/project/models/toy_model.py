from lightning_template import LightningModule
from torch import nn


class ToyModel(LightningModule):
    def __init__(self, dim=1024, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fcs = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.loss = nn.BCELoss()

    def forward(self, batch, *args, **kwargs):
        preds = self.fcs(batch["data"]).squeeze()
        return {
            "loss_dict": {
                "loss_cls": self.loss(preds, batch["label"].float()),
                "loss_parameter": sum(p.pow(2).mean() for p in self.parameters()),
            },
            "metric_dict": {"preds": preds, "target": batch["label"]},
        }

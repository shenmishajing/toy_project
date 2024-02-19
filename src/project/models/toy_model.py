import os

from lightning_template import LightningModule
from torch import nn
from torchmetrics import ConfusionMatrix


class ToyModel(LightningModule):
    def __init__(self, dim=1024, *args, predict_tasks=None, **kwargs) -> None:
        if predict_tasks is None:
            predict_tasks = ["confusion_matrix"]
        super().__init__(*args, predict_tasks=predict_tasks, **kwargs)
        self.fcs = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.loss = nn.BCELoss()

        if "confusion_matrix" in self.predict_tasks:
            self.confusion_matrix = ConfusionMatrix(task="binary", num_classes=2)

    def forward(self, batch, *args, **kwargs):
        preds = self.fcs(batch["data"]).squeeze()
        return {
            "loss_dict": {
                "loss_cls": self.loss(preds, batch["label"].float()),
                "loss_parameter": sum(p.pow(2).mean() for p in self.parameters()),
            },
            "metric_dict": {"preds": preds, "target": batch["label"]},
        }

    def predict_forward(self, batch, *args, **kwargs):
        return {"preds": self.fcs(batch["data"]).squeeze()}

    def predict_confusion_matrix(self, batch, *args, output_path, preds, **kwargs):
        self.confusion_matrix.update(preds, batch["label"])
        fig, _ = self.confusion_matrix.plot()
        fig.savefig(os.path.join(output_path, f"{batch['index'][0]}.png"))

    def on_predict_end(self) -> None:
        super().on_predict_end()
        if "confusion_matrix" in self.predict_tasks:
            fig, _ = self.confusion_matrix.plot()
            fig.savefig(
                os.path.join(self.predict_path, "confusion_matrix/confusion_matrix.png")
            )

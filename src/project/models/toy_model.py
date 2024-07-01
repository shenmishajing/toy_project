import os

from lightning_template import LightningModule
from torch import nn
from torchmetrics import ConfusionMatrix


class ToyModel(LightningModule):
    def __init__(
        self, input_dim=16, hidden_dim=32, *args, predict_tasks=None, **kwargs
    ) -> None:
        if predict_tasks is None:
            predict_tasks = ["confusion_matrix"]
        super().__init__(*args, predict_tasks=predict_tasks, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def build_model(self) -> None:
        # Define the model in configure_model method instead of __init__ method,
        # which is required by advanced distributed training frameworks,
        # such as deepspeed and fsdp
        self.fcs = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
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

    def predict_forward_dependency(self, batch, *args, **kwargs):
        return {"preds": self.fcs(batch["data"]).squeeze()}

    def predict_confusion_matrix_start(self, *args, **kwargs):
        return {"dependency": ["forward"], "result": []}

    def predict_confusion_matrix(self, batch, *args, output_path, preds, **kwargs):
        self.confusion_matrix.update(preds, batch["label"])
        fig, _ = self.confusion_matrix.plot()
        fig.savefig(os.path.join(output_path, f"{batch['index'][0]}.png"))

    def predict_confusion_matrix_end(self, output_path, *args, **kwargs) -> None:
        fig, _ = self.confusion_matrix.plot()
        fig.savefig(os.path.join(output_path, "confusion_matrix/confusion_matrix.png"))

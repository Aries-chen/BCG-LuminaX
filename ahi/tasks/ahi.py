import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
import pandas as pd
import rich
import torch
import torch.nn as nn
import typing as t
import numpy as np
from wuji_dl.display import df_to_rich, df_to_wandb
from wuji_dl.factory import create_instance
from wuji_dl.metrics import ClassificationMetrics
from wuji_dl.optim import SingleOptimizerFactory
from data.ahi import AHIBatch, AHINetInput
#from models.snore import YOHOEncoder,YOHOConfig
import pdb
from metrics.event_eval import cal_cls_confusion_matrix, cal_batch_cls_segs
import wandb
from wuji import sed_eval
from wuji.Evaluator.EventDetector import EventDetectorEvaluator
from wuji.Evaluator.NightlySleepMetrics import NightlySleepMetricsEvaluator
from wuji.algo.Oxygen import detect_oxygen_desaturation
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import pearsonr



class AHITask(L.LightningModule):
    def __init__(
        self,
        model: t.Dict[str, t.Any],
        criterion: t.Dict[str, t.Any],
        optimizer: t.Dict[str, t.Any],
        lr_scheduler: t.Optional[t.Dict[str, t.Any]] = None,
        bad_case_study: bool = False,
        predict: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = create_instance(nn.Module, model)
        self.criterion = create_instance(nn.Module, criterion)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.predict = predict

    def forward(self, net_input: AHINetInput) -> t.Dict[str, t.Any]:
        #return self.model.forward(net_input["source"])
        # net_input.pop('length')
        return self.model.forward(**net_input)
        #return self.model.forward(net_input["bcg_raw"])

    def configure_optimizers(self) -> t.Any:
        optimizer = create_instance(
            torch.optim.Optimizer, self.optimizer, self.parameters()
        )
        if self.lr_scheduler is None:
            return optimizer
        lr_scheduler = self.lr_scheduler.copy()
        lr_scheduler["scheduler"] = create_instance(
            object, lr_scheduler["scheduler"], optimizer
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_fit_start(self) -> None:
        if isinstance(self.logger, WandbLogger):
            self.logger.watch(self)

    def training_step(
        self, batch: AHIBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        #pdb.set_trace()
        net_output = self.forward(batch["net_input"])['ahi_event']
        batch['cls_labels'] = batch['ahi_event']
        loss = self.criterion(net_output, batch)
        #self.log("train/loss", loss["loss"], prog_bar=True)
        self.log_dict(
            {"train/" + k: v for k, v in loss.items()},
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        return loss["loss"]



    def calculate_metrics(self, pred, label):
        """
        Calculate F1 score, Pearson correlation, Precision, and Recall.
        
        Args:
            pred (torch.Tensor): Predicted tensor with shape [batch_size, length].
            label (torch.Tensor): Ground truth tensor with shape [batch_size, length].
        
        Returns:
            dict: A dictionary containing F1, Pearson, Precision, and Recall.
        """
        assert pred.shape == label.shape, "Pred and label must have the same shape."
        
        # Convert to numpy arrays for compatibility with metrics functions
        pred = pred.float().cpu().numpy()
        label = label.cpu().numpy()
        
        # Flatten tensors to 1D arrays
        pred_flat = pred.flatten()
        label_flat = label.flatten()
        
        # For F1, Precision, and Recall, threshold predictions at 0.5 (assuming binary classification)
        pred_binary = (pred_flat > 0.5).astype(int)
        label_binary = label_flat.astype(int)
        
        # F1 Score
        f1 = f1_score(label_binary, pred_binary, zero_division=0)
        
        # Precision
        precision = precision_score(label_binary, pred_binary, zero_division=0)
        
        # Recall
        recall = recall_score(label_binary, pred_binary, zero_division=0)
        
        # Pearson Correlation
        pearson_corr, _ = pearsonr(pred_flat, label_flat)
        
        return {
            "F1": f1,
            "Pearson": pearson_corr,
            "Precision": precision,
            "Recall": recall
        }



    @torch.no_grad()
    def validation_step(
        self, batch: AHIBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        net_output = self.forward(batch["net_input"])['ahi_event']
        batch['cls_labels'] = batch['ahi_event']
        loss = self.criterion(net_output, batch)
        #self.log("val/loss", loss["loss"], prog_bar=True)
        

        #pdb.set_trace()

        results = self.calculate_metrics(net_output, batch['cls_labels'])       
        self.log_dict(
            {"val/" + k: v for k, v in loss.items()} | {"val/" + p: n for p,n in results.items()
             },
            prog_bar=True,
            on_epoch=True,
            sync_dist=True
        )



    @torch.no_grad()
    def test_step(
        self, batch: AHIBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        #pdb.set_trace()
        id_index = [str(idx) for idx in batch["id"]]
        net_output = self.forward(batch["net_input"])['ahi_event']
        batch['cls_labels'] = batch['ahi_event']
        self.confusion_matrix += cal_cls_confusion_matrix(batch["cls_labels"].detach().cpu().numpy().reshape(-1), 
                                                          net_output['x'][0].detach().cpu().numpy().reshape(-1),
                                                          batch['net_input']["bcg_raw"])
        batch_gts, batch_preds = cal_batch_cls_segs(batch["cls_labels"], net_output, batch_idx, id_index)
        for i in range(len(batch_gts)):
            #pdb.set_trace()
            #save_test_result(batch["path"][i], batch_gts[i], batch_preds[i])
            if id_index[i] not in self.id_Metrics:
                self.id_Metrics[id_index[i]] = sed_eval.sound_event.EventBasedMetrics(np.array(['snore']), t_collar=5.00,percentage_of_length=0.1)
            self.id_Metrics[id_index[i]].evaluate(batch_gts[i], batch_preds[i])
            self.Metrics.evaluate(batch_gts[i], batch_preds[i])
        
        
    def on_test_start(self) -> None:
        if isinstance(self.logger, WandbLogger):
            self.logger.watch(self)
        self.Metrics = sed_eval.sound_event.EventBasedMetrics(np.array(['snore']), t_collar=5.00,percentage_of_length=0.1)
        self.id_Metrics = {}
        self.confusion_matrix = np.zeros((2,2))

    @torch.no_grad()
    def on_test_epoch_end(self):
        print(self.confusion_matrix)
        f1, precision, recall = self.Metrics.overall_f_measure().values()
        self.log_dict(
            {"test/precision": precision, "test/recall": recall, "test/f1": f1},
        )
        table = wandb.Table(columns=["id", "precision", "recall", "f1", "correct_label_num"])
        for id in self.id_Metrics:
            self.id_Metrics[id].results()
            print(self.id_Metrics[id].overall)
            f1, precision, recall = self.id_Metrics[id].overall_f_measure().values()
            Nref = self.id_Metrics[id].overall['Nref']
            #self.log_dict(
            #    {"test/"+id+"/precision": precision, "test/"+id+"/recall": recall, "test/"+id+"/f1": f1},
            #)
            table.add_data(id, precision, recall, f1, Nref)
        self.logger.experiment.log({"test/id_Metrics": table})
        table = wandb.Table(columns=["Criterion", "Value"])
        # 添加数据到表格
        table.add_data("precision", precision)
        table.add_data("recall", recall)
        table.add_data("f1", f1)
        # 记录表格到 W&B
        self.logger.experiment.log({"test/Metrics": table})

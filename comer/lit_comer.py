import zipfile
from typing import List

import pytorch_lightning as pl
import torch.optim as optim
from torch import FloatTensor, LongTensor
import torch


import math 
from torch.optim.lr_scheduler import LambdaLR
from comer.datamodule import Batch, vocab
from comer.model.comer import CoMER
from comer.utils.utils import (ExpRateRecorder, Hypothesis, ce_loss,
                               to_bi_tgt_out,WERRecorder,BLEURecorder,LinearWarmupScheduler,CombinedScheduler)


class LitCoMER(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        # training
        learning_rate: float,
        milestones: List[int],
        patience: int,
        warmup_epochs: int,
        warmup_steps: int,
        scheduler_name: str,
        optimizer_name: str,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.warmup_epochs = warmup_epochs

        self.comer_model = CoMER(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        self.exprate_recorder = ExpRateRecorder()
        self.wer_recorder = WERRecorder()
        self.bleu_recorder = BLEURecorder()

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.comer_model(img, img_mask, tgt)

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        
        
        # TODO  100 epochs for warm up
        # For the first 30 epochs, log a  increasing but small value for val_ExpRate to avoid triggering scheduler
        if self.current_epoch < self.warmup_epochs:
            self.log(
                "val_ExpRate",
                self.exprate_recorder,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log( 
                "val_WER", 
                self.wer_recorder, prog_bar=True, on_step=False, on_epoch=True
            )
            self.log(
            "val_BLEU", 
            self.bleu_recorder, prog_bar=True, on_step=False, on_epoch=True
            )   
            return

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.wer_recorder([h.seq for h in hyps], batch.indices)
        self.bleu_recorder([h.seq for h in hyps], batch.indices)
        
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_WER", self.wer_recorder, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log(
            "val_BLEU", self.bleu_recorder, prog_bar=True, on_step=False, on_epoch=True
        )

    
    def test_step(self, batch: Batch, _):

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.wer_recorder([h.seq for h in hyps], batch.indices)
        self.bleu_recorder([h.seq for h in hyps], batch.indices)
        return batch.img_bases, [vocab.indices2label(h.seq) for h in hyps]

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        wer = self.bleu_recorder.compute()
        bleu = self.bleu_recorder.compute()
        
        print()
        print(f"Validation ExpRate: {exprate}")
        print(f"Validation WER: {self.wer_recorder.wer/self.wer_recorder.total_line}")
        print(f"Validation BLEU: {self.bleu_recorder.total_bleu / self.bleu_recorder.total_line}")

        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds in test_outputs:
                for img_base, pred in zip(img_bases, preds):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)


    def approximate_joint_search(
        self, img: FloatTensor, mask: LongTensor
    ) -> List[Hypothesis]:
        return self.comer_model.beam_search(img, mask, **self.hparams)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5
        )
        if self.hparams.optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate
            )
        elif self.hparams.optimizer_name == 'adadelta':
            optimizer = optim.Adadelta(
                self.parameters(),
                lr=self.hparams.learning_rate,
                eps=1e-6,
                weight_decay=1e-4,
            )
        elif self.hparams.optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5
            )

        # init scheduler
        scheduler = None
        init_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.hparams.milestones, gamma=0.1
        )
        
        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.1,
                patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
            )
        
        if self.hparams.scheduler_name == 'warmup_plateau':
            
            def warmup_lr(epoch):
                if epoch < self.warmup_epochs:
                    return 1.0
                else:
                    return 0.1 ** ((epoch - self.warmup_epochs) // self.hparams.patience)

            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": warmup_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "reduce_on_plateau": False,  # Warmup阶段不监控指标
                },
                "reduce_lr_on_plateau": {
                    "scheduler": reduce_scheduler,
                    "monitor": "val_ExpRate",
                    "interval": "epoch",
                    "frequency": self.trainer.check_val_every_n_epoch,
                    "strict": True,
                },
            }
            
        if self.hparams.scheduler_name == 'warmup_step':
            warmup_scheduler = LinearWarmupScheduler(
                optimizer, warmup_steps=5, final_lr=self.hparams.learning_rate)
            scheduler = CombinedScheduler(warmup_scheduler, init_scheduler)
        if self.hparams.scheduler_name == 'warmup_cosine':
            # Define a lambda function for the learning rate schedule
            def lr_lambda(current_step):
                warmup_steps = self.hparams.warmup_steps
                max_epochs = self.trainer.max_epochs

                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / \
                    float(max(1, max_epochs - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
        if scheduler is None:
            scheduler = init_scheduler
            
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

        
  
        


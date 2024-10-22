from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from comer.datamodule import vocab
from einops import rearrange
from torch import LongTensor
from torchmetrics import Metric

import editdistance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (CosineAnnealingLR, MultiStepLR, StepLR,
                                      _LRScheduler)

class Hypothesis:
    seq: List[int]
    score: float

    def __init__(
        self,
        seq_tensor: LongTensor,
        score: float,
        direction: str,
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()

        if direction == "r2l":
            result = raw_seq[::-1]
        else:
            result = raw_seq

        self.seq = result
        self.score = score

    def __len__(self):
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}"


class ExpRateRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[List[int]], indices: List[List[int]]):
        for pred, truth in zip(indices_hat, indices):
            pred = vocab.indices2label(pred)
            truth = vocab.indices2label(truth)

            is_same = pred == truth

            if is_same:
                self.rec += 1

            self.total_line += 1

    def compute(self) -> float:
        exp_rate = self.rec / self.total_line
        return exp_rate


def ce_loss(
    output_hat: torch.Tensor,
    output: torch.Tensor,
    ignore_idx: int = vocab.PAD_IDX,
    reduction: str = "mean",
) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction)
    return loss


def to_tgt_output(
    tokens: Union[List[List[int]], List[LongTensor]],
    direction: str,
    device: torch.device,
    pad_to_len: Optional[int] = None,
) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    tokens : Union[List[List[int]], List[LongTensor]]
        indices: [b, l]
    direction : str
        one of "l2f" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}

    if isinstance(tokens[0], list):
        tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]

    if direction == "l2r":
        tokens = tokens
        start_w = vocab.SOS_IDX
        stop_w = vocab.EOS_IDX
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        start_w = vocab.EOS_IDX
        stop_w = vocab.SOS_IDX

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]

    length = max(lens) + 1
    if pad_to_len is not None:
        length = max(length, pad_to_len)

    tgt = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        tgt[i, 1 : (1 + lens[i])] = token

        out[i, : lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out


def to_bi_tgt_out(
    tokens: List[List[int]], device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)

    return tgt, out



class WERRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("wer", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[List[int]], indices: List[List[int]]):
        for pred, truth in zip(indices_hat, indices):
            d = editdistance.eval(pred, truth)
            self.wer += d / len(truth)
            self.total_line += 1

    def compute(self) -> float:
        wer = self.wer / self.total_line
        return wer
    
    
class BLEURecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_bleu", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[List[int]], indices: List[List[int]]):
        smooth = SmoothingFunction().method4
        for pred, truth in zip(indices_hat, indices):
            pred = vocab.indices2label(pred)
            truth = [vocab.indices2label(truth)]

            bleu_score = sentence_bleu(truth, pred, smoothing_function=smooth)

            self.total_bleu += bleu_score
            self.total_line += 1

    def compute(self) -> float:
        bleu = self.total_bleu / self.total_line
        return bleu


class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, final_lr: float, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.final_lr = final_lr
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.final_lr * (self.last_epoch + 1) / self.warmup_steps for _ in self.optimizer.param_groups]
        else:
            return [group['lr'] for group in self.optimizer.param_groups]

class CombinedScheduler:
    def __init__(self, warmup_scheduler, main_scheduler):
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler

    def step(self, epoch=None, metrics=None):
        if self.warmup_scheduler.last_epoch < self.warmup_scheduler.warmup_steps:
            self.warmup_scheduler.step(epoch)
        else:
            self.main_scheduler.step(epoch if epoch is not None else self.warmup_scheduler.last_epoch - self.warmup_scheduler.warmup_steps, metrics)

    def state_dict(self):
        return {
            "warmup_scheduler": self.warmup_scheduler.state_dict(),
            "main_scheduler": self.main_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.warmup_scheduler.load_state_dict(state_dict["warmup_scheduler"])
        self.main_scheduler.load_state_dict(state_dict["main_scheduler"])
        

class CombinedScheduler2:
    def __init__(self, warmup_scheduler, reduce_scheduler, warmup_epochs):
        self.warmup_scheduler = warmup_scheduler
        self.reduce_scheduler = reduce_scheduler
        self.warmup_epochs = warmup_epochs
        self.current_scheduler = warmup_scheduler

    def step(self, epoch=None, metrics=None):
        if epoch is not None and epoch >= self.warmup_epochs:
            self.current_scheduler = self.reduce_scheduler
        
        if metrics is not None:
            self.current_scheduler.step(metrics, epoch)
        else:
            self.current_scheduler.step(epoch)

    def state_dict(self):
        return {
            'warmup_scheduler': self.warmup_scheduler.state_dict(),
            'reduce_scheduler': self.reduce_scheduler.state_dict(),
            'current_scheduler': self.current_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        self.reduce_scheduler.load_state_dict(state_dict['reduce_scheduler'])
        self.current_scheduler.load_state_dict(state_dict['current_scheduler'])

    def get_last_lr(self):
        return self.current_scheduler.get_last_lr()
 # 使用 CombinedScheduler 包装两个调度器
        # combined_scheduler = CombinedScheduler(
        #     warmup_scheduler, reduce_scheduler, self.warmup_epochs
        # )

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": combined_scheduler
        # }
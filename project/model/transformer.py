"""
Example template for defining a system.
"""
from argparse import ArgumentParser

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchaudio
import torchvision.transforms as transforms
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader

from project.utils.functions import data_processing, GreedyDecoder, cer, wer
from project.utils.cosine_annearing_with_warmup import CosineAnnealingWarmUpRestarts

from .decoder import Decoder
from .encoder import Encoder

IGNORE_ID = -1


def cal_performance(pred, gold, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    Args:
        pred: N x T x C, score before softmax
        gold: N x T
    """

    pred = pred.view(-1, pred.size(2))
    gold = gold.contiguous().view(-1)
    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(IGNORE_ID)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """

    if False or smoothing > 0.0:
        eps = smoothing
        n_class = pred.size(1)

        # Generate one-hot matrix: N x C.
        # Only label position is 1 and all other positions are 0
        # gold include -1 value (IGNORE_ID) and this will lead to assert error
        gold_for_scatter = gold.ne(IGNORE_ID).long() * gold
        one_hot = torch.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(IGNORE_ID)
        n_word = non_pad_mask.sum().item()
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum() / n_word
    else:
        loss = F.cross_entropy(
            pred, gold, ignore_index=IGNORE_ID, reduction="elementwise_mean"
        )

    return loss


class Transformer(LightningModule):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder=None, decoder=None, **kwargs):
        super(Transformer, self).__init__()
        self.save_hyperparameters()

        if encoder is not None and decoder is not None:
            self.encoder = encoder
            self.decoder = decoder

            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            self.encoder = Encoder()
            self.decoder = Decoder()
        self.criterion = nn.CTCLoss()

    def forward(self, padded_input, input_lengths, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        # pred is score before softmax
        pred, gold, *_ = self.decoder(
            padded_target, encoder_padded_outputs, input_lengths
        )
        return pred, gold

    def recognize(self, input, input_length, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        encoder_outputs, *_ = self.encoder(input.unsqueeze(0), input_length)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0], char_list, args)
        return nbest_hyps

    def serialize(self, optimizer, epoch, tr_loss, val_loss):
        package = {
            "state_dict": self.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        if tr_loss is not None:
            package["tr_loss"] = tr_loss
            package["val_loss"] = val_loss
        return package

    # ---------------------
    # Pytorch lightning overrides
    # ---------------------
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        # spectrograms, labels, input_lengths, label_lengths = batch
        # y_hat = self(spectrograms)
        # output = F.log_softmax(y_hat, dim=2)
        # output = output.transpose(0, 1)  # (time, batch, n_class)
        # loss = self.criterion(output, labels, input_lengths, label_lengths)
        # tensorboard_logs = {"Loss/train": loss}
        spectrograms, labels, input_lengths, label_lengths = batch
        spectrograms = spectrograms.squeeze().permute(0, 2, 1)
        input_lengths = torch.tensor(input_lengths)
        # Forward prop.
        pred, gold = self(spectrograms, input_lengths, labels)
        # print(pred.dtype, gold.dtype)
        loss, n_correct = cal_performance(pred, gold.long(), smoothing=0)
        tensorboard_logs = {"Loss/train": loss}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        spectrograms, labels, input_lengths, label_lengths = batch
        spectrograms = spectrograms.squeeze().permute(0, 2, 1)
        input_lengths_t = torch.tensor(input_lengths)
        # Forward prop.
        pred, gold = self(spectrograms, input_lengths_t, labels)
        # self.criterion(output, labels, input_lengths, label_lengths)
        # print(pred.shape)
        # print( torch.argmax(pred, dim=2).shape)
        # print(gold.shape)

        # print(pred)
        # print( torch.argmax(pred, dim=2))
        # print(gold)
        # sys.exit(0)
        loss, n_correct = cal_performance(pred, gold.long(), smoothing=0)

        decoded_preds, decoded_targets = GreedyDecoder(torch.argmax(pred, dim=2), labels, label_lengths)
        tensorboard_logs = {"Loss/train": loss}

        test_cer, test_wer = [], []
        for j in range(len(decoded_preds)):
            test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

        avg_cer = torch.FloatTensor([sum(test_cer) / len(test_cer)])
        avg_wer = torch.FloatTensor(
            [sum(test_wer) / len(test_wer)]
        )  # Need workt to make all operations in torch
        logs = {
            "cer": avg_cer,
            "wer": avg_wer,
        }
        return {
            "val_loss": loss,
            "n_correct_pred": n_correct,
            "n_pred": len(spectrograms),
            "log": logs,
            "wer": avg_wer,
            "cer": avg_cer,
        }

    def test_step(self, batch, batch_idx):
        spectrograms, labels, input_lengths, label_lengths = batch
        spectrograms = spectrograms.squeeze().permute(0, 2, 1)
        input_lengths_t = torch.tensor(input_lengths)
        # Forward prop.
        pred, gold = self(spectrograms, input_lengths_t, labels)
        loss, n_correct = cal_performance(pred, gold.long(), smoothing=0)

        decoded_preds, decoded_targets = GreedyDecoder(torch.argmax(pred, dim=2), labels, label_lengths)
        tensorboard_logs = {"Loss/train": loss}

        test_cer, test_wer = [], []
        for j in range(len(decoded_preds)):
            test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

        avg_cer = torch.FloatTensor([sum(test_cer) / len(test_cer)])
        avg_wer = torch.FloatTensor(
            [sum(test_wer) / len(test_wer)]
        )  # Need workt to make all operations in torch
        logs = {
            "cer": avg_cer,
            "wer": avg_wer,
        }
        return {
            "val_loss": loss,
            "n_correct_pred": n_correct,
            "n_pred": len(spectrograms),
            "log": logs,
            "wer": avg_wer,
            "cer": avg_cer,
        }

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(
            x["n_pred"] for x in outputs
        )
        avg_wer = torch.stack([x["wer"] for x in outputs]).mean()
        avg_cer = torch.stack([x["cer"] for x in outputs]).mean()
        tensorboard_logs = {
            "Loss/val": avg_loss,
            "val_acc": val_acc,
            "Metrics/wer": avg_wer,
            "Metrics/cer": avg_cer,
        }
        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "wer": avg_wer,
            "cer": avg_cer,
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(
            x["n_pred"] for x in outputs
        )
        avg_wer = torch.stack([x["wer"] for x in outputs]).mean()
        avg_cer = torch.stack([x["cer"] for x in outputs]).mean()
        tensorboard_logs = {
            "Loss/test": avg_loss,
            "test_acc": test_acc,
            "Metrics/wer": avg_wer,
            "Metrics/cer": avg_cer,
        }
        return {
            "test_loss": avg_loss,
            "log": tensorboard_logs,
            "wer": avg_wer,
            "cer": avg_cer,
        }

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.1, betas=(0.9, 0.98), eps=1e-09
        )
        # lr_scheduler = {'scheduler':optim.lr_scheduler.CyclicLR(optimizer,base_lr=self.hparams.learning_rate/5,max_lr=self.hparams.learning_rate,step_size_up=2000,cycle_momentum=False),
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                steps_per_epoch=int(len(self.train_dataloader())),
                epochs=self.hparams.epochs,
                anneal_strategy="linear",
                final_div_factor=0.06,
                pct_start=0.008,
            ),
            # 'scheduler': CosineAnnealingWarmUpRestarts(optimizer, T_0=int(len(self.train_dataloader())*math.pi), T_mult=2, eta_max=self.learning_rate,  T_up=int(len(self.train_dataloader()))*2, gamma=0.8),
            "name": "learning_rate",  # Name for tensorboard logs
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]

    def prepare_data(self):
        if not os.path.exists(self.hparams.data_root):
            os.makedirs(self.hparams.data_root)
        a = [
            torchaudio.datasets.LIBRISPEECH(
                self.hparams.data_root, url=path, download=True
            )
            for path in self.hparams.data_train
        ]
        b = [
            torchaudio.datasets.LIBRISPEECH(
                self.hparams.data_root, url=path, download=True
            )
            for path in self.hparams.data_test
        ]
        return a, b

    def setup(self, stage):
        self.train_data = data.ConcatDataset(
            [
                torchaudio.datasets.LIBRISPEECH(
                    self.hparams.data_root, url=path, download=True
                )
                for path in self.hparams.data_train
            ]
        )
        self.test_data = data.ConcatDataset(
            [
                torchaudio.datasets.LIBRISPEECH(
                    self.hparams.data_root, url=path, download=True
                )
                for path in self.hparams.data_test
            ]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=lambda x: data_processing(x, "train"),
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda x: data_processing(x, "valid"),
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda x: data_processing(x, "valid"),
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        # parser.add_argument("--n_cnn_layers", default=3, type=int)
        # parser.add_argument("--n_rnn_layers", default=5, type=int)
        # parser.add_argument("--rnn_dim", default=512, type=int)
        # parser.add_argument("--n_class", default=29, type=int)
        # parser.add_argument("--n_feats", default=128, type=str)
        # parser.add_argument("--stride", default=2, type=int)
        # parser.add_argument("--dropout", default=0.1, type=float)

        return parser

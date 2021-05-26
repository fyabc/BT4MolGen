#! /usr/bin/python
# -*- coding: utf-8 -*-

"""NAO criterions."""

import torch.nn.functional as F

from fairseq import utils

from . import register_criterion
from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion('nao_label_smoothed_cross_entropy')
class NaoLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    """Label smoothed cross entropy + NAO predictor loss."""

    def __init__(self, args, task):
        super().__init__(args, task)
        self.mse_ratio = args.nao_mse_ratio
        self.seq_ratio = args.nao_seq_ratio
        self.eval_score_only = False

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)

        parser.add_argument('--nao-mse-ratio', default=1000, type=float,
                            help='Ratio of NAO predictor MSE loss, default is %(default)r')
        parser.add_argument('--nao-seq-ratio', default=1, type=float,
                            help='Ratio of NAO decoder sequence loss, default is %(default)r')

    def forward(self, model, sample, reduce=True):
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        if self.eval_score_only:
            predict_value = model.encode_and_predict(**sample['net_input'])
            score = sample['score']
            mse_loss = F.mse_loss(predict_value, score, reduction='sum' if reduce else 'none')
            loss = mse_loss
            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'mse_loss': utils.item(mse_loss.data) if reduce else mse_loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['target'].size(0),
                'sample_size': sample_size,
            }
            return loss, sample_size, logging_output

        net_output = model(**sample['net_input'])
        loss, nll_loss, mse_loss, seq_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'mse_loss': utils.item(mse_loss.data) if reduce else mse_loss.data,
            'seq_loss': utils.item(seq_loss.data) if reduce else seq_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        """Compute loss.

        Notes:
            Add predict value MSE loss into the total loss item (but not nll_loss).
        """
        seq_loss, nll_loss = super().compute_loss(model, net_output, sample, reduce=reduce)

        predict_value = net_output[1]['predict_value']
        score = sample['score']
        mse_loss = F.mse_loss(predict_value, score, reduction='sum' if reduce else 'none')

        loss = self.mse_ratio * mse_loss + self.seq_ratio * seq_loss

        return loss, nll_loss, mse_loss, seq_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        sup_log_outputs = LabelSmoothedCrossEntropyCriterion.aggregate_logging_outputs(logging_outputs)
        sample_size = sup_log_outputs['sample_size']
        sup_log_outputs.update({
            'mse_loss': sum(log.get('mse_loss', 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.,
            'seq_loss': sum(log.get('seq_loss', 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.,
        })

        return sup_log_outputs

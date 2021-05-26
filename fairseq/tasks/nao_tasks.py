#! /usr/bin/python
# -*- coding: utf-8 -*-

"""NAO tasks."""

import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.data import ConcatDataset, NaoLanguagePairDataset
from fairseq.data.nao_dataset import SingleTensorDataset

from . import register_task
from .translation import TranslationTask


# TODO: Move it to molecule related folder.
DEFAULT_BOUNDS = {
    '01': (0.0, 1.0),
    'default': (0.0, 1.0),
    'drd2-m1': (0.0, 0.05),
    'qed-m1': (0.7, 0.8),
    'logp04-m1': (-10.0, 2.0),
    'logp06-m1': (-10.0, 4.0),

    'drd2-default': (0.0, 1.0),
    'qed-default': (0.0, 1.0),
    'logp-default': (-10.0, 5.0),
}


def _get_bound(bound_str: str):
    if bound_str is None:
        return None
    if ',' in bound_str:
        low, high = bound_str.split(',')
        return float(low), float(high)
    bound = DEFAULT_BOUNDS.get(bound_str, None)
    if bound is None:
        print('| WARNING: default bound name {!r} not found, fall back to [0, 1].'.format(bound_str))
        bound = (0.0, 1.0)
    return bound


def load_score(
    data_path, split, src, props, bounds,
    combine, upsample_primary,
):
    if len(props) > 1:
        raise NotImplementedError('multiple property scores not supported now')
    if len(props) != len(bounds):
        raise RuntimeError('property and bound length mismatch')
    prop = props[0]
    bound = _get_bound(bounds[0])
    prop_name = '' if prop is None else '-{}'.format(prop)

    score_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        filename = os.path.join(data_path, '{}.score{}.{}.npz'.format(split_k, prop_name, src))

        if not os.path.exists(filename):
            if k > 0:
                break
            else:
                raise FileNotFoundError('Score dataset not found: {}, {} ({})'.format(split, prop, data_path))

        data = np.load(filename)['arr_0']
        if bound is None:
            if np.any(data > 1.0) and np.any(data < 0.0):
                raise RuntimeError('scores must be scaled to [0, 1]')
        else:
            assert bound[0] < bound[1]
            data = np.maximum(data, bound[0])
            data = np.minimum(data, bound[1])
            data = (data - bound[0]) / (bound[1] - bound[0])

        dataset = SingleTensorDataset(torch.from_numpy(data).to(dtype=torch.float32))
        score_datasets.append(dataset)

        print('| {} {} {}-score{} {} examples'.format(data_path, split_k, src, prop_name, len(score_datasets[-1])))

        if not combine:
            break

    if len(score_datasets) == 1:
        score_dataset = score_datasets[0]
    else:
        sample_ratios = [1] * len(score_datasets)
        sample_ratios[0] = upsample_primary
        score_dataset = ConcatDataset(score_datasets, sample_ratios)

    return score_dataset


@register_task('nao_translation')
class NaoTranslationTask(TranslationTask):
    """Translation task with NAO prediction.

    Include sources, targets and scores.
    """

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)

        parser.add_argument('--disable-score', action='store_true', default=False,
                            help='disable score dataset, train as common translation tasks')
        parser.add_argument('--score-prop', default=None,
                            help='colon separated score property list, will use "score" name by default')
        parser.add_argument('--score-bound', default=None,
                            help='colon separated score bound list in format "<LOW>,<HIGH>" or string name, '
                                 'default is no scaling')

        # Evaluation arguments.
        parser.add_argument('--eval-score-only', action='store_true',
                            help='Only evaluate predicted scores for NAO tasks')
        parser.add_argument('--nao-gen-step', action='store_true',
                            help='Generate new target sequences for NAO tasks')
        parser.add_argument('--nao-lambda-max', type=float, default=1000.0,
                            help='Max value of NAO predict lambda, default is %(default)r')

    @classmethod
    def setup_task(cls, args, **kwargs):
        instance = super().setup_task(args, **kwargs)

        if instance.args.eval_score_only:
            print('| NAO evaluate score only')
        if instance.args.nao_gen_step:
            print('| NAO generate | max-lambda {:6.1f}'.format(instance.args.nao_lambda_max))

        return instance

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        super().load_dataset(split, epoch=epoch, combine=combine, **kwargs)

        if self.args.disable_score:
            score = None
        else:
            paths = self.args.data.split(':')
            assert len(paths) > 0
            data_path = paths[epoch % len(paths)]

            # infer langcode
            src, tgt = self.args.source_lang, self.args.target_lang

            if self.args.score_prop is None:
                props = [None]
            else:
                props = self.args.score_prop.split(':')
            if self.args.score_bound is None:
                bounds = [None]
            else:
                bounds = self.args.score_bound.split(':')

            score = load_score(
                data_path, split, src, props, bounds,
                combine=combine, upsample_primary=self.args.upsample_primary,
            )

        self.datasets[split] = NaoLanguagePairDataset.from_base_dataset(self.datasets[split], score)

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return NaoLanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def valid_step(self, sample, model, criterion):
        if self.args.eval_score_only:
            assert hasattr(criterion, 'eval_score_only'), 'the criterion does not support --eval-score-only mode'
            old_flag = criterion.eval_score_only
            criterion.eval_score_only = True

            loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

            criterion.eval_score_only = old_flag

            return loss, sample_size, logging_output
        else:
            return super().valid_step(sample, model, criterion)

    def predict_step(self, sample, model):
        model.eval()
        with torch.no_grad():
            predict_value = model.encode_and_predict(**sample['net_input'])
        return predict_value

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        if self.args.nao_gen_step:
            # TODO
            pass
        else:
            return super().inference_step(generator, models, sample, prefix_tokens=prefix_tokens)

    def generate_new_seq_step(self):
        # TODO: Add new sequence generation step of NAO tasks.
        pass

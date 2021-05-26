#! /usr/bin/python
# -*- coding: utf-8 -*-


from .translation import TranslationTask
from . import register_task


@register_task('cycle_back_translation')
class DualTranslation(TranslationTask):
    r"""A task for cycle-loop dual translation (back translation).

    Forward model f: x -> y
    Backward model g: y -> x

    Parallel data: (X_p, Y_p)
    Monolingual data: X_m1, Y_m2

    Training process:

        Translation: Y'_p = f(X_p) (train, grad)
        Back translation:
            Y'_m1 = f(X_m1) (gen, no_grad); X''_m1 = g(Y'_m1) = g(f(X_m1)) (forward, grad)
            X'_m2 = g(Y_m2) (gen, no_grad); Y''_m2 = f(X'_m2) = f(g(Y_m2)) (forward, grad)
        Loss: 3 components
            Loss =
            L(Y_p, Y'_p) +
            \lambda1 * L(X_m1, X''_m1) +
            \lambda2 * L(Y_m2, Y''_m2)

    Training process (2):

        Translation: Y'_p = f(X_p) (train, grad)
        Back translation:
            Y'_m1 = f(X_m1) (gen, no_grad)
            X'_m2 = g(Y_m2) (gen, no_grad)
        Loss: 3 components
            L(Y_p, Y'_p) + TODO


    References:

    1. Dual learning tutorial: https://taoqin.github.io/DualLearning_ACML18.pdf
    2. We follow the `bt_step` method of https://github.com/facebookresearch/XLM/blob/master/src/trainer.py#L870
    to write our implementation.
    3. Another reference: https://github.com/apeterswu/RL4NMT/blob/master/tensor2tensor/utils/model_builder.py#L132
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        TranslationTask.add_args(parser)

        # fmt: off
        parser.add_argument('--mono-src', default=None, metavar='PATH', help='Path to monolingual source data')
        parser.add_argument('--mono-tgt', default=None, metavar='PATH', help='Path to monolingual target data')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

        # Monolingual datasets.
        self.mono_src_dataset = None
        self.mono_tgt_dataset = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return super().setup_task(args, **kwargs)

    def _load_monolingual_dataset(self):
        # TODO
        pass

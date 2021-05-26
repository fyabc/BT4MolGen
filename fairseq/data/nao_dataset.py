#! /usr/bin/python
# -*- coding: utf-8 -*-

"""NAO dataset."""

from torch.utils.data import Dataset

from .language_pair_dataset import LanguagePairDataset


def collate_score(samples, batch):
    score = None
    if samples[0].get('score', None) is not None:
        values = [s['score'] for s in samples]
        score = values[0].new(len(values))
        for i, v in enumerate(values):
            score[i].copy_(v)

    batch['score'] = score


class SingleTensorDataset(Dataset):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return len(self.tensor)


class NaoLanguagePairDataset(LanguagePairDataset):
    """LanguagePairDataset + NAO predictor scores.

    Args:
        score (torch.utils.data.Dataset): score dataset to wrap
    """

    def __init__(self, *args, **kwargs):
        score = kwargs.pop('score', None)

        super().__init__(*args, **kwargs)

        self.score = score

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        score_item = self.score[index] if self.score is not None else None

        sample['score'] = score_item
        return sample

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        See `LanguagePairDataset.collater` for more details.

        Returns:
            dict: a mini-batch with the keys in `LanguagePairDataset.collater` and following *extra* keys:

                - `score` (FloatTensor): an 1D Tensor of scores of source sentences.
        """
        batch = super().collater(samples)
        collate_score(samples, batch)
        return batch

    @property
    def supports_prefetch(self):
        return (
            super().supports_prefetch
            and (getattr(self.score, 'supports_prefetch', False) or self.score is None)
        )

    def prefetch(self, indices):
        super().prefetch(indices)
        if self.score is not None:
            self.score.prefetch(indices)

    @classmethod
    def from_base_dataset(cls, base, score):
        """Create dataset from base dataset.

        Args:
            base (LanguagePairDataset): the original dataset
            score (torch.utils.data.Dataset): score dataset to wrap

        Returns:
            NaoLanguagePairDataset:
        """

        return cls(
            base.src, base.src_sizes, base.src_dict,
            tgt=base.tgt, tgt_sizes=base.tgt_sizes, tgt_dict=base.tgt_dict,
            left_pad_source=base.left_pad_source, left_pad_target=base.left_pad_target,
            max_source_positions=base.max_source_positions, max_target_positions=base.max_target_positions,
            shuffle=base.shuffle, input_feeding=base.input_feeding, remove_eos_from_source=base.remove_eos_from_source,
            append_eos_to_target=base.append_eos_to_target,
            score=score,
        )

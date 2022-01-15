from typing import Dict

from datasets.base_label_mapper import BaseLabelMapper


class ReverseLangMnliDatasetMapper(BaseLabelMapper):

    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        self._id2label, self._label2id = self._load_vocab(vocab_file)

    def map(self, item) -> int:
        return self.label2id[item]

    def reverse_map(self, item: int):
        return self.id2label[item]

    @property
    def num_classes(self) -> int:
        return len(self._id2label)

    @property
    def positive_label(self):
        return self.reverse_map(self.positive_label_index)

    @property
    def positive_label_index(self) -> int:
        return 3

    @property
    def label2id(self) -> Dict[str, int]:
        return self._label2id

    @property
    def id2label(self) -> Dict[int, str]:
        return self._id2label

    def _load_vocab(self, vocab_file):
        id2label = {}
        label2id = {}
        with open(vocab_file) as f:
            lines = f.readlines()
        for i, l in enumerate(lines):
            id2label[i] = l
            label2id[l] = i
        return id2label, label2id

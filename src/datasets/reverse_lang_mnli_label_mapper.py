class ReverseLangMnliDatasetMapper:

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def map(self, item) -> int:
        return item

    def reverse_map(self, item: int):
        return item

    @property
    def num_classes(self) -> int:
        return self.vocab_size

    @property
    def positive_label(self):
        return 0

    @property
    def positive_label_index(self) -> int:
        return 0

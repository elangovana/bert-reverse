import os
from unittest import TestCase

from datasets.reverse_lang_mnli_dataset import ReverseLangMnliDataset


class TestReverseLangMnliDataset(TestCase):
    def test_get_item(self):
        # Arrange
        input_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample_mnli.jsonl")
        sut = ReverseLangMnliDataset(input_file)
        expected_x = "Conceptually cream skimming has two basic dimensions - product and geography."
        expected_y = ".yhpargoeg dna tcudorp - snoisnemid cisab owt sah gnimmiks maerc yllautpecnoC"
        # Act
        actual_x, actual_y = sut[0]

        # Assert
        self.assertEqual(actual_x, expected_x)
        self.assertEqual(actual_y, expected_y)

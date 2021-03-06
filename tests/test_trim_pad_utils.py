from unittest import TestCase

import torch

from utils.trim_pad_utils import trim_lpad_confidence, trim_lpad, align_predicted_raw_text, \
    get_raw_token_len_without_pad


class TestTrimPadUtils(TestCase):
    def test_trim_lpad_confidence_no_pad(self):
        data_label = torch.tensor([[1, 2, 3, 4, 5]])
        data_predicted_conf = torch.tensor([[[0.1, 0.2, 0.3],
                                             [0.3, 0.4, 0.5],
                                             [0.7, 0.8, 0.9],
                                             [.3, .3, 0.3],
                                             [.7, .7, 0.7]
                                             ]
                                            ])
        expected_label = torch.tensor([2, 3, 4, 5])
        expected_conf = torch.tensor([
            [0.3, 0.4, 0.5],
            [0.7, 0.8, 0.9],
            [.3, .3, 0.3],
            [.7, .7, 0.7]

        ])

        # Act
        actual_label, actual_conf = trim_lpad_confidence(data_label, data_predicted_conf)

        # Assert
        self.assertTrue(torch.equal(expected_label, actual_label),
                        f"Expected {expected_label} doesnt match actual {actual_label}")
        self.assertTrue(torch.equal(expected_conf, actual_conf),
                        f"Expected \n{expected_conf} \ndoesnt match actual \n{actual_conf}")

    def test_trim_lpad_confidence_pad_multiple(self):
        data_label = torch.tensor([[0, 0, 8, 1, 2], [2, 1, 3, 4, 5]])
        data_predicted_conf = torch.tensor([
            [
                [0.1, 0.2, 0.3],
                [0.3, 0.4, 0.5],
                [0.7, 0.8, 0.9],
                [.3, .3, 0.3],
                [.7, .7, 0.7]
            ],
            [
                [0.1, 0.2, 0.3],
                [0.1, 0.4, 0.5],
                [0.1, 0.8, 0.9],
                [.9, .3, 0.3],
                [.7, .7, 0.6]
            ],
        ])
        expected_label = torch.tensor([8, 1, 2, 1, 3, 4, 5])
        expected_conf = torch.tensor([
            # [0.1, 0.2, 0.3],
            # [0.3, 0.4, 0.5],
            [0.7, 0.8, 0.9],
            [.3, .3, 0.3],
            [.7, .7, 0.7],
            #  [0.1, 0.2, 0.3],
            [0.1, 0.4, 0.5],
            [0.1, 0.8, 0.9],
            [.9, .3, 0.3],
            [.7, .7, 0.6]

        ])

        # Act
        actual_label, actual_conf = trim_lpad_confidence(data_label, data_predicted_conf)

        # Assert
        self.assertTrue(torch.equal(expected_label, actual_label),
                        f"Expected \n{expected_label} \ndoesnt match actual \n{actual_label}")
        self.assertTrue(torch.equal(expected_conf, actual_conf),
                        f"Expected \n{expected_conf} \ndoesnt match actual \n{actual_conf}")

    def test_trim_lpad_confidence_with_pad(self):
        data_label = torch.tensor([[0, 0, 0, 1, 2]])
        data_predicted_conf = torch.tensor([[[0.9, 0.2, 0.3],
                                             [0.8, 0.4, 0.5],
                                             [0.7, 0.8, 0.6],
                                             [.3, .3, 0.3],
                                             [.7, .7, 0.7]
                                             ]
                                            ])
        expected_label = torch.tensor([1, 2])
        expected_conf = torch.tensor([[.3, .3, 0.3],
                                      [.7, .7, 0.7]
                                      ])
        # Act
        actual_label, actual_conf = trim_lpad_confidence(data_label, data_predicted_conf)

        # Assert
        self.assertTrue(torch.equal(expected_label, actual_label),
                        f"Expected {expected_label} doesnt match actual {actual_label}")
        self.assertTrue(torch.equal(expected_conf, actual_conf),
                        f"Expected \n{expected_conf} \ndoesnt match actual \n{actual_conf}")

    def test_trim_lpad_no_pad(self):
        data_x = torch.tensor([[1, 2, 0, 0, 0]])
        data_y = torch.tensor([[1, 2, 3, 4, 5]])
        expected_x = torch.tensor([2, 0, 0, 0])
        expected_y = torch.tensor([2, 3, 4, 5])

        # Act
        actual_x, actual_y = trim_lpad(data_x, data_y)

        # Assert
        self.assertTrue(torch.equal(expected_x, actual_x),
                        f"Expected {expected_x} doesnt match actual {actual_x}")
        self.assertTrue(torch.equal(expected_y, actual_y),
                        f"Expected \n{expected_y} \ndoesnt match actual \n{actual_y}")

    def test_trim_lpad_pad_multiple(self):
        data_x = torch.tensor([[0, 0, 0, 1, 2],
                               [0, 0, 4, 1, 2]
                               ])
        data_y = torch.tensor([[1, 2, 3, 4, 5],
                               [0, 0, 6, 7, 8]
                               ])
        expected_x = torch.tensor([1, 2, 4, 1, 2])
        expected_y = torch.tensor([4, 5, 6, 7, 8])

        # Act
        actual_x, actual_y = trim_lpad(data_x, data_y)

        # Assert
        self.assertTrue(torch.equal(expected_x, actual_x),
                        f"Expected \n{expected_x} doesnt match actual \n{actual_x}")
        self.assertTrue(torch.equal(expected_y, actual_y),
                        f"Expected \n{expected_y} \ndoesnt match actual \n{actual_y}")

    def test_trim_pad_conf_with_pad(self):
        data_x = torch.tensor([[1, 0, 0, 1, 2]])
        data_y = torch.tensor([[0, 2, 3, 4, 5]])
        expected_x = torch.tensor([1, 2])
        expected_y = torch.tensor([4, 5])

        # Act
        actual_x, actual_y = trim_lpad(data_x, data_y)

        # Assert
        self.assertTrue(torch.equal(expected_x, actual_x),
                        f"Expected {expected_x} doesnt match actual {actual_x}")
        self.assertTrue(torch.equal(expected_y, actual_y),
                        f"Expected \n{expected_y} \ndoesnt match actual \n{actual_y}")

    def test_align_predicted_raw_text(self):
        text = ["the", "c", "##ourt"]
        pred = ["[SEP]", "[PAD]", "[PAD]", "##ourt", "c", "the", "[CLS]"]
        expected = "the court"
        token_len = len(text)

        # Act
        actual = align_predicted_raw_text(pred, token_len)

        # Assert
        self.assertEqual(expected, actual)

    def test_get_raw_token_len_without_pad(self):
        # Input with CLS at start and SEP at the end
        input = torch.tensor([3, 2, 4, 0, 0, 1])
        expected_len = 2

        # Act
        actual = get_raw_token_len_without_pad(input, pad_index=0)

        # Assert
        self.assertEqual(expected_len, actual)

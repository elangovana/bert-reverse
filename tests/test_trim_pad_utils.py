from unittest import TestCase

import torch

from utils.trim_pad_utils import trim_lpad_confidence, trim_lpad


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
        expected_label = torch.tensor([1, 2, 3, 4])
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

    def test_trim_lpad_confidence_no_pad_multiple(self):
        data_label = torch.tensor([[0, 0, 8, 1, 2], [2, 0, 3, 4, 5]])
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
        expected_label = torch.tensor([0, 0, 8, 1, 2, 0, 3, 4])
        expected_conf = torch.tensor([
            # [0.1, 0.2, 0.3],
            [0.3, 0.4, 0.5],
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
        data_label = torch.tensor([[1, 2, 0, 0, 0]])
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
        data_x = torch.tensor([[0, 0, 0, 1, 2]])
        data_y = torch.tensor([[1, 2, 3, 4, 5]])
        expected_x = torch.tensor([0, 0, 0, 1])
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
        expected_x = torch.tensor([0, 0, 0, 1, 0, 0, 4, 1])
        expected_y = torch.tensor([2, 3, 4, 5, 0, 6, 7, 8])

        # Act
        actual_x, actual_y = trim_lpad(data_x, data_y)

        # Assert
        self.assertTrue(torch.equal(expected_x, actual_x),
                        f"Expected \n{expected_x} doesnt match actual \n{actual_x}")
        self.assertTrue(torch.equal(expected_y, actual_y),
                        f"Expected \n{expected_y} \ndoesnt match actual \n{actual_y}")

    def test_trim_pad_conf_with_pad(self):
        data_x = torch.tensor([[1, 2, 0, 0, 1]])
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

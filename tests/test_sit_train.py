import json
import os
import tempfile
from unittest import TestCase

import numpy as np


from main_train_pipeline import TrainPipeline


class TestSitTrain(TestCase):

    def test_train_with_no_exception_reverse_lang_mnli(self):
        # Arrange
        train_data_file = os.path.join(os.path.dirname(__file__), "data", "sample_mnli.jsonl")
        batch = 3

        # Bert Config
        vocab_size = 100
        sequence_len = 20
        num_classes = vocab_size

        bert_config = {"vocab_size": vocab_size, "hidden_size": 10, "num_hidden_layers": 1,
                       "tokenisor_max_seq_len": sequence_len, "num_labels": num_classes,
                       "num_attention_heads": 1}
        bert_config_file = self._write_bert_config_file(bert_config)
        # Additional args
        dataset_factory = "datasets.reverse_lang_mnli_dataset_factory.ReverseLangMnliDatasetFactory"
        model_factory = "models.bert_model_factory.BertModelFactory"
        tokenisor_data_dir = os.path.join(os.path.dirname(__file__), "data", "tokensior_data")
        additional_args = {"model_config": bert_config_file,
                           "tokenisor_data_dir": tokenisor_data_dir,
                           "datasetfactory": dataset_factory,
                           "modelfactory": model_factory,
                           "vocab_size":vocab_size,
                           "batch": batch,
                           "numworkers": 1,
                           "epochs": 2,
                           "earlystoppingpatience": 1
                           }

        # Act
        self._run_train(train_data_file, additional_args)



    def _run_train(self, train_data_file, additional_args, tempdir_model=None, tempdir_checkpoint=None):
        tempdir_model = tempdir_model or tempfile.mkdtemp()
        tempdir_checkpoint = tempdir_checkpoint or tempfile.mkdtemp()
        tempdir_out = tempfile.mkdtemp()

        # Runs
        return TrainPipeline().run_train(train_data_file,
                                                train_data_file,
                                                checkpoint_dir=tempdir_checkpoint,
                                                model_dir=tempdir_model,
                                                additional_args=additional_args
                                                )

    def _write_bert_config_file(self, bert_config):
        bert_config_file = os.path.join(tempfile.mkdtemp(), "config.json")
        with open(bert_config_file, "w") as f:
            json.dump(bert_config, f)

        return bert_config_file

import json
import os
import tempfile
from unittest import TestCase

from dataset_builder import DatasetBuilder
from inference.predictor import Predictor
from main_train_pipeline import TrainPipeline
from models.bert_model_factory import BertModelFactory


class TestSitTrain(TestCase):

    def test_train_with_no_exception_reverse_lang_mnli(self):
        # Arrange
        train_data_file = os.path.join(os.path.dirname(__file__), "data", "sample_mnli.jsonl")
        batch = 3
        vocab_file = os.path.join(os.path.dirname(__file__), "data", "tokensior_data", "vocab.txt")

        # Bert Config
        vocab_size = self._get_vocab_size(vocab_file)
        sequence_len = 20
        num_classes = vocab_size
        temp_model_dir = tempfile.mkdtemp()

        bert_config = {"vocab_size": vocab_size, "hidden_size": 10, "num_hidden_layers": 1,
                       "num_labels": num_classes,
                       "num_attention_heads": 1, "max_position_embeddings": sequence_len}
        # Additional args
        dataset_factory = "datasets.reverse_lang_mnli_dataset_factory.ReverseLangMnliDatasetFactory"
        model_factory = "models.bert_model_factory.BertModelFactory"

        bert_config_file = self._write_bert_config_file(bert_config, temp_model_dir)
        tokenisor_data_dir = os.path.dirname(vocab_file)

        additional_args = {"model_config": bert_config_file,
                           "tokenisor_data_dir": tokenisor_data_dir,
                           "datasetfactory": dataset_factory,
                           "modelfactory": model_factory,
                           "vocab_file": vocab_file,
                           "batch": batch,
                           "numworkers": 1,
                           "epochs": 2,
                           "tokenisor_max_seq_len": sequence_len,
                           "earlystoppingpatience": 1
                           }

        # Act
        self._run_train(train_data_file, additional_args, temp_model_dir)

    def test_predict_with_no_exception_reverse_lang_mnli(self):
        # Arrange
        train_data_file = os.path.join(os.path.dirname(__file__), "data", "sample_mnli.jsonl")
        batch = 3
        vocab_file = os.path.join(os.path.dirname(__file__), "data", "tokensior_data", "vocab.txt")

        # Bert Config
        vocab_size = self._get_vocab_size(vocab_file)
        sequence_len = 20
        num_classes = vocab_size
        temp_model_dir = tempfile.mkdtemp()

        bert_config = {"vocab_size": vocab_size, "hidden_size": 10, "num_hidden_layers": 1,
                       "num_labels": num_classes,
                       "num_attention_heads": 1, "max_position_embeddings": sequence_len}

        # Additional args
        dataset_factory = "datasets.reverse_lang_mnli_dataset_factory.ReverseLangMnliDatasetFactory"
        model_factory = "models.bert_model_factory.BertModelFactory"
        tokenisor_data_dir = os.path.dirname(vocab_file)
        bert_config_file = self._write_bert_config_file(bert_config, temp_model_dir)

        train_additional_args = {"model_config": bert_config_file,
                                 "tokenisor_data_dir": tokenisor_data_dir,
                                 "datasetfactory": dataset_factory,
                                 "modelfactory": model_factory,
                                 "vocab_file": vocab_file,
                                 "batch": batch,
                                 "numworkers": 1,
                                 "epochs": 2,
                                 "tokenisor_max_seq_len": sequence_len,
                                 "earlystoppingpatience": 1
                                 }

        # train
        self._run_train(train_data_file, train_additional_args, tempdir_model=temp_model_dir)

        # Prepare data for predict
        predict_additional_args = {}
        train_args = {**train_additional_args, **predict_additional_args}

        # Dataset Builder
        model_factory_name = train_args["modelfactory"]
        dataset_builder = DatasetBuilder(val_data=train_data_file, dataset_factory_name=train_args["datasetfactory"],
                                         tokenisor_factory_name=model_factory_name,
                                         num_workers=1, batch_size=batch,
                                         addition_args_dict=train_args)
        model_factory = BertModelFactory()
        model = model_factory.get_model(dataset_builder.num_classes, checkpoint_dir=temp_model_dir, **train_args)

        sut = Predictor()

        # Act
        predictions_data = sut.predict(model, dataset_builder.get_val_dataloader())

        # Assert
        self.assertSequenceEqual(predictions_data[0].shape, predictions_data[1].shape)

    def _run_train(self, train_data_file, additional_args, tempdir_model, tempdir_checkpoint=None):
        tempdir_checkpoint = tempdir_checkpoint or tempfile.mkdtemp()

        # Runs
        return TrainPipeline().run_train(train_data_file,
                                         train_data_file,
                                         checkpoint_dir=tempdir_checkpoint,
                                         model_dir=tempdir_model,
                                         additional_args=additional_args
                                         )

    def _write_bert_config_file(self, bert_config, model_dir):
        bert_config_file = os.path.join(model_dir, "config.json")
        with open(bert_config_file, "w") as f:
            json.dump(bert_config, f)

        return bert_config_file

    def _get_vocab_size(self, vocab_file):
        with open(vocab_file) as f:
            return len(f.readlines())

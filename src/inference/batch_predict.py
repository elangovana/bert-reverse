import glob
import json
import logging
import os
import tarfile
import traceback
from pathlib import Path
from typing import Dict

from dataset_builder import DatasetBuilder
from inference.predictor import Predictor
from locator import Locator
from utils.trim_pad_utils import align_predicted_raw_text, get_raw_token_len_without_pad


class BatchPredict:

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def predict_from_directory(self, datajson, base_artefacts_dir, is_ensemble, output_dir=None, numworkers=None,
                               batch=32, additional_args=None, raw_data_reader_func=None,
                               get_raw_text_token_len_func=None):
        data_files = [datajson]
        if os.path.isdir(datajson):
            data_files = glob.glob("{}/*.jsonl".format(datajson))

        for d in data_files:
            output_file = "{}.jsonl".format(os.path.join(output_dir, Path(d).name)) if output_dir else None
            self._logger.info("Running inference on file {} with output in {}".format(d, output_file))
            try:
                prediction = self.predict_from_file(d, base_artefacts_dir, is_ensemble, output_file, numworkers, batch,
                                                    additional_args, raw_data_reader_func, get_raw_text_token_len_func)

                yield prediction
            except AssertionError as ex:
                self._logger.warning("Failed processing file {} due to error {}".format(d, traceback.format_exc()))
                yield None

    def predict_from_file(self, data_file, base_artifacts_dir, is_ensemble, output_file=None, numworkers=None, batch=32,
                          additional_args=None, raw_data_reader_func=None, get_raw_text_token_len_func=None):
        additional_args = additional_args or {}

        self._logger.info(f"Processing data file {data_file}")

        artifacts_directories = []
        if is_ensemble:
            for d in os.listdir(base_artifacts_dir):
                artifacts_dir = os.path.join(base_artifacts_dir, d)
                artifacts_directories.append(artifacts_dir)
        else:
            artifacts_directories = [base_artifacts_dir]

        # Extract gz
        for artifacts_dir in artifacts_directories:
            # Persist params
            dir_contents = os.listdir(artifacts_dir)

            if len(dir_contents) == 1 and dir_contents[0].endswith("tar.gz"):
                self._extract_file(os.path.join(artifacts_dir, dir_contents[0]))

        # Load params
        output_config = os.path.join(artifacts_directories[0], "training_config_parameters.json")
        with open(output_config, "r") as f:
            train_args = json.load(f)

        train_args = {**train_args, **additional_args}

        self._logger.info("Using args :{}".format(train_args))

        # Dataset Builder
        model_factory_name = train_args["modelfactory"]
        dataset_builder = DatasetBuilder(val_data=data_file, dataset_factory_name=train_args["datasetfactory"],
                                         tokenisor_factory_name=model_factory_name,
                                         num_workers=numworkers, batch_size=batch,
                                         addition_args_dict=train_args)

        # Load ensemble
        models = []
        for artifact_dir in artifacts_directories:
            output_config = os.path.join(artifacts_directories[0], "training_config_parameters.json")
            with open(output_config, "r") as f:
                model_train_args = json.load(f)

            model_factory = Locator().get(model_factory_name)
            model = model_factory.get_model(dataset_builder.num_classes, checkpoint_dir=artifact_dir,
                                            **model_train_args)
            models.append(model)

        model = models[0]
        predictions_data = Predictor().predict(model,
                                               dataset_builder.get_val_dataloader())

        raw_data_iter = raw_data_reader_func(data_file) if raw_data_reader_func else None
        self.write_results_to_file(predictions_data, dataset_builder.get_label_mapper(),
                                   output_file,
                                   raw_data_iter,
                                   dataset_builder.get_tokenisor(),
                                   get_raw_text_token_len_func)

        self._logger.info(f"Completed file {data_file}")

        return predictions_data

    def write_results_to_file(self, predictions_data_tuple, label_mapper,
                              output_file,
                              raw_data_iter=None, tokenisor=None, get_raw_text_token_len_func=None):

        result = []

        predictions_tensor = predictions_data_tuple[0]
        confidence_scores = predictions_data_tuple[1]

        if raw_data_iter is None:
            raw_data_iter = [None] * len(predictions_tensor)

        assert len(raw_data_iter) == len(
            predictions_tensor), "The length of raw data iterator {} doesnt match the prediction len {}".format(
            len(raw_data_iter), len(predictions_tensor))

        # Convert indices to labels
        for i, raw_data in enumerate(raw_data_iter):

            pred_i_tensor = predictions_tensor[i]
            conf_i_tensor = confidence_scores[i]

            pred_i = pred_i_tensor.cpu().tolist()
            conf_i = conf_i_tensor.cpu().tolist()

            label_mapped_prediction = [label_mapper.reverse_map(pi) for pi in pred_i]
            predicted_raw_text = None
            raw_data_token_mapped = None
            predicted_confidence = conf_i

            if isinstance(raw_data, str) and tokenisor is not None and get_raw_text_token_len_func is not None:
                raw_tokens = tokenisor(raw_data)
                raw_data_token_mapped = [label_mapper.reverse_map(ri) for ri in raw_tokens.cpu().tolist()]
                raw_token_len = get_raw_token_len_without_pad(raw_tokens)
                predicted_raw_text = align_predicted_raw_text(label_mapped_prediction,
                                                              raw_token_len)

            r = {
                "prediction": label_mapped_prediction,
                #   "confidence": predicted_confidence,
                "predicted_raw_text": predicted_raw_text,
                "raw_data_tokens": raw_data_token_mapped
            }

            r = {**r}

            # Add raw data if available
            if isinstance(raw_data, Dict):
                r = {**raw_data, **r}
            else:
                r["raw_data"] = raw_data

            result.append(r)

        self._logger.info("Records to write: {}".format(len(result)))

        if len(result) > 0:
            self._logger.info(f"Writing to file {output_file}")
            # Write json to file
            with open(output_file, "w") as f:
                for item in result:
                    json.dump(item, f)
                    f.write("\n")

    # def _get_predicted_raw_text(self, label_mapped_prediction):
    #     predicted_raw_text = ""
    #     last_pad_position = None
    #     for ti, t in enumerate(label_mapped_prediction):
    #
    #         if t.startswith("##"):
    #             t = t[2:]
    #         elif ti != 0:
    #             predicted_raw_text = predicted_raw_text + " "
    #
    #         predicted_raw_text = predicted_raw_text + t
    #
    #         # Reset pad position, and get continuous one, after the first token #SEP
    #         if t != "[PAD]" and last_pad_position is None and ti > 0:
    #             last_pad_position = len(predicted_raw_text) - 1
    #
    #     if predicted_raw_text[0:5] == '[SEP]':
    #         predicted_raw_text = predicted_raw_text[6:]
    #         last_pad_position = last_pad_position - 6
    #     if last_pad_position is not None:
    #         predicted_raw_text = predicted_raw_text[last_pad_position:]
    #
    #     if predicted_raw_text.endswith('[CLS]'):
    #         predicted_raw_text = predicted_raw_text[:-5]
    #     return predicted_raw_text

    def _extract_file(self, targzfile, dest=None):

        dest = dest or os.path.dirname(targzfile)
        with tarfile.open(targzfile) as f:
            # extracting file
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, dest)

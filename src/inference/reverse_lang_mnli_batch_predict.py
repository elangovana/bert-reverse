import argparse
import json
import logging
import sys

from inference.batch_predict import BatchPredict
from utils.trim_pad_utils import get_raw_token_len_without_pad


class ReverseLangMnliBatchPredict:

    def __init__(self, batch_predict):
        self.batch_predict = batch_predict

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def _get_mnli_inference_reader(self, data_File):
        lines = []
        with open(data_File) as f:
            for l in f:
                data_l = json.loads(l)
                lines.append(data_l["sentence1"])
                lines.append(data_l["sentence2"])
        return lines

    def predict_from_directory(self, datajson, base_artefacts_dir, is_ensemble, output_dir, numworkers=None, batch=32,
                               additional_args=None, raw_data_reader_func=None):
        raw_data_reader_func = raw_data_reader_func or self._get_mnli_inference_reader

        get_raw_text_token_len_func = lambda x: get_raw_token_len_without_pad(x, pad_index=0)

        # Invoke underlying batch predict
        return list(self.batch_predict.predict_from_directory(datajson, base_artefacts_dir,
                                                              is_ensemble, output_dir=output_dir,
                                                              numworkers=numworkers, batch=batch,
                                                              additional_args=additional_args,
                                                              raw_data_reader_func=raw_data_reader_func,
                                                              get_raw_text_token_len_func=get_raw_text_token_len_func))

    def predict_from_file(self, datajson, base_artefacts_dir, is_ensemble, output_file, numworkers=None, batch=32,
                          additional_args=None, raw_data_reader_func=None, get_raw_text_token_len_func=None):
        raw_data_reader_func = raw_data_reader_func or self._get_mnli_inference_reader
        return self.batch_predict.predict_from_file(datajson, base_artefacts_dir,
                                                    is_ensemble, output_file=output_file,
                                                    numworkers=numworkers, batch=batch,
                                                    additional_args=additional_args,
                                                    raw_data_reader_func=raw_data_reader_func,
                                                    get_raw_text_token_len_func=get_raw_text_token_len_func
                                                    )


def parse_args_run():
    global args, additional_dict
    parser = argparse.ArgumentParser()
    parser.add_argument("datajson",
                        help="The json data to predict")
    parser.add_argument("artefactsdir", help="The base of artefacts dir that contains directories of model, vocab etc")
    parser.add_argument("outdir", help="The output dir")
    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})
    parser.add_argument("--numworkers", help="The number of workers to use", type=int, default=None)
    parser.add_argument("--batch", help="The batchsize", type=int, default=32)
    parser.add_argument("--ensemble", help="Set to 1 if ensemble model", type=int, default=0, choices={0, 1})

    args, additional_args = parser.parse_known_args()
    print(args.__dict__)
    # Convert additional args into dict
    additional_dict = {}
    for i in range(0, len(additional_args), 2):
        additional_dict[additional_args[i].lstrip("--")] = additional_args[i + 1]
    print(additional_dict)

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ReverseLangMnliBatchPredict(BatchPredict()).predict_from_directory(args.datajson,
                                                                       args.artefactsdir,
                                                                       args.ensemble,
                                                                       args.outdir,
                                                                       args.numworkers,
                                                                       args.batch,
                                                                       additional_dict)


if "__main__" == __name__:
    parse_args_run()

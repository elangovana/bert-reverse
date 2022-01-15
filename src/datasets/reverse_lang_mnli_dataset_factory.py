import logging
from typing import List

from datasets.base_dataset_factory import BaseDatasetFactory
from datasets.reverse_lang_mnli_dataset import ReverseLangMnliDataset
from datasets.reverse_lang_mnli_label_mapper import ReverseLangMnliDatasetMapper
from datasets.transformer_chain import TransformerChain
from scorers.result_scorer_accuracy_factory import ResultScorerAccuracyFactory


class ReverseLangMnliDatasetFactory(BaseDatasetFactory):

    def get_scorers(self):
        scores = [ResultScorerAccuracyFactory().get(),

                  ]
        return scores

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def get_label_mapper(self, data=None, postprocessors=None, **kwargs):
        vocab_file = self._get_value(kwargs, "vocab_file", None)
        assert vocab_file is not None, "The vocab_file parameter is mandatory"
        return ReverseLangMnliDatasetMapper(vocab_file)

    def get_dataset(self, data, postprocessors=None, **kwargs):

        transformer_list = [ ]

        postprocessors = postprocessors or []
        if not isinstance(postprocessors, List):
            postprocessors = [postprocessors]

        # Add additional preprocessors
        transformer_list = transformer_list + postprocessors

        transformer_chain = TransformerChain(transformer_list)
        return ReverseLangMnliDataset(data, input_transformer=transformer_chain)

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self._logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

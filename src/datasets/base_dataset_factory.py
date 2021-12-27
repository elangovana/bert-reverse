class BaseDatasetFactory:
    def get_dataset(self, data, postprocessors=None, **kwargs):
        raise NotImplementedError

    def get_label_mapper(self, data=None, postprocessors=None, **kwargs):
        raise NotImplementedError

    def get_scorers(self):
        raise NotImplementedError

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import Token
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register('tg-san-predictor')
class TgSanPredictor(Predictor):
    """
    tg-san predictor
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, inputs: JsonDict) -> JsonDict:
        """
        Parameters
        ----------
        inputs: {"text": "", "head": {"begin": 1, "end": 4}, "tail": {"begin": 1, "end": 4}, "relation": ""}

        Returns
        -------

        """
        return self.predict_json(inputs)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["text"]

        head_begin = json_dict["head"]["begin"]
        head_end = json_dict["head"]["end"]

        tail_begin = json_dict["tail"]["begin"]
        tail_end = json_dict["tail"]["end"]

        relation = json_dict["relation"]

        tokens = [Token(token) for token in sentence]

        return self._dataset_reader.text_to_instance(tokens, relation, head_begin, head_end, tail_begin, tail_end)

from typing import Dict, List, Sequence, Iterable, Any
import itertools
import logging
import json

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, LabelField, Field, MetadataField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("tg-san-reader")
class TgSanDatasetReader(DatasetReader):
    """
    tg-san dataset reader
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 label_namespace: str = "labels") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        self.label_namespace = label_namespace

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            for line in data_file:
                json_data = json.loads(line.strip())
                sentence = json_data["text"]

                # positive or negative
                label = json_data["label"]
                head_begin = json_data["head"]["begin"]
                head_end = json_data["head"]["end"]

                tail_begin = json_data["tail"]["begin"]
                tail_end = json_data["tail"]["end"]

                relation = json_data["relation"]

                tokens = [Token(token) for token in sentence]

                yield self.text_to_instance(tokens, relation, head_begin, head_end, tail_begin, tail_end, label)

    def text_to_instance(self, # type: ignore
                         tokens: List[Token],
                         relation: str,
                         head_begin: int,
                         head_end: int,
                         tail_begin: int,
                         tail_end: int,
                         label: str = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': sequence}
        instance_fields["metadata"] = MetadataField({"head_begin": head_begin,
                                                     "head_end": head_end,
                                                     "tail_begin": tail_begin,
                                                     "tail_end": tail_end})

        if label is not None:
            instance_fields['label'] = LabelField(label)

        instance_fields['relation_label'] = LabelField(relation, "relation_labels")

        return Instance(instance_fields)


@DatasetReader.register("contrastive-reader")
class ContrastiveDatasetReader(DatasetReader):
    """
    contrastive dataset reader
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 label_namespace: str = "labels") -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        self.label_namespace = label_namespace

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            for line in data_file:
                json_data = json.loads(line.strip())
                positive_samples = json_data["positive"]
                negative_samples = json_data["negative"]

                yield self.text_to_instance(positive_samples, negative_samples)

    def text_to_instance(self, # type: ignore
                         positive_samples: List[Any],
                         negative_samples: List[Any]) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        data_token_fields = list()
        data_relation_label_fields = list()
        data_meta_fields = list()
        data_label_fields = list()
        positive_samples.extend(negative_samples)

        for item in positive_samples:
            text = item["text"]
            head_begin = item["head"]["begin"]
            head_end = item["head"]["end"]
            tail_begin = item["tail"]["begin"]
            tail_end = item["tail"]["end"]
            tokens = [Token(token) for token in text]
            data_token_fields.append(TextField(tokens, self._token_indexers))
            relation = item["relation"]
            data_relation_label_fields.append(LabelField(relation, "relation_labels"))
            label = item["label"]
            data_label_fields.append(LabelField(label))
            data_meta_fields.append(MetadataField({"head_begin": head_begin,
                                                     "head_end": head_end,
                                                     "tail_begin": tail_begin,
                                                     "tail_end": tail_end}))



        data_token_fields = ListField(data_token_fields)
        data_relation_label_fields = ListField(data_relation_label_fields)
        data_meta_fields = ListField(data_meta_fields)
        data_label_fields = ListField(data_label_fields)

        instance_fields = {'tokens': data_token_fields}
        instance_fields["metadata"] = data_meta_fields
        instance_fields["label"] = data_label_fields
        instance_fields['relation_label'] = data_relation_label_fields

        return Instance(instance_fields)
#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

"""
brief "emnlp model code"
"""
from typing import Iterator, List, Dict
from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from tg_san_contrastive_model.predictor.tg_san_predictor import TgSanPredictor


class TgsanContrastivePredictor(object):
    """
    Predictor class for negoation.
    """
    def __init__(self):
        self._predictor = None

    def load_model(self) -> None:
        """
        加载模型
        :return: None
        """
        if self._predictor is None:
            # 模型保存路径,自定义设置
            serialization_dir = './archive_file'
            config = Params.from_file(serialization_dir + "/config.json")
            model = Model.load(config.duplicate(),
                               weights_file=serialization_dir + "/best.th",
                               serialization_dir=serialization_dir,
                               cuda_device=0)
            model.eval()
            dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
            predictor = TgSanPredictor(model, dataset_reader=dataset_reader)
            self._predictor = predictor

    def predict(self, input_texts: List[str]) -> List[Dict]:
        """
        对输入一个batch句子进行预测，得到一个batch的结果列表
        :param input_texts: 输入句子文本列表
        :return: 结果列表
        """
        if self._predictor is None:
            raise RuntimeError("before predict you must use method load_model")
        json_list = list()
        result_list = list()
        for sentence in input_texts:
            json_item = sentence
            json_list.append(json_item)
        tag_logits_list = self._predictor.predict_batch_json(json_list)
        return tag_logits_list


if __name__ == "__main__":
    tcp = TgsanContrastivePredictor()
    tcp.load_model()
    texts = [{"text": "部分患者合并胸腹水及CA125升高,易误诊为恶性肿瘤.", 
              "head": {"begin": 22 , "end": 26}, 
              "tail": {"begin": 10, "end": 17}, 
              "relation": "并发症"},
             {"text": "结论乳腺癌根治术后预防皮瓣下积液的关键是:防止淋巴管瘘,合理使用电刀,置双引流管并负压吸引,胸带加压包扎.", 
              "head": {"begin": 11 , "end": 16}, 
              "tail": {"begin": 21, "end": 27}, 
              "relation": "预防"}
            ]      
    result = tcp.predict(texts)
    print(result)

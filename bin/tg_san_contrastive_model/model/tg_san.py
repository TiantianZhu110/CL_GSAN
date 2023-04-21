# coding: utf-8
from typing import Dict, Optional, List, Any

from overrides import overrides
import torch
import random
from torch.nn.modules.linear import Linear
from torch.nn.parameter import Parameter

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, Average


@Model.register("tg-san-model")
class TgSan(Model):
    """
    否定识别模型基于门控机制TG-SAN

    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 contrastive_encoder: Seq2SeqEncoder,
                 feedforward: Optional[FeedForward],
                 r: int = 3,
                 d_wt: int = 50,
                 relation_embedding_size: int = 30,
                 label_namespace: str = "labels",
                 relation_label_namespace: str = "relation_labels",
                 is_contrastive: bool = False,
                 dropout: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        self.relation_label_namespace = relation_label_namespace
        self.is_contrastive = is_contrastive
        self.contrastive_encoder = contrastive_encoder
        self.text_field_embedder = text_field_embedder
        self.num_relation = self.vocab.get_vocab_size(self.relation_label_namespace)
        self.num_label = self.vocab.get_vocab_size(self.label_namespace)
        self.encoder = encoder
        self.r = r
        self.d_wt = d_wt

        self.encoder_output_dim = self.encoder.get_output_dim()

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self._feedforward = feedforward

        self.feed_output_dim = feedforward.get_output_dim()

        self.relation_embedding_size = relation_embedding_size
        self.relation_embedding = torch.nn.Embedding(self.num_relation, relation_embedding_size)

        self.Wt_1 = Parameter(torch.Tensor(d_wt, self.encoder_output_dim))
        self.Wt_2 = Parameter(torch.Tensor(r, d_wt))

        self.Wc = Parameter(torch.Tensor(self.encoder_output_dim + relation_embedding_size, self.encoder_output_dim))

        self.layer_norm = torch.nn.LayerNorm([r, self.encoder_output_dim])

        self.Wm_1 = Parameter(torch.Tensor(d_wt, self.encoder_output_dim + relation_embedding_size))
        self.wm_2 = Parameter(torch.Tensor(1, d_wt))

        self.U_context = Parameter(torch.Tensor(self.encoder_output_dim,
                                                self.encoder_output_dim + relation_embedding_size))

        self.REL_context = Parameter(torch.Tensor(self.encoder_output_dim, self.relation_embedding_size))

        self.gate_W_f = Linear(self.encoder_output_dim, self.encoder_output_dim)

        self.W_f = Linear(2 * self.encoder_output_dim, self.encoder_output_dim)
        self.W_f_2 = Linear(2 * self.encoder_output_dim + self.relation_embedding_size, self.encoder_output_dim)

        self.projection_layer = Linear(self.encoder_output_dim, self.num_label)

        self.f1_metric = F1Measure(self.vocab.get_token_index("negative", self.label_namespace))
        self.ave_metric = Average()

        self.criterion = torch.nn.CrossEntropyLoss(torch.tensor([0.3, 0.7]))

        initializer(self)

    def model_encoder_output(self,
                             tokens: Dict[str, torch.LongTensor],
                             relation_label: torch.LongTensor = None,
                             metadata: List[Dict[str, Any]] = None):
        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, mask)

        contrastive_encoded_text = self.contrastive_encoder(embedded_text_input, mask)
        encoded_text = encoded_text + contrastive_encoded_text

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        batch_context_memory = list()
        batch_target_memory = list()

        for instance_encoded_text, instance_meta, instance_mask in zip(encoded_text, metadata, mask):
            head_begin = instance_meta["head_begin"]
            head_end = instance_meta["head_end"]
            tail_begin = instance_meta["tail_begin"]
            tail_end = instance_meta["tail_end"]

            if head_begin < tail_begin:
                begin_1 = head_begin
                end_1 = head_end
                begin_2 = tail_begin
                end_2 = tail_end
            else:
                begin_1 = tail_begin
                end_1 = tail_end
                begin_2 = head_begin
                end_2 = head_end
            mask_sum = int(torch.sum(instance_mask))
            target_vector = torch.cat([instance_encoded_text[begin_1: end_1],
                                       instance_encoded_text[begin_2: end_2]], dim=0)
            context_vector = torch.cat([instance_encoded_text[0: begin_1], instance_encoded_text[end_1: begin_2],
                                        instance_encoded_text[end_2: mask_sum]], dim=0)
            batch_context_memory.append(context_vector)
            batch_target_memory.append(target_vector)
        relation_label_embedding = self.relation_embedding(relation_label)

        batch_Rt = list()
        batch_P = list()
        batch_Rc = list()
        batch_ac = list()
        for ins_rel_emb, instance_target_memory, instance_context_memory in zip(relation_label_embedding,
                                                                                batch_target_memory,
                                                                                batch_context_memory):
            att_linear = torch.nn.functional.tanh(torch.matmul(self.Wt_1, instance_target_memory.t()))
            At = torch.nn.functional.softmax(torch.matmul(self.Wt_2, (att_linear)))
            Rt = torch.matmul(At, instance_target_memory)

            rel_embedding = ins_rel_emb.expand(self.r, self.relation_embedding_size)
            Rt = torch.cat([Rt, rel_embedding], -1)

            P_mat = torch.matmul(At, At.t()) - torch.eye(self.r).to(Rt.device)
            P = torch.norm(P_mat) ** 2
            batch_P.append(P)

            Ac = torch.matmul(Rt, torch.matmul(self.Wc, instance_context_memory.t()))
            Ac = torch.nn.functional.softmax(Ac)
            Rc = torch.matmul(Ac, instance_context_memory)
            Rc = torch.unsqueeze(Rc, 0)
            batch_Rc.append(Rc)
            Rt = torch.unsqueeze(Rt, 0)
            batch_Rt.append(Rt)
        batch_Rt = torch.cat(batch_Rt, 0)
        batch_P = torch.tensor(batch_P)
        batch_Rc = torch.cat(batch_Rc, 0)

        Rc_f = torch.nn.functional.relu(self._feedforward(batch_Rc))
        batch_Rc = self.layer_norm(batch_Rc + Rc_f)

        batch_Rt_transpose = torch.transpose(batch_Rt, 1, 2)


        a_t = torch.nn.functional.softmax(torch.matmul(self.wm_2, torch.nn.functional.tanh(
            torch.matmul(self.Wm_1, batch_Rt_transpose))), dim=-1)
        r_t = torch.matmul(a_t, batch_Rt)

        rt_transpose = torch.transpose(r_t, 1, 2)
        att_score = torch.matmul(batch_Rc, torch.matmul(self.U_context, rt_transpose))
        att_score = torch.nn.functional.softmax(att_score, dim=1)
        att_transpose = torch.transpose(att_score, 1, 2)
        r_c = torch.matmul(att_transpose, batch_Rc)

        r_rel_att = torch.matmul(torch.matmul(batch_Rc, self.REL_context),
                                 torch.unsqueeze(relation_label_embedding, -1))
        r_rel_att_transpose = torch.transpose(r_rel_att, 1, 2)
        r_rel = torch.matmul(r_rel_att_transpose, batch_Rc)

        r_t = torch.squeeze(r_t, 1)
        r_c = torch.squeeze(r_c, 1)
        r_rel = torch.squeeze(r_rel, 1)

        tanh_r_c = torch.nn.functional.tanh(self.gate_W_f(r_c))
        tanh_r_rel = torch.nn.functional.tanh(self.gate_W_f(r_rel))

        sigmod_r_c = torch.nn.functional.sigmoid(r_c)
        sigmod_r_rel = torch.nn.functional.sigmoid(r_rel)
        G_1 = (sigmod_r_c + sigmod_r_rel) / 2
        G_1[G_1 < 0.5] = 0.0
        r_c_significant_feature = G_1 * tanh_r_c
        r_rel_significant_feature = G_1 * tanh_r_rel
        r_c_fushion_feature = r_rel_significant_feature + r_c
        r_rel_fushion_feature = r_c_significant_feature + r_rel
        r_global_fushion_context = torch.cat([r_c_fushion_feature, r_rel_fushion_feature], -1)
        r_global_fushion_context = self.W_f(r_global_fushion_context)

        r_ct = self.W_f_2(torch.cat([r_global_fushion_context, r_t], -1))
        return r_ct, batch_P

    def contrastive_output(self,
                        tokens: Dict[str, torch.LongTensor]):
        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)
        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.contrastive_encoder(embedded_text_input, mask)
        return torch.mean(encoded_text, dim=1)

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                relation_label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        tg-san forward
        """
        if not self.is_contrastive:
            r_ct, batch_P = self.model_encoder_output(tokens, relation_label, metadata)
            logits = self.projection_layer(r_ct)
            predict_label = torch.argmax(logits, dim=1)

            logits_norm = torch.nn.functional.softmax(logits, dim=1)

            output = {"logits": logits, "predict_labels": predict_label}

            if label is not None:
                self.f1_metric(logits, label)
                l2_regularization = torch.tensor(0.0).to(r_ct.device)
                for param in self.parameters():
                    l2_regularization += torch.norm(param) ** 2

                loss = self.criterion(logits, label) + 0.0006 * torch.sum(batch_P)
                output["loss"] = loss

            return output
        else:
            output = {}
            cluster_positive_length = torch.sum(label == self.vocab.get_token_index("positive", self.label_namespace), dim=1)
            cluster_negative_length = torch.sum(label == self.vocab.get_token_index("negative", self.label_namespace), dim=1)
            cluster_length = cluster_negative_length + cluster_positive_length

            tokens_list = []
            label_list = []
            relation_label_list = []

            for cluster_index, real_value in enumerate(cluster_length):
                real_value = int(real_value)
                cluster_tokens_dict = {}
                for token_key, token_val in tokens.items():
                    cluster_tokens_dict[token_key] = tokens[token_key][cluster_index][0: real_value]
                tokens_list.append(cluster_tokens_dict)

                label_list.append(label[cluster_index][0: real_value])
                relation_label_list.append(relation_label[cluster_index][0: real_value])

            total_loss = 0.0
            for cluster_index in range(len(cluster_length)):
                cluster_rep = self.contrastive_output(tokens_list[cluster_index])
                cluster_pos_rep = cluster_rep[0: cluster_positive_length[cluster_index]]
                cluster_neg_rep = cluster_rep[cluster_positive_length[cluster_index]: ]

                pos_rand_list = [i for i in range(len(cluster_pos_rep))]
                random.shuffle(pos_rand_list)
                pos_rep_1 = cluster_pos_rep[pos_rand_list[0]].view(1, -1)
                pos_rep_2 = cluster_pos_rep[pos_rand_list[1]].view(1, -1)

                neg_rand_list = [i for i in range(len(cluster_neg_rep))]
                random.shuffle(neg_rand_list)
                neg_rep_1 = cluster_neg_rep[neg_rand_list[0]].view(1, -1)
                neg_rep_2 = cluster_neg_rep[neg_rand_list[1]].view(1, -1)

                pos_molecular = torch.exp(torch.cosine_similarity(pos_rep_1, pos_rep_2, dim=1))[0]
                pos_denominator =  pos_molecular + torch.sum(torch.exp(torch.cosine_similarity(pos_rep_1, cluster_neg_rep, dim=1)))
                pos_loss =  -torch.log(pos_molecular / pos_denominator)

                neg_molecular = torch.exp(torch.cosine_similarity(neg_rep_1, neg_rep_2, dim=1))[0]
                neg_denominator =  neg_molecular + torch.sum(torch.exp(torch.cosine_similarity(neg_rep_1, cluster_pos_rep, dim=1)))
                neg_loss =  -torch.log(neg_molecular / neg_denominator)

                total_loss += (neg_loss + pos_loss)
            output["loss"] = total_loss / len(cluster_length)
            self.ave_metric(output["loss"])
            return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        if not self.is_contrastive:
            output_dict["predict_labels"] = [self.vocab.get_token_from_index(int(tag), namespace=self.label_namespace)
                     for tag in output_dict["predict_labels"]]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}
        if not self.is_contrastive:
            f1_dict = self.f1_metric.get_metric(reset=reset)
            metrics_to_return.update({"precision": f1_dict[0], "recall": f1_dict[1], "f1": f1_dict[2]})
        else:
            ave_loss = self.ave_metric.get_metric(reset=reset)
            metrics_to_return["ave_loss"] = float(ave_loss)
        return metrics_to_return

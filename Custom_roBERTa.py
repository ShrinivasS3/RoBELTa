from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from transformers import BatchEncoding, PreTrainedTokenizerBase

from roBERTa import RobertaClassifier
from Split import transform_list_of_texts


class RobertaClassifierPooling(RobertaClassifier):
    """set device as cuda and many_gpus as True finally"""
    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        chunk_size: int,
        stride: int,
        minimal_chunk_length: int,
        pooling_strategy: str = "mean",
        maximal_text_length: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        neural_network: Optional[Module] = None,
        pretrained_model_name_or_path: str = "roberta-base",
        device: str = "mps",
        many_gpus: bool = True,
    ):
        super().__init__(
            batch_size,
            learning_rate,
            epochs,
            tokenizer,
            neural_network,
            pretrained_model_name_or_path,
            device,
            many_gpus,
        )

        self.chunk_size = chunk_size
        self.stride = stride
        self.minimal_chunk_length = minimal_chunk_length
        if pooling_strategy in ["mean", "max"]:
            self.pooling_strategy = pooling_strategy
        else:
            raise ValueError("Unknown pooling strategy!")
        self.maximal_text_length = maximal_text_length

        additional_params = {
            "chunk_size": self.chunk_size,
            "stride": self.stride,
            "minimal_chunk_length": self.minimal_chunk_length,
            "pooling_strategy": self.pooling_strategy,
            "maximal_text_length": self.maximal_text_length,
        }
        self._params.update(additional_params)

        self.device = device
        self.collate_fn = RobertaClassifierPooling.collate_fn_pooled_tokens

    def _tokenize(self, texts: list[str]) -> BatchEncoding:

        tokens = transform_list_of_texts(
            texts, self.tokenizer, self.chunk_size, self.stride, self.minimal_chunk_length, self.maximal_text_length
        )
        return tokens

    def _evaluate_single_batch(self, batch: tuple[Tensor]) -> Tensor:
        input_ids = batch[0]
        attention_mask = batch[1]
        number_of_chunks = [len(x) for x in input_ids]

        input_ids_combined = []
        for x in input_ids:
            input_ids_combined.extend(x.tolist())

        input_ids_combined_tensors = torch.stack([torch.tensor(x).to(self.device) for x in input_ids_combined])

        attention_mask_combined = []
        for x in attention_mask:
            attention_mask_combined.extend(x.tolist())

        attention_mask_combined_tensors = torch.stack(
            [torch.tensor(x).to(self.device) for x in attention_mask_combined]
        )

        preds = self.neural_network(input_ids_combined_tensors, attention_mask_combined_tensors)

        preds = preds.flatten().cpu()

        preds_split = preds.split(number_of_chunks)

        if self.pooling_strategy == "mean":
            pooled_preds = torch.cat([torch.mean(x).reshape(1) for x in preds_split])
        elif self.pooling_strategy == "max":
            pooled_preds = torch.cat([torch.max(x).reshape(1) for x in preds_split])
        else:
            raise ValueError("Unknown pooling strategy!")

        return pooled_preds
    @staticmethod
    def collate_fn_pooled_tokens(data):
        input_ids = [data[i][0] for i in range(len(data))]
        attention_mask = [data[i][1] for i in range(len(data))]
        if len(data[0]) == 2:
            collated = [input_ids, attention_mask]
        else:
            labels = Tensor([data[i][2] for i in range(len(data))])
            collated = [input_ids, attention_mask, labels]
        return collated

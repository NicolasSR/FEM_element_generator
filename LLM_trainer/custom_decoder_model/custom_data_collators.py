from copy import deepcopy
import random
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


@dataclass
class DataCollatorForDualTokenizerSeq2SeqCausalCollapse:
    ### Modified version of DataCollatorForSeq2Seq, original in
    # https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L544


    """
    Data collator that will dynamically pad the inputs received with the encoder's tokenizer,
    and then pad the labels with the decoder tokenizer.
    This Data Collator itroduces the dynamics of Causal Collapse when computing the decoder_input_ids
    and the label for the batch. These affect the tokens marked as var0_, var1_, etc. and the ones
    marked as coord_ (in the decoder tokenizer)

    Args:
        encoder_tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data for the encoder.
        decoder_tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data for the decoder.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        encoder_max_length (`int`, *optional*):
            Padding length for the encoder in case it is required (see above).
        decoder_max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above) for the decoder.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    encoder_tokenizer: PreTrainedTokenizerBase
    decoder_tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    encoder_max_length: Optional[int] = None
    decoder_max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __post_init__(self):
        
        interchangeable_tokens_dict = {
            'var0_':[],
            'var1_':[],
            'var2_':[],
            'var3_':[],
            'var4_':[],
            'coord_':[]
            }

        self.tokenizer_vocab = self.decoder_tokenizer.get_vocab()

        for token_label, token_id in self.tokenizer_vocab.items():
            if token_label[:-1] in interchangeable_tokens_dict.keys():
                interchangeable_tokens_dict[token_label[:-1]].append(token_id)
        
        for intearchangeable_id_list in interchangeable_tokens_dict.values():
            intearchangeable_id_list = intearchangeable_id_list.sort()

        self.interchangeable_tokens_dict = interchangeable_tokens_dict
        self.tokens_exchange_dict = {v: v for v in self.tokenizer_vocab.values()}
        self.randomized_tokens_dict = deepcopy(self.interchangeable_tokens_dict)
        
        self.inverse_interchangeable_tokens_dict = {}
        for token_id in self.tokenizer_vocab.values():
            for group_name, group_ids_list in self.interchangeable_tokens_dict.items():
                if token_id in group_ids_list:
                    self.inverse_interchangeable_tokens_dict[token_id] = group_name
                    break

        self.vocab_size = len(self.tokenizer_vocab)



    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # reconvert list[None] to None if necessary
        # this might occur when we pass {..., "labels": None}
        if labels is not None and all(label is None for label in labels):
            labels = None
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.encoder_tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.encoder_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # we have to pad the labels manually as we cannot rely on `tokenizer.pad` and we need them to be of the same length to return tensors
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        if labels is not None:
            if no_padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.decoder_max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.decoder_max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.decoder_tokenizer.padding_side
                if isinstance(features[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                label,
                            ]
                        )
                        for label in labels
                    ]

        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get("labels", None) is not None:
            if return_tensors == "pt":
                import torch

                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
            elif return_tensors == "tf":
                import tensorflow as tf

                batch["labels"] = tf.constant(batch["labels"], dtype=tf.int64)
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            decoder_input_ids = self.apply_causal_collapse(decoder_input_ids)
            batch["decoder_input_ids"] = decoder_input_ids
        elif not hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            raise Exception("""DataCollatorForDualTokenizerSeq2SeqCausalCollapse needs the language model
                            to incorporate a prepare_decoder_input_ids_from_labels() method.""")
        else:
            raise Exception("Did not find id_labels in batch data")
        
        # transform id_labels into the class probabilities that will be the actual label
        batch["labels"] = self.get_class_probabilities_with_causal_collapse(batch["labels"])

        return batch
    
    def apply_causal_collapse(self, decoder_input_ids):
        device = decoder_input_ids.device
        decoder_input_ids = decoder_input_ids.detach().cpu().tolist()
        updated_decoder_input_ids = []
        for input_ids in decoder_input_ids:
            for group, tokens_list in self.randomized_tokens_dict.items():
                random.shuffle(tokens_list)
                for original_id, shuffled_id in zip(self.interchangeable_tokens_dict[group], tokens_list):
                    self.tokens_exchange_dict[original_id] = shuffled_id
            updated_decoder_input_ids.append([self.tokens_exchange_dict[input_id] for input_id in input_ids])
        updated_decoder_input_ids = torch.tensor(updated_decoder_input_ids, device=device)

        return updated_decoder_input_ids
    
    def get_class_probabilities_with_causal_collapse(self, id_labels_list):
        device = id_labels_list.device
        id_labels_list = id_labels_list.detach().cpu().tolist()
        class_probabilities_list = []
        for id_labels in id_labels_list:
            dynamic_inverse_dict = deepcopy(self.inverse_interchangeable_tokens_dict)
            dynamic_dict = deepcopy(self.interchangeable_tokens_dict)
            class_probabilities_list.append([])
            for current_id in id_labels:
                class_probabilities_list[-1].append([0 for i in range(self.vocab_size)])
                if current_id in dynamic_inverse_dict.keys():
                    avialiable_ids = dynamic_dict[dynamic_inverse_dict[current_id]]
                    for id in avialiable_ids:
                        class_probabilities_list[-1][-1][id] = 1/len(avialiable_ids)
                    del dynamic_inverse_dict[current_id]
                    avialiable_ids.remove(current_id)
                elif current_id == -100: # Loss ignore index
                    pass
                    # "Warning: Custom data collator has a hardcoded ignore index of -100'"
                else:
                    class_probabilities_list[-1][-1][current_id] = 1.0
        
        class_probabilities_list = torch.tensor(class_probabilities_list, device=device)
        return class_probabilities_list

                

    



@dataclass
class DataCollatorForDualTokenizerSeq2Seq:
    ### Modified version of DataCollatorForSeq2Seq, original in
    # https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L544

    """
    Data collator that will dynamically pad the inputs received with the encoder's tokenizer,
    and then pad the labels with the decoder tokenizer.

    Args:
        encoder_tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data for the encoder.
        decoder_tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data for the decoder.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        encoder_max_length (`int`, *optional*):
            Padding length for the encoder in case it is required (see above).
        decoder_max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above) for the decoder.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    encoder_tokenizer: PreTrainedTokenizerBase
    decoder_tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    encoder_max_length: Optional[int] = None
    decoder_max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # reconvert list[None] to None if necessary
        # this might occur when we pass {..., "labels": None}
        if labels is not None and all(label is None for label in labels):
            labels = None
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.encoder_tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.encoder_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # we have to pad the labels manually as we cannot rely on `tokenizer.pad` and we need them to be of the same length to return tensors
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        if labels is not None:
            if no_padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.decoder_max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.decoder_max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.decoder_tokenizer.padding_side
                if isinstance(features[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                label,
                            ]
                        )
                        for label in labels
                    ]

        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get("labels", None) is not None:
            if return_tensors == "pt":
                import torch

                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
            elif return_tensors == "tf":
                import tensorflow as tf

                batch["labels"] = tf.constant(batch["labels"], dtype=tf.int64)
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids

        return batch
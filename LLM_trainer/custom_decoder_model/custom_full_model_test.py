## Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/configuration_bart.py#L31
## Then modified

import math
from typing import Optional, List, Union, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.models.bart.modeling_bart import BartModel, BartEncoder, BartDecoder, BartScaledWordEmbedding
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, BaseModelOutput, shift_tokens_right, ACT2FN
from transformers.models.bart.configuration_bart import BartConfig
from transformers.utils import logging
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput

from custom_decoder_model.custom_model_configuration_explicit_vars import CustomConfig

logger = logging.get_logger(__name__)

class CustomBartModel(BartModel):
    # _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    _tied_weights_keys = []

    def __init__(self, config_encoder: BartConfig, config_decoder: CustomConfig, dual_embedding_size: bool):
        super().__init__(config_encoder)

        padding_idx_encoder, vocab_size_encoder = config_encoder.pad_token_id, config_encoder.vocab_size
        padding_idx_decoder, vocab_size_decoder = config_decoder.pad_token_id, config_decoder.vocab_size
        embed_scale_encoder = math.sqrt(config_encoder.d_model) if config_encoder.scale_embedding else 1.0
        embed_scale_decoder = math.sqrt(config_decoder.d_model) if config_decoder.scale_embedding else 1.0
        self.word_embedding_encoder = BartScaledWordEmbedding(vocab_size_encoder, config_encoder.d_model, padding_idx_encoder, embed_scale=embed_scale_encoder)
        self.word_embedding_decoder = BartScaledWordEmbedding(vocab_size_decoder, config_decoder.d_model, padding_idx_decoder, embed_scale=embed_scale_decoder)

        self.encoder = BartEncoder(config_encoder, self.word_embedding_encoder)
        self.decoder = BartDecoder(config_decoder, self.word_embedding_decoder)
        del(self.shared)

        # Layers and activations for the embeddings adapter
        self.dual_embedding_size = dual_embedding_size
        if self.dual_embedding_size:
            self.embed_adapter_activation_fn = ACT2FN[config_encoder.activation_function]
            self.embed_adapter_fc1 = nn.Linear(config_encoder.d_model, config_encoder.encoder_ffn_dim)
            self.embed_adapter_fc2 = nn.Linear(config_encoder.encoder_ffn_dim, config_decoder.d_model)
            self.embed_adapter_activation_dropout = config_encoder.activation_dropout
            self.embed_adapter_dropout = config_encoder.dropout
            self.embed_adapter_norm_layer = nn.LayerNorm(config_decoder.d_model)

        # Initialize weights and apply final processing
        self.post_init()

    def _tie_weights(self):
        if self.config.tie_word_embeddings and hasattr(self, 'word_embedding_encoder') and hasattr(self, 'word_embedding_decoder'):
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.word_embedding_encoder)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.word_embedding_decoder)

    def get_input_embeddings(self):
        return self.word_embedding_decoder

    def set_input_embeddings(self, value):
        self.word_embedding_decoder = value
        self.encoder.embed_tokens = self.word_embedding_decoder
        # self.decoder.embed_tokens = self.shared

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        encoder_hidden_states=encoder_outputs[0]
        if self.dual_embedding_size:
            # Embedding sizes adaptation (Fully Connected layers)
            
            encoder_hidden_states = self.embed_adapter_activation_fn(self.embed_adapter_fc1(encoder_hidden_states))
            encoder_hidden_states = nn.functional.dropout(encoder_hidden_states, p=self.embed_adapter_activation_dropout, training=self.training)
            encoder_hidden_states = self.embed_adapter_fc2(encoder_hidden_states)
            encoder_hidden_states = nn.functional.dropout(encoder_hidden_states, p=self.embed_adapter_dropout, training=self.training)
                # Note that we cannot have a residual connection here. This may hinder the training process
            encoder_hidden_states = self.embed_adapter_norm_layer(encoder_hidden_states)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class CustomBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"
    _tied_weights_keys = ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "lm_head.weight"]
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]

    def __init__(self, config_encoder: BartConfig, config_decoder: CustomConfig, dual_embedding_size: bool = False):
        super().__init__(config_encoder)
        self.model = CustomBartModel(config_encoder, config_decoder, dual_embedding_size)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.word_embedding_decoder.num_embeddings)))
        self.lm_head = nn.Linear(config_decoder.d_model, self.model.word_embedding_decoder.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.model.decoder.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:

            if labels.shape[-1] == self.model.decoder.config.vocab_size: # Assume labels are class probabilities
                # In this case, we have to remove entries of lm_logits corresponding to ignore_index (-100) in decoder_input_ids
                lm_logits_cropped = lm_logits.view(-1, self.model.decoder.config.vocab_size)
                labels = labels.view(-1, self.model.decoder.config.vocab_size)
                mask = labels.abs().sum(dim=1) != 0.0
                lm_logits_cropped = lm_logits_cropped[mask]
                labels = labels[mask]

            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            if labels.shape[-1] == self.model.decoder.config.vocab_size: # Assume labels are class probabilities
                masked_lm_loss = loss_fct(lm_logits_cropped, labels)
            else:  # Assume labels are token IDs
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.model.decoder.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
import torch
import os
from cpeft import PeftConfig, PromptTuningEmbedding, AttemptSubModule

from transformers import PreTrainedModel
from transformers.utils import PushToHubMixin
from typing import Dict, Any, List, Optional
from copy import deepcopy

from .utils import _prepare_prompt_learning_config, infer_device, _get_batch_size
from .save_and_load import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    load_peft_weights,
)


class PeftModel(PushToHubMixin, torch.nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        adapter_name: str = "peft",
    ):
        super().__init__()
        self.base_model = model
        self.device = self.base_model.device
        self.active_adapter = adapter_name

        self._peft_config = {adapter_name: peft_config}
        self._is_prompt_learning = peft_config.is_prompt_learning

        self.config = getattr(self.base_model, "config", {"model_type": "custom"})
        self.add_adapter(adapter_name, peft_config)

        prefix_embeddings = []
        if self.peft_config[self.active_adapter].peft_type == "attempt":
            for path in self.peft_config[self.active_adapter].prompt_embedding_paths:
                emb = torch.load(path)
                # this is because of original attempt prompts are not dict
                if type(emb) == dict:
                    prefix_embeddings.append(emb["prompt_embeddings"].to(self.device))
                else:
                    prefix_embeddings.append(emb.to(self.device))

            self.attention_module[adapter_name].store_prefix_weights(prefix_embeddings)

    def save_pretrained(
        self,
        save_directory: str,
        selected_adapters: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        if os.path.isfile(save_directory):
            raise ValueError(f"{save_directory} is not a directory.")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())

        os.makedirs(save_directory, exist_ok=True)

        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]
            output_state_dict = get_peft_model_state_dict(
                self,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
            )
            output_dir = (
                os.path.join(save_directory, adapter_name)
                if adapter_name != "peft"
                else save_directory
            )

            os.makedirs(output_dir, exist_ok=True)

            torch.save(output_state_dict, os.path.join(output_dir, "adapter_model.bin"))

            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True

            peft_config.save_pretrained(output_dir)
            peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(
        cls,
        model: PreTrainedModel,
        model_id: str,
        adapter_name: str = "peft",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs: Any,
    ):
        if config is None:
            model_path = model_id

            if adapter_name != "peft":
                model_path = os.path.join(model_path, adapter_name)

            from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING

            config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(model_id)
            ].from_pretrained(model_path, **kwargs)

        if config.is_prompt_learning and is_trainable:
            raise ValueError(
                "Cannot set a prompt learning adapter to trainable when loading pretrained adapter."
            )
        else:
            config.inference_mode = not is_trainable

        model = PeftModelForSeq2SeqLM(model, config, adapter_name)
        model.load_adapter(
            model_path, adapter_name, is_trainable=is_trainable, **kwargs
        )

        return model

    @property
    def peft_config(self) -> Dict[str, str]:
        return self._peft_config

    @property
    def active_peft_config(self):
        return self.peft_config[self.active_adapter]

    @peft_config.setter
    def peft_config(self, value: Dict[str, str]):
        self.peft_config = value

    def print_trainable_parameters(self):
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    def get_nb_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for n, param in self.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                # print(n, param)
                trainable_params += num_params

        return trainable_params, all_param

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig):
        self.peft_config[adapter_name] = peft_config

        if peft_config.is_prompt_learning:
            if hasattr(self.config, "to_dict"):
                dict_config = self.config.to_dict()
            else:
                dict_config = self.config

            peft_config = _prepare_prompt_learning_config(peft_config, dict_config)
            self._setup_prompt_encoder(adapter_name)

            if peft_config.peft_type == "attempt":
                self._setup_attention_module(adapter_name)

    def load_adapter(
        self,
        model_id: str,
        adapter_name: str,
        is_trainable: bool = False,
        **kwargs: Any,
    ):
        torch_device = infer_device()

        adapters_weights = load_peft_weights(model_id, device=torch_device, **kwargs)

        load_result = set_peft_model_state_dict(
            self, adapters_weights, adapter_name=adapter_name
        )

        if not is_trainable:
            self.eval()

        return load_result

    def get_prompt_embedding_to_save(self, adapter_name: str):
        prompt_encoder = self.prompt_encoder[adapter_name]
        prompt_tokens = (
            self.prompt_tokens[adapter_name]
            .unsqueeze(0)
            .expand(1, -1)
            .to(prompt_encoder.embedding.weight.device)
        )
        prompt_embeddings = prompt_encoder(prompt_tokens)
        return prompt_embeddings[0].detach().cpu()

    def get_prompt(self, batch_size: int, task_ids: List[int]):
        peft_config = self.active_peft_config
        prompt_encoder = self.prompt_encoder[self.active_adapter]
        prompt_tokens = (
            self.prompt_tokens[self.active_adapter]
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(self.device)
        )

        print(prompt_tokens, self.device)

        if peft_config.inference_mode:
            prompts = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
        else:
            prompts = prompt_encoder(prompt_tokens, task_ids)

        print(prompts, prompts.shape)
        return prompts

    def get_instance_prompt(self, inputs_embeds, prompts):
        attention_module = self.attention_module[self.active_adapter]

        instance_prompts = attention_module(inputs_embeds, prompts)

        return instance_prompts

    def _setup_attention_module(self, adapter_name):
        config = self.peft_config[adapter_name]

        if not hasattr(self, "attention_module"):
            self.attention_module = torch.nn.ModuleDict({})

        if config.attn_method == "sub":
            attention_module = AttemptSubModule(config)

        attention_module = attention_module.to(self.device)
        self.attention_module.update(
            torch.nn.ModuleDict({adapter_name: attention_module})
        )

    def _setup_prompt_encoder(self, adapter_name):
        config = self.peft_config[adapter_name]

        if not hasattr(self, "prompt_encoder"):
            self.prompt_encoder = torch.nn.ModuleDict({})
            self.prompt_tokens = {}

        transformer_backbone = None

        # freeze all base model parameters
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False

            if isinstance(module, PreTrainedModel):
                # Make sure to freeze Tranformers model
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name

        if transformer_backbone is None:
            transformer_backbone = self.base_model

        # set the number of modules
        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = (
                2 if config.task_type == "seq_2_seq_lm" else 1
            )

        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(
                    named_param.replace(".weight", "")
                )
                break

        if config.peft_type == "prompt_tuning" or config.peft_type == "attempt":
            prompt_encoder = PromptTuningEmbedding(config, self.word_embeddings)

        print(self.device)
        prompt_encoder = prompt_encoder.to(self.device)
        self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).long()

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(self, *args: Any, **kwargs: Any):
        return self.get_base_model()(*args, **kwargs)


class PeftModelForSeq2SeqLM(PeftModel):
    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        adapter_name: str = "seq2seq_peft",
    ):
        super().__init__(model, peft_config, adapter_name)

        self.base_model_prepare_inputs_for_generation = (
            self.base_model.prepare_inputs_for_generation
        )
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config

        batch_size = _get_batch_size(input_ids, inputs_embeds)

        if decoder_attention_mask is not None:
            prefix_attention_mask = torch.ones(
                batch_size, peft_config.num_virtual_tokens
            ).to(decoder_attention_mask.device)
            decoder_attention_mask = torch.cat(
                (prefix_attention_mask, decoder_attention_mask), dim=1
            )

        kwargs.update(
            {
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(
                batch_size, peft_config.num_virtual_tokens
            ).to(attention_mask.device)
            kwargs["attention_mask"] = torch.cat(
                (prefix_attention_mask, attention_mask), dim=1
            )

        prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)

        if peft_config.peft_type == "attempt":
            prompts = self.get_instance_prompt(inputs_embeds, prompts)

        prompts = prompts.to(inputs_embeds.dtype)
        # print(prompts[:, : peft_config.num_virtual_tokens].shape)
        # print(inputs_embeds.shape)

        inputs_embeds = torch.cat(
            (prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1
        )

        return self.base_model(
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            **kwargs,
        )

    def generate(self, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = (
            self.prepare_inputs_for_generation
        )
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self._prepare_encoder_decoder_kwargs_for_generation
        )

        kwargs = deepcopy(kwargs)

        input_ids = kwargs.pop("input_ids")
        inputs_embeds = self.word_embeddings(input_ids)
        batch_size = inputs_embeds.shape[0]
        prompts = self.get_prompt(batch_size=batch_size)

        if peft_config.peft_type == "attempt":
            prompts = self.get_instance_prompt(inputs_embeds, prompts)

        prompts = prompts.to(inputs_embeds.dtype)

        inputs_embeds = torch.cat(
            (prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1
        )
        kwargs["inputs_embeds"] = inputs_embeds

        if "attention_mask" in kwargs:
            prefix_attention_mask = torch.ones(
                batch_size, peft_config.num_virtual_tokens
            ).to(kwargs["attention_mask"].device)
            kwargs["attention_mask"] = torch.cat(
                (prefix_attention_mask, kwargs["attention_mask"]), dim=1
            )

        return self.base_model.generate(**kwargs)

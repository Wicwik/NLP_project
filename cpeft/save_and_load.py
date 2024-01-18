import os
import torch

from typing import Optional

from .utils import infer_device


def get_peft_model_state_dict(model, state_dict=None, adapter_name="peft"):
    # state dict not used yet, but may be used when we would like to save additional peft layers
    # fro promtp tuning we save only the prompt embedding weights
    config = model.peft_config[adapter_name]

    to_return = {}
    if config.inference_mode:
        prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
    else:
        prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
    to_return["prompt_embeddings"] = prompt_embeddings

    if config.peft_type == "attempt":
        to_return["attention_module"] = model.attention_module.state_dict()

    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    return to_return


def set_peft_model_state_dict(model, peft_state_dict, adapter_name="peft"):
    config = model.peft_config[adapter_name]
    state_dict = peft_state_dict

    peft_model_state_dict = state_dict
    # print(peft_model_state_dict)

    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    if config.is_prompt_learning:
        if model.prompt_encoder[adapter_name].embedding == torch.nn.ModuleList:
            print(peft_model_state_dict["prompt_embeddings"])
            exit(0)
        else:
            model.prompt_encoder[adapter_name].embedding.load_state_dict(
                {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
            )

        if config.peft_type == "attempt":
            model.attention_module.load_state_dict(
                peft_model_state_dict["attention_module"]
            )

    return load_result


def load_peft_weights(model_id: str, device: Optional[str] = None):
    path = model_id

    if device is None:
        device = infer_device()

    filename = os.path.join(path, "adapter_model.bin")
    adapters_weights = torch.load(filename, map_location=torch.device(device))

    return adapters_weights

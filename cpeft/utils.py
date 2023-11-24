from .config import PeftConfig

def _prepare_prompt_learning_config(peft_config: PeftConfig, model_config):
    peft_config.num_layers = model_config["num_layers"]
    peft_config.token_dim = model_config["d_model"]
    peft_config.num_attention_heads = model_config["num_heads"]

    if getattr(peft_config, "encoder_hidden_size", None) is None:
        setattr(peft_config, "encoder_hidden_size", peft_config.token_dim)

    return peft_config
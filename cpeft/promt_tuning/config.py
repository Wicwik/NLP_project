from ..config import PeftConfig

from dataclasses import field, dataclass

@dataclass
class PromptTuningConfig(PeftConfig):
    promt_init : str = field(default="RANDOM", metadata={"help": "Prompt init type [RANDOM, TEXT], default is RANDOM"})
    num_virtual_tokens: int = field(default=100, metadata={"help": "Soft prompt lenght"})

    def __post_init__(self):
        self.peft_type = "prompt_tuning"
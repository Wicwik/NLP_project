from dataclasses import field, dataclass, asdict
from typing import Optional

from typing import List

from ..prompt_tuning.config import PromptTuningConfig


@dataclass
class AttemptConfig(PromptTuningConfig):
    prompt_init_embedding: str = field(
        default=None,
        metadata={"help": "Embedding to init the promt with."},
    )
    prompt_embedding_paths: List[str] = field(
        default=None,
        metadata={"help": "List of paths to source prompts, default is empty list."},
    )
    attn_method: str = field(
        default="sub",
        metadata={
            "help": "Method of attention calculation, default is sub (subnetwork)."
        },
    )
    prefix_num: int = field(default=0, metadata={"help": "Number of source prompts."})
    temperature: int = field(
        default=2087,
        metadata={"help": "Initial temperature to calculate attention with."},
    )
    shared_attn: bool = field(
        default=False, metadata={"help": "multi-task attention sharing"}
    )

    def __post_init__(self):
        self.peft_type = "attempt"

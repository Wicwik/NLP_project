# TODO fix autotask so that AutoTask.get() does not require config as parameter

import functools

import os
import torch

# from peft import TaskType
from cpeft import PromptTuningConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from torch.utils.data import DataLoader
from datetime import datetime

from tasks import AutoTask, TaskDataCollatorForSeq2Seq
from trainer import Trainer


class PeftTraining:
    configs = None
    use_wandb = None
    metric_fs = None

    def __init__(self, configs, use_wandb=True):
        self.configs = configs
        self.use_wandb = use_wandb

    def preprocess_function(
        self, examples, config, tokenizer, max_target_length, task_id=None
    ):
        inputs = tokenizer(
            examples["source"],
            max_length=config["max_source_length"],
            padding=False,
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                max_length=max_target_length,
                padding=False,
                truncation=True,
            )

        inputs["labels"] = labels["input_ids"]
        inputs["extra_fields"] = examples["extra_fields"]

        if task_id is not None:
            inputs["task_ids"] = [task_id for _ in examples["extra_fields"]]

        return inputs

    def get_data(self, config, tokenizer, add_prefix=True):
        cols_to_remove = ["source", "target"]

        max_target_length = AutoTask.get(
            config["datasets"][0], config
        ).get_max_target_length(
            tokenizer, default_max_length=config["max_target_length"]
        )

        config["max_target_length"] = max_target_length

        train_dataset = AutoTask.get(config["datasets"][0], config).get(
            split="train",
            split_validation_test=config["split_validation_test"],
            add_prefix=True,
            n_obs=config["max_train_samples"]
            if "max_train_samples" in config
            else None,
        )
        train_dataset = train_dataset.map(
            functools.partial(
                self.preprocess_function,
                config=config,
                tokenizer=tokenizer,
                max_target_length=max_target_length,
            ),
            batched=True,
            load_from_cache_file=False,
            desc="Running preprocess_function on train_dataset",
        )

        train_dataset = train_dataset.remove_columns(cols_to_remove)

        valid_dataset = AutoTask.get(config["datasets"][0], config).get(
            split="validation",
            split_validation_test=config["split_validation_test"],
            add_prefix=True,
            n_obs=config["max_valid_samples"]
            if "max_valid_samples" in config
            else None,
        )
        valid_dataset = valid_dataset.map(
            functools.partial(
                self.preprocess_function,
                config=config,
                tokenizer=tokenizer,
                max_target_length=max_target_length,
            ),
            batched=True,
            load_from_cache_file=False,
            desc="Running preprocess_function on valid_dataset",
        )

        valid_dataset = valid_dataset.remove_columns(cols_to_remove)

        test_dataset = AutoTask.get(config["datasets"][0], config).get(
            split="test",
            split_validation_test=config["split_validation_test"],
            add_prefix=True,
            n_obs=config["max_test_samples"] if "max_test_samples" in config else None,
        )
        test_dataset = test_dataset.map(
            functools.partial(
                self.preprocess_function,
                config=config,
                tokenizer=tokenizer,
                max_target_length=max_target_length,
            ),
            batched=True,
            load_from_cache_file=False,
            desc="Running preprocess_function on test_dataset",
        )

        test_dataset = test_dataset.remove_columns(cols_to_remove)

        data_collator = TaskDataCollatorForSeq2Seq(tokenizer)

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=config["batch_size"],
            pin_memory=True,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            collate_fn=data_collator,
            batch_size=config["batch_size"],
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=data_collator,
            batch_size=config["batch_size"],
            pin_memory=True,
        )

        # print(train_dataset)

        return train_dataloader, valid_dataloader, test_dataloader

    # create torchmetric metrics with their names
    def create_metric_fs(self, config):
        return {
            n: m()
            for n, m in zip(
                AutoTask.get(config["datasets"][0], config).metric_names,
                AutoTask.get(config["datasets"][0], config).metrics,
            )
        }

    def compute_metrics(self, eval_preds, tokenizer, config, prefix):
        preds, labels, data_info = eval_preds
        postprocessor = AutoTask.get(config["datasets"][0], config).postprocessor
        decoded_preds, decoded_labels = postprocessor(
            preds,
            labels,
            tokenizer,
            ignore_pad_token_for_loss=True,
            data_info=data_info,
        )

        # print(decoded_preds, decoded_labels)

        metrics = {
            n: m(decoded_preds, decoded_labels) for n, m in self.metric_fs.items()
        }

        result_m = {}
        for n, m in metrics.items():
            if "squad" in n.lower():
                result_m[f"{prefix}_em"] = m["em"]
                result_m[f"{prefix}_f1"] = m["f1"]
            else:
                result_m[n] = m

        return result_m

    def run(self):
        for config in self.configs:
            config["output_dir"] = os.path.join(
                os.path.dirname(__file__), config["output_dir"]
            )

            from cpeft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[config["peft_type"]](
                task_type=config["task_type"],
                num_virtual_tokens=config["num_virtual_tokens"],
                prompt_init=config["prompt_init"],
            )

            if config["peft_type"] == "attempt":
                peft_config.prompt_init_embedding = config["prompt_init_embedding"]
                peft_config.prompt_embedding_paths = config["prompt_embedding_paths"]
                peft_config.attn_method = config["attn_method"]
                peft_config.prefix_num = config["prefix_num"]
                peft_config.temperature = config["temperature"]

            # peft_config = PromptTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=config["num_virtual_tokens"])

            for nr in range(config["n_runs"]):
                config["run"] = nr + 1

                print(f"Started {config['run']}. run...")

                model = AutoModelForSeq2SeqLM.from_pretrained(
                    config["model_name_or_path"]
                )
                model = get_peft_model(model, peft_config)

                # pretrained_attempt = torch.load(os.path.join(config["output_dir"], "attempt_original/MNLI/adapter_model.bin"))
                # print(pretrained_attempt, pretrained_attempt.size())
                # print(model.prompt_encoder.peft.embedding.weight)

                # model.prompt_encoder.peft.embedding.weight = pretrained_attempt
                # print(model.prompt_encoder.peft.embedding.weight)

                model.print_trainable_parameters()
                model.to(config["device"])
                config["timestamp"] = datetime.now().strftime("%m%d%Y%H%M%S")

                self.metric_fs = self.create_metric_fs(config)

                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=config["learning_rate"]
                )

                tokenizer = AutoTokenizer.from_pretrained(
                    config["tokenizer_name_or_path"],
                    model_max_length=512,
                    use_fast=True,
                )
                model.resize_token_embeddings(len(tokenizer))

                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                trainer = Trainer(
                    model=model,
                    config=config,
                    dataloaders=self.get_data(config, tokenizer),
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    metric_fs=self.metric_fs,
                    compute_metrics=self.compute_metrics,
                )

                trainer.run()

                print(f"Finished {config['run']}. run...")

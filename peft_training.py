# TODO fix autotask so that AutoTask.get() does not require config as parameter

import functools

import os
import torch

# from peft import TaskType
from cpeft import get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from datasets import concatenate_datasets
from torch.utils.data import DataLoader
from datetime import datetime

from tasks import AutoTask, TaskDataCollatorForSeq2Seq, ExtraDefaultDataCollator
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
        padding = "max_length" if config["pad_to_max_length"] else False

        inputs = tokenizer(
            examples["source"],
            max_length=config["max_source_length"],
            padding=padding,
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                max_length=max_target_length,
                padding=padding,
                truncation=True,
            )

        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        inputs["labels"] = labels["input_ids"]
        inputs["extra_fields"] = examples["extra_fields"]

        if task_id is not None:
            inputs["task_ids"] = [task_id for _ in examples["extra_fields"]]

        return inputs

    def get_data(self, config, tokenizer, add_prefix=True):
        cols_to_remove = ["source", "target"]

        max_target_lengths = [
            AutoTask.get(dataset_name, config).get_max_target_length(
                tokenizer, default_max_length=config["max_target_length"]
            )
            for dataset_name in config["datasets"]
        ]

        config["max_target_length"] = max(max_target_lengths)
        print(
            "Max target length: ",
            config["max_target_length"],
            "Chosen from: ",
            max_target_lengths,
        )

        train_datasets = [
            AutoTask.get(dataset_name, config).get(
                split="train",
                split_validation_test=config["split_validation_test"],
                add_prefix=True,
                n_obs=config["max_train_samples"]
                if "max_train_samples" in config
                else None,
            )
            for dataset_name in config["datasets"]
        ]

        for i, train_dataset in enumerate(train_datasets):
            if config["shared_attn"] is True:
                train_datasets[i] = train_datasets[i].map(
                    functools.partial(
                        self.preprocess_function,
                        config=config,
                        tokenizer=tokenizer,
                        max_target_length=max_target_lengths[i],
                        task_id=i,
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running preprocess_function on train_dataset",
                )
            else:
                train_datasets[i] = train_datasets[i].map(
                    functools.partial(
                        self.preprocess_function,
                        config=config,
                        tokenizer=tokenizer,
                        max_target_length=max_target_lengths[i],
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running preprocess_function on train_dataset",
                )

        train_dataset = concatenate_datasets(train_datasets)
        train_dataset = train_dataset.remove_columns(cols_to_remove)

        valid_datasets = {
            dataset_name: AutoTask.get(dataset_name, config).get(
                split="validation",
                split_validation_test=config["split_validation_test"],
                add_prefix=True,
                n_obs=config["max_valid_samples"]
                if "max_valid_samples" in config
                else None,
            )
            for dataset_name in config["datasets"]
        }

        for i, name in enumerate(valid_datasets):
            if config["shared_attn"] is True:
                valid_datasets[name] = valid_datasets[name].map(
                    functools.partial(
                        self.preprocess_function,
                        config=config,
                        tokenizer=tokenizer,
                        max_target_length=max_target_lengths[i],
                        task_id=i,
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running preprocess_function on valid_dataset",
                )
            else:
                valid_datasets[name] = valid_datasets[name].map(
                    functools.partial(
                        self.preprocess_function,
                        config=config,
                        tokenizer=tokenizer,
                        max_target_length=max_target_lengths[i],
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running preprocess_function on valid_dataset",
                )

            valid_datasets[name] = valid_datasets[name].remove_columns(cols_to_remove)

        test_datasets = {
            dataset_name: AutoTask.get(dataset_name, config).get(
                split="test",
                split_validation_test=config["split_validation_test"],
                add_prefix=True,
                n_obs=config["max_test_samples"]
                if "max_test_samples" in config
                else None,
            )
            for dataset_name in config["datasets"]
        }

        for i, name in enumerate(test_datasets):
            if config["shared_attn"] is True:
                test_datasets[name] = test_datasets[name].map(
                    functools.partial(
                        self.preprocess_function,
                        config=config,
                        tokenizer=tokenizer,
                        max_target_length=max_target_lengths[i],
                        task_id=i,
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running preprocess_function on test_dataset",
                )
            else:
                test_datasets[name] = test_datasets[name].map(
                    functools.partial(
                        self.preprocess_function,
                        config=config,
                        tokenizer=tokenizer,
                        max_target_length=max_target_lengths[i],
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running preprocess_function on test_dataset",
                )

            test_datasets[name] = test_datasets[name].remove_columns(cols_to_remove)

        if config["pad_to_max_length"]:
            data_collator = ExtraDefaultDataCollator(return_tensors="pt")
        else:
            data_collator = TaskDataCollatorForSeq2Seq(tokenizer, return_tensors="pt")

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=config["batch_size"],
            pin_memory=True,
        )

        valid_dataloaders = {
            name: DataLoader(
                valid_datasets[name],
                collate_fn=data_collator,
                batch_size=config["batch_size"],
                pin_memory=True,
            )
            for name in valid_datasets
        }

        test_dataloaders = {
            name: DataLoader(
                test_datasets[name],
                collate_fn=data_collator,
                batch_size=config["batch_size"],
                pin_memory=True,
            )
            for name in test_datasets
        }

        print(train_dataset)

        return train_dataloader, valid_dataloaders, test_dataloaders

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
                result_m[f"{prefix}_{n}"] = m

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

                if "shared_attn" in config:
                    peft_config.shared_attn = config["shared_attn"]
                    peft_config.n_targets = len(config["datasets"])

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

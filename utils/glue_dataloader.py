"""Utilities for loading GLUE datasets with the Hugging Face Datasets API.

This module provides a `load_glue` function that tokenizes the data and returns
datasets ready for PyTorch DataLoader.  It is a light wrapper around
`datasets.load_dataset` and works with both single‑sentence and
sentence‑pair GLUE tasks.  See `GLUE_TASK_TO_KEYS` for supported tasks.

Example:

    from transformers import AutoTokenizer
    from utils.glue_dataloader import load_glue

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    processed = load_glue("sst2", tokenizer, max_length=128)
    train_loader = DataLoader(processed.train_dataset, batch_size=8, collate_fn=processed.data_collator)

"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

# Mapping from GLUE task names to the column names containing the text inputs.
GLUE_TASK_TO_KEYS: Dict[str, Tuple[str, Optional[str]]] = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "cola": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}


@dataclass
class GlueProcessed:
    """Container for processed GLUE datasets."""
    train_dataset: object
    eval_dataset: object
    tokenizer: AutoTokenizer
    data_collator: DataCollatorWithPadding


def load_glue(task_name: str, tokenizer: AutoTokenizer, max_length: int = 256) -> GlueProcessed:
    """Load and tokenize a GLUE task.

    Args:
        task_name: Name of the GLUE task (e.g. "sst2", "mnli").
        tokenizer: Hugging Face tokenizer to use.
        max_length: Maximum sequence length.

    Returns:
        A `GlueProcessed` object containing tokenized training and eval datasets,
        the tokenizer and a data collator for dynamic padding.
    """
    if task_name not in GLUE_TASK_TO_KEYS:
        raise ValueError(f"Unsupported GLUE task: {task_name}")

    # Load the raw GLUE dataset
    raw = load_dataset("glue", task_name)
    if task_name == "mnli":
        eval_dataset = raw["validation_matched"]
    else:
        eval_dataset = raw["validation"]

    sent1_key, sent2_key = GLUE_TASK_TO_KEYS[task_name]

    # Tokenization function
    def preprocess(examples):
        if sent2_key is None:
            texts = examples[sent1_key]
            return tokenizer(texts, truncation=True, max_length=max_length)
        else:
            texts1 = examples[sent1_key]
            texts2 = examples[sent2_key]
            return tokenizer(texts1, texts2, truncation=True, max_length=max_length)

    # Tokenize training and validation splits
    processed_train = raw["train"].map(
        preprocess,
        batched=True,
        remove_columns=raw["train"].column_names,
    )
    processed_eval = eval_dataset.map(
        preprocess,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    return GlueProcessed(
        train_dataset=processed_train,
        eval_dataset=processed_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

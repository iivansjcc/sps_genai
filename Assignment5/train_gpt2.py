# Assignment5/train_gpt2.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "openai-community/gpt2"   # small GPT-2
OUTPUT_DIR = "outputs/gpt2-finetuned-squad"
MAX_LENGTH = 256


@dataclass
class QADataset(Dataset):
    input_ids: List[torch.Tensor]
    attention_mask: List[torch.Tensor]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],
        }


def build_qa_text(question: str, context: str, answer: str) -> str:
    """
    Build the training text in the style we want the model to answer:

    "That is a great question. Question: ... Context: ... Answer: ... 
     Let me know if you have any other questions."
    """
    return (
        "That is a great question.\n"
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Answer: {answer} "
        "Let me know if you have any other questions."
    )


def prepare_dataset(tokenizer, split: str = "train", max_samples: int | None = 2000) -> QADataset:
    squad = load_dataset("squad")[split]

    if max_samples is not None:
        squad = squad.select(range(min(max_samples, len(squad))))

    input_ids_list = []
    attention_mask_list = []

    for ex in squad:
        question = ex["question"]
        context = ex["context"]
        # take the first ground-truth answer
        answer = ex["answers"]["text"][0] if ex["answers"]["text"] else ""

        text = build_qa_text(question, context, answer)

        enc = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids_list.append(enc["input_ids"].squeeze(0))
        attention_mask_list.append(enc["attention_mask"].squeeze(0))

    return QADataset(input_ids_list, attention_mask_list)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT-2 doesn’t have pad_token by default – reuse eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    print("Preparing datasets...")
    train_ds = prepare_dataset(tokenizer, split="train", max_samples=4000)
    val_ds = prepare_dataset(tokenizer, split="validation", max_samples=500)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=2,            # small number so it runs
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model and tokenizer...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Done. Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
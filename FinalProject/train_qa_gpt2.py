# train_qa_gpt2.py
#
# Fine-tune distilgpt2 as a simple causal LM on SQuAD-style QA text.
# Each training example is:
# "Context: ...\nQuestion: ...\nAnswer: ..."

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "outputs/qa-gpt2"


def build_example_text(example) -> str:
    context = example["context"].strip()
    question = example["question"].strip()
    # Take the first gold answer
    answer = example["answers"]["text"][0].strip()
    text = (
        "Context: " + context + "\n"
        "Question: " + question + "\n"
        "Answer: " + answer
    )
    return text


def main():
    # 1. Load dataset
    squad = load_dataset("squad")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Ensure pad token is set (GPT-2 normally only has EOS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    max_length = 256  # keep sequences reasonably short

    def preprocess(example):
        text = build_example_text(example)

        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # For causal LM we can simply use input_ids as labels
        enc["labels"] = enc["input_ids"].copy()
        return enc

    # Use a subset so training finishes in a reasonable time
    train_subset = squad["train"].shuffle(seed=42).select(range(5000))

    tokenized_train = train_subset.map(
        preprocess,
        remove_columns=train_subset.column_names,
        batched=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=5e-5,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
    )

    trainer.train()

    # Save final model + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
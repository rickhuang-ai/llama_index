# --- Transfer Learning on SQuAD v2.0 Parquet Data ---

import pandas as pd
# from datasets import Dataset, load_metric
from datasets import Dataset, load, load_dataset, load_from_disk, load_dataset_builder
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
import torch
import os
import json


# 1. Read Parquet Data
train_df = pd.read_parquet('./data/squad_v2_parquet/train.parquet')
val_df = pd.read_parquet('./data/squad_v2_parquet/validation.parquet')


# 2. Format Data to SQuAD v2.0 style
def format_squad(df):
    print("\n[DEBUG] df[\"answers\"].iloc[0] = ...\n", df["answers"].iloc[0])
    # Each entry is a dict with numpy arrays for 'text' and 'answer_start'
    def fix_answer(ans):
        # Convert numpy arrays to lists
        return {
            "text": list(ans["text"]),
            "answer_start": list(ans["answer_start"])
        }
    return {
        "id": df["id"],
        "title": df.get("title", [""]*len(df)),
        "context": df["context"],
        "question": df["question"],
        "answers": [fix_answer(ans) for ans in df["answers"]]
    }

train_data = format_squad(train_df)
val_data = format_squad(val_df)

def filter_short_examples(dataset):
    # Remove examples with empty or too-short context/question
    def is_valid(example):
        return bool(example['context']) and bool(example['question']) and len(example['context']) > 10 and len(example['question']) > 3
    return dataset.filter(is_valid)

train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)

train_dataset = filter_short_examples(train_dataset)
val_dataset = filter_short_examples(val_dataset)

# 3. Load Pretrained Model & Tokenizer
model_name = "deepset/roberta-base-squad2"  # or your local model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 4. Tokenize Data
def preprocess_function(examples):
    # Tokenize as before
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        max_length=318,
        stride=133,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    # Compute start_positions and end_positions
    start_positions = []
    end_positions = []

    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    
    for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sample_index = sample_mapping[i]
        answer = examples["answers"][sample_index]
        # answer = examples["answers"][i]
        if len(answer["answer_start"]) == 0:
            # No answer case
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = tokenized_examples.sequence_ids(i)
            # Find the start and end token indices
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            # Detect if the answer is out of the span
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the answer boundaries
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)
    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="./data/models/finetuned-squad2",
    # evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./data/models/logs",
    logging_steps=10,
    report_to="none"
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# 7. Train and Evaluate
train_result = trainer.train()
eval_result = trainer.evaluate()
print("Evaluation results:", eval_result)

# 8. Save Model
trainer.save_model("./data/models/finetuned-squad2")
tokenizer.save_pretrained("./data/models/finetuned-squad2")

# 9. (Optional) Evaluate with NLTK, ROUGE, F1, etc.
from datasets import load_metric
rouge = load_metric("rouge")
# You can add more metrics as needed

# Example: Compute ROUGE on validation set predictions
# predictions = trainer.predict(tokenized_val)
# rouge_score = rouge.compute(predictions=predictions.predictions, references=val_df["answers"])
# print("ROUGE score:", rouge_score)
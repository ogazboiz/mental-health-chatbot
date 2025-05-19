import json
import os
import logging
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import torch
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        dataset = Dataset.from_list([{"text": item["text"], "label": item["label"]} for item in data])
        label_counts = Counter(item["label"] for item in data)
        logging.info(f"Dataset label distribution: {label_counts}")
        if max(label_counts.keys()) > 7:
            raise ValueError("Dataset contains labels > 7, but model expects 0–7")
        return dataset
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        raise

def train_model():
    # Load dataset
    logging.info("Loading dataset...")
    dataset = load_dataset('intent_dataset.json')
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']

    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=8,  # 8 labels (0–7)
            id2label={i: f"LABEL_{i}" for i in range(8)},
            label2id={f"LABEL_{i}": i for i in range(8)}
        )
    except Exception as e:
        logging.error(f"Failed to initialize model/tokenizer: {str(e)}")
        raise

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    try:
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    except Exception as e:
        logging.error(f"Failed to tokenize dataset: {str(e)}")
        raise

    # Set format for training
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./intent_model",
        eval_strategy="epoch",  # Updated from evaluation_strategy
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"  # Disable wandb logging
    )

    # Define compute_metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        accuracy = (predictions == torch.tensor(labels)).float().mean().item()
        return {"accuracy": accuracy}

    # Initialize trainer
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics
        )
    except Exception as e:
        logging.error(f"Failed to initialize trainer: {str(e)}")
        raise

    # Train
    logging.info("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

    # Save model
    try:
        trainer.save_model("./intent_model")
        tokenizer.save_pretrained("./intent_model")
        logging.info("Model and tokenizer saved to ./intent_model")
    except Exception as e:
        logging.error(f"Failed to save model: {str(e)}")
        raise

if __name__ == "__main__":
    if os.path.exists("./intent_model"):
        os.system("rmdir /s /q intent_model" if os.name == "nt" else "rm -rf ./intent_model")
    train_model()
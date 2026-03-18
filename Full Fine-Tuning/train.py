# Import Libraries
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DataCollatorWithPadding
import torch
import random
from transformers import TrainingArguments, Trainer
import numpy as np
import pandas as pd
import os
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import psutil
import time
from threading import Thread
import datetime
import json
import gc

torch.manual_seed(67)
np.random.seed(67)
random.seed(67)
if torch.cuda.is_available():
   torch.cuda.manual_seed_all(67)

# Global variables to track peak memory usage DURING TRAINING
peak_gpu_memory_during_training = 0
peak_ram_usage_during_training = 0
monitoring = True

gpu_usage_log = []
ram_usage_log = []

results_dir = f'./results'
os.makedirs(results_dir, exist_ok=True)

# Function to monitor peak memory usage DURING TRAINING
def monitor_memory_usage_during_training():
    global peak_gpu_memory_during_training, peak_ram_usage_during_training, monitoring

    while monitoring:
        try:
            if torch.cuda.is_available():
                current_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_usage_log.append(current_gpu_memory)
                peak_gpu_memory_during_training = max(peak_gpu_memory_during_training, current_gpu_memory)
            current_ram_usage = psutil.virtual_memory().used / (1024**3)
            ram_usage_log.append(current_ram_usage)
            peak_ram_usage_during_training = max(peak_ram_usage_during_training, current_ram_usage)
            time.sleep(0.1)
        except:
            break


# Start memory monitoring in a separate thread
memory_thread = Thread(target=monitor_memory_usage_during_training)
memory_thread.daemon = True

# Load IMDb dataset directly
dataset = load_dataset("stanfordnlp/imdb")

# Create validation split from train
train_val = dataset["train"].train_test_split(test_size=0.2, seed=67)
train_dataset = train_val["train"]
eval_dataset = train_val["test"]
test_dataset = dataset["test"]

# Wrap into DatasetDict for consistency
dataset = DatasetDict({
    "train": train_dataset,
    "validation": eval_dataset,
    "test": test_dataset
})

# Tokenization
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load the Pretrained Model WITHOUT LoRA – all parameters will be trained
model = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_labels=2)

print("Total parameters:", sum(p.numel() for p in model.parameters()))
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Fix dataset variable names
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]
test_dataset = tokenized_datasets["test"]

# Data collator
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=False,
    return_tensors="pt"
)

# Compute metrics function for evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Training arguments – note the lower learning rate for full fine-tuning
training_args = TrainingArguments(
    output_dir=f"./{results_dir}/deberta-v3-small-full",
    learning_rate=2e-5,                     
    per_device_train_batch_size=16,          
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    logging_steps=100,
    fp16=torch.cuda.is_available(),         
    dataloader_pin_memory=False,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Clear memory before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# Get baseline memory usage BEFORE training
baseline_ram = psutil.virtual_memory().used / (1024**3)
if torch.cuda.is_available():
    baseline_gpu = torch.cuda.memory_allocated() / (1024**3)

print(f"Baseline memory - RAM: {baseline_ram:.2f} GB, GPU: {baseline_gpu:.2f} GB" if torch.cuda.is_available() else f"Baseline memory - RAM: {baseline_ram:.2f} GB")

# Start memory monitoring DURING TRAINING
print("Starting memory monitoring DURING TRAINING...")
memory_thread.start()

# Start training and measure time
print("Starting training...")
start_time = time.time()
trainer.train()
training_time = time.time() - start_time

# Stop memory monitoring
monitoring = False
memory_thread.join()

print(f"Training completed in: {training_time:.2f} seconds")

# Store log
log = trainer.state.log_history
with open(f'./{results_dir}/log.json', 'w') as f:
    json.dump(log, f, indent=2)

# Save the model
trainer.save_model(f"./{results_dir}/deberta-v3-small-full")

# Evaluate on validation set
print("Evaluating on validation set...")
val_results = trainer.evaluate(eval_dataset=eval_dataset)
val_accuracy = val_results.get('eval_accuracy', 0)
print(f"Validation accuracy: {val_accuracy:.4f}")

# Test accuracy evaluation
print("\n" + "="*60)
print("TEST SET ACCURACY MEASUREMENT")
print("="*60)

# Clear memory before test evaluation
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# Make predictions on test set
test_start_time = time.time()
test_predictions = trainer.predict(test_dataset)
test_time = time.time() - test_start_time

# Calculate test accuracy
test_preds = np.argmax(test_predictions.predictions, axis=1)
test_labels = test_predictions.label_ids
test_accuracy = accuracy_score(test_labels, test_preds)

prediction_answers = {
    'predictions': test_preds.tolist(),
    'labels': test_labels.tolist()
}


with open(f'./{results_dir}/prediction_answers.json', 'w') as f:
    json.dump(prediction_answers, f, indent=2)

print(f"\nTEST RESULTS:")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Inference Time: {test_time:.2f} seconds")

print(f"\nPEAK MEMORY USAGE DURING TRAINING:")
print(f"Peak GPU Memory DURING TRAINING: {peak_gpu_memory_during_training:.2f} GB")
print(f"Peak RAM Usage DURING TRAINING: {peak_ram_usage_during_training:.2f} GB")

print(f"\nMEMORY INCREASE DUE TO TRAINING:")
if torch.cuda.is_available():
    gpu_increase = peak_gpu_memory_during_training - baseline_gpu
    print(f"GPU Memory Increase for Training: {gpu_increase:.2f} GB")
ram_increase = peak_ram_usage_during_training - baseline_ram
print(f"RAM Increase for Training: {ram_increase:.2f} GB")

print(f"\nTRAINING PERFORMANCE:")
print(f"Total Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
print(f"Training Speed: {len(train_dataset)/training_time:.2f} samples/second")

# Detailed test results
print(f"\nDETAILED TEST ANALYSIS:")
print(f"Total Test Samples: {len(test_dataset)}")
print(f"Correct Predictions: {np.sum(test_preds == test_labels)}")
print(f"Incorrect Predictions: {np.sum(test_preds != test_labels)}")

# Classification report
print(f"\nCLASSIFICATION REPORT:")
print(classification_report(test_labels, test_preds, 
                      target_names=['Negative', 'Positive']))

# Save comprehensive results
results = {
    'test_accuracy': float(test_accuracy),
    'test_accuracy_percentage': float(test_accuracy * 100),
    'validation_accuracy': float(val_accuracy),
    'training_time_seconds': float(training_time),
    'test_inference_time_seconds': float(test_time),
    'peak_gpu_memory_during_training_gb': float(peak_gpu_memory_during_training),
    'peak_ram_usage_during_training_gb': float(peak_ram_usage_during_training),
    'gpu_memory_increase_during_training_gb': float(peak_gpu_memory_during_training - baseline_gpu) if torch.cuda.is_available() else 0,
    'ram_increase_during_training_gb': float(peak_ram_usage_during_training - baseline_ram),
    'baseline_ram_gb': float(baseline_ram),
    'baseline_gpu_gb': float(baseline_gpu) if torch.cuda.is_available() else 0,
    'total_parameters': sum(p.numel() for p in model.parameters()),
    'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
}

with open(f'./{results_dir}/training_results_detailed.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to 'training_results_detailed.json'")
print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Peak GPU Memory DURING TRAINING: {peak_gpu_memory_during_training:.2f} GB")
print(f"Peak RAM Usage DURING TRAINING: {peak_ram_usage_during_training:.2f} GB")

with open(f'./{results_dir}/gpu_usage_log.json', 'w') as f:
    json.dump(gpu_usage_log, f)

with open(f'./{results_dir}/ram_usage_log.json', 'w') as f:
    json.dump(ram_usage_log, f)
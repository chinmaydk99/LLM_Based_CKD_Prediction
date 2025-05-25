#!/usr/bin/env python3
"""
Llama-based CKD Prediction Model Training

This script fine-tunes a Llama model for Chronic Kidney Disease (CKD) prediction
using LoRA (Low-Rank Adaptation) and k-fold cross-validation.

Usage:
    python llama_ckd_prediction.py --train_data /path/to/train.pkl --val_data /path/to/val.pkl --test_data /path/to/test.pkl --output_dir /path/to/output
"""

import random
import numpy as np
import torch
import os
import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, auc, precision_recall_curve
from datasets import Dataset
from transformers import (
    TrainingArguments, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def preprocess_patient_data(row):
    """
    Format patient data into a natural language description.
    
    Args:
        row: DataFrame row containing patient information
        
    Returns:
        str: Formatted patient history text
    """
    gender = "male" if row["gender"] == "M" else "female"
    age = f"{row['age']} years old"
    diagnoses = "none" if not row["diagnoses"] else ", ".join(row["diagnoses"])
    abnormal_labs = "none" if not row["abnormal_labs"] else ", ".join(row["abnormal_labs"])
    
    formatted_history = (
        f"The patient is a {age} {gender}. Diagnoses include {diagnoses}. "
        f"Abnormal lab tests include {abnormal_labs}."
    )
    return formatted_history


def tokenize_texts(texts, tokenizer, max_length=512):
    """
    Tokenize input texts for model training.
    
    Args:
        texts: List of text strings
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        
    Returns:
        dict: Tokenized inputs
    """
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )


def compute_metrics(pred):
    """
    Compute evaluation metrics for model performance.
    
    Args:
        pred: Predictions from the model
        
    Returns:
        dict: Dictionary of computed metrics
    """
    logits, labels = pred
    predictions = np.argmax(logits, axis=1)
    probabilities = logits[:, 1]
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    roc_auc = roc_auc_score(labels, probabilities)
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probabilities)
    pr_auc = auc(recall_curve, precision_curve)
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


def create_datasets(train_fold, val_fold, tokenizer):
    """
    Create Hugging Face datasets from pandas DataFrames.
    
    Args:
        train_fold: Training data DataFrame
        val_fold: Validation data DataFrame
        tokenizer: Hugging Face tokenizer
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Tokenize training data
    train_encodings = tokenize_texts(train_fold["formatted_history"].tolist(), tokenizer)
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_fold["labels"].tolist()
    })
    
    # Tokenize validation data
    val_encodings = tokenize_texts(val_fold["formatted_history"].tolist(), tokenizer)
    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_fold["labels"].tolist()
    })
    
    return train_dataset, val_dataset


def create_model_and_trainer(model_id, tokenizer, train_dataset, val_dataset, 
                           class_weights_tensor, fold, output_dir):
    """
    Create and configure the model and trainer for a specific fold.
    
    Args:
        model_id: Hugging Face model identifier
        tokenizer: Tokenizer for the model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        class_weights_tensor: Class weights for imbalanced data
        fold: Current fold number
        output_dir: Output directory for results
        
    Returns:
        Trainer: Configured trainer object
    """
    # Initialize model
    config = AutoConfig.from_pretrained(model_id, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, config=config, device_map='auto'
    )
    
    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='SEQ_CLS'
    )
    model = get_peft_model(model, lora_config)
    
    # Configure model settings
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir / f"results_fold_{fold}",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        fp16=True,
        seed=42,
        report_to="none",
        logging_steps=10,
        save_total_limit=1
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    return trainer


def train_with_kfold(data, model_id, output_dir, k_folds=3):
    """
    Train model using k-fold cross-validation.
    
    Args:
        data: Combined training and validation data
        model_id: Hugging Face model identifier
        output_dir: Output directory for results
        k_folds: Number of folds for cross-validation
        
    Returns:
        dict: Results from all folds and best model path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # K-fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    best_model_path = output_path / "best_model"
    best_roc_auc = 0
    
    logger.info(f"Starting {k_folds}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        logger.info(f"Starting fold {fold + 1}/{k_folds}...")
        
        # Split the data
        train_fold = data.iloc[train_idx]
        val_fold = data.iloc[val_idx]
        
        logger.info(f"Fold {fold + 1} - Train: {len(train_fold)}, Val: {len(val_fold)}")
        
        # Create datasets
        train_dataset, val_dataset = create_datasets(train_fold, val_fold, tokenizer)
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_fold["labels"]),
            y=train_fold["labels"].tolist()
        )
        class_weights_tensor = torch.tensor(
            class_weights, dtype=torch.float32
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Class weights: {class_weights}")
        
        # Create model and trainer
        trainer = create_model_and_trainer(
            model_id, tokenizer, train_dataset, val_dataset,
            class_weights_tensor, fold, output_path
        )
        
        # Train the model
        logger.info(f"Training fold {fold + 1}...")
        trainer.train()
        
        # Evaluate the model
        logger.info(f"Evaluating fold {fold + 1}...")
        results = trainer.evaluate(val_dataset)
        logger.info(f"Results for fold {fold + 1}: {results}")
        fold_results.append(results)
        
        # Save the best model based on ROC AUC
        current_roc_auc = results.get("eval_roc_auc", 0)
        if current_roc_auc > best_roc_auc:
            best_roc_auc = current_roc_auc
            trainer.save_model(best_model_path)
            logger.info(f"New best model saved with ROC AUC: {best_roc_auc}")
    
    # Calculate average metrics across folds
    avg_metrics = {}
    for metric in fold_results[0].keys():
        if metric.startswith('eval_'):
            metric_name = metric.replace('eval_', '')
            avg_value = np.mean([result[metric] for result in fold_results])
            std_value = np.std([result[metric] for result in fold_results])
            avg_metrics[f'avg_{metric_name}'] = avg_value
            avg_metrics[f'std_{metric_name}'] = std_value
    
    logger.info("=== CROSS-VALIDATION RESULTS ===")
    for metric, value in avg_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return {
        'fold_results': fold_results,
        'avg_metrics': avg_metrics,
        'best_model_path': str(best_model_path),
        'best_roc_auc': best_roc_auc
    }


def evaluate_on_test(test_data, model_path, model_id):
    """
    Evaluate the best model on test data.
    
    Args:
        test_data: Test dataset
        model_path: Path to the best trained model
        model_id: Original model identifier
        
    Returns:
        dict: Test evaluation results
    """
    logger.info("Evaluating on test data...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Preprocess test data
    test_data["formatted_history"] = test_data.apply(preprocess_patient_data, axis=1)
    test_data["labels"] = test_data["is_ckd"].astype(int)
    
    # Tokenize test data
    test_encodings = tokenize_texts(test_data["formatted_history"].tolist(), tokenizer)
    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': test_data["labels"].tolist()
    })
    
    # Load the best model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Create trainer for evaluation
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test results: {test_results}")
    
    return test_results


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='Train Llama model for CKD prediction')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data pickle file')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to validation data pickle file')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data pickle file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for model and results')
    parser.add_argument('--model_id', type=str, default="unsloth/Llama-3.2-1B-bnb-4bit",
                        help='Hugging Face model identifier')
    parser.add_argument('--k_folds', type=int, default=3,
                        help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load data
    logger.info("Loading data...")
    train_data = pd.read_pickle(args.train_data)
    val_data = pd.read_pickle(args.val_data)
    test_data = pd.read_pickle(args.test_data)
    
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Preprocess data
    logger.info("Preprocessing data...")
    train_data["formatted_history"] = train_data.apply(preprocess_patient_data, axis=1)
    val_data["formatted_history"] = val_data.apply(preprocess_patient_data, axis=1)
    
    # Combine training and validation data for k-fold cross-validation
    data = pd.concat([train_data, val_data], ignore_index=True)
    data["labels"] = data["is_ckd"].astype(int)
    
    logger.info(f"Combined data: {len(data)} samples")
    logger.info(f"Class distribution: {data['labels'].value_counts().to_dict()}")
    
    # Train with k-fold cross-validation
    cv_results = train_with_kfold(data, args.model_id, args.output_dir, args.k_folds)
    
    # Evaluate on test data
    test_results = evaluate_on_test(test_data, cv_results['best_model_path'], args.model_id)
    
    # Save results
    results_file = Path(args.output_dir) / "training_results.txt"
    with open(results_file, 'w') as f:
        f.write("=== CROSS-VALIDATION RESULTS ===\n")
        for metric, value in cv_results['avg_metrics'].items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write(f"\nBest ROC AUC: {cv_results['best_roc_auc']:.4f}\n")
        f.write(f"Best model path: {cv_results['best_model_path']}\n")
        f.write("\n=== TEST RESULTS ===\n")
        for metric, value in test_results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    logger.info(f"Training complete! Results saved to {results_file}")


if __name__ == "__main__":
    main() 
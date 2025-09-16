#!/usr/bin/env python3
"""
Model Comparison for Educational Story Analysis

This script compares three approaches on the same test set:
1. Fine-tuned DistilBERT (dual-sequence: situation + solution → outcome)
2. Fine-tuned RoBERTa (dual-sequence: situation + solution → outcome)
3. Most Frequent Class Baseline (always predicts most common class)

Usage:
    python distilbert_comparison.py [--epochs 3] [--batch-size 8]
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from collections import Counter

# Import required libraries
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, 
        Trainer, TrainingArguments, EarlyStoppingCallback
    )
    from datasets import Dataset, load_dataset
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Install with: pip install transformers datasets scikit-learn torch")
    exit(1)


def load_and_prepare_data():
    """Load and prepare data for dual-sequence classification"""
    print("📊 Loading MU-NLPC/Edustories-en dataset...")
    
    try:
        dataset = load_dataset("MU-NLPC/Edustories-en")
        df = pd.DataFrame(dataset['train'])
        print(f"✅ Loaded dataset with {len(df)} examples")
        
        # Clean data
        df = df.dropna(subset=['implications_annotated'])
        df = df[df['implications_annotated'].str.strip() != '']
        
        # Create situation text (description + anamnesis)
        def combine_situation(row):
            parts = []
            if pd.notna(row.get('description')):
                desc = str(row['description']).strip()
                if desc and desc.lower() not in ['nan', 'none', '']:
                    parts.append(desc)
            
            if pd.notna(row.get('anamnesis')):
                anam = str(row['anamnesis']).strip()
                if anam and anam.lower() not in ['nan', 'none', '']:
                    parts.append(anam)
            
            return " ".join(parts) if parts else "No situation described"
        
        # Create solution text
        def get_solution(row):
            solution = str(row.get('solution', '')).strip()
            if solution and solution.lower() not in ['nan', 'none', '']:
                return solution
            return "No solution described"
        
        df['situation_text'] = df.apply(combine_situation, axis=1)
        df['solution_text'] = df.apply(get_solution, axis=1)
        
        # Remove empty sequences
        df = df[df['situation_text'] != 'No situation described']
        df = df[df['solution_text'] != 'No solution described']
        
        # Binary classification mapping
        def map_to_binary(label):
            if pd.isna(label) or str(label).strip() == "":
                return 'Failure'
            
            label_str = str(label).lower().strip()
            
            if "failure" in label_str:
                return 'Failure'
            
            success_indicators = ["longterm success", "shortterm success", "partial success"]
            if any(success_type in label_str for success_type in success_indicators):
                return 'Success'
            
            return 'Failure'
        
        df['binary_label'] = df['implications_annotated'].apply(map_to_binary)
        df = df[df['binary_label'].isin(['Success', 'Failure'])]
        
        # Create label mapping
        label_mapping = {'Failure': 0, 'Success': 1}
        reverse_label_mapping = {0: 'Failure', 1: 'Success'}
        df['labels'] = df['binary_label'].map(label_mapping)
        
        print(f"📈 Final dataset: {len(df)} examples")
        label_counts = df['binary_label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
        
        # Split data
        situations = df['situation_text'].tolist()
        solutions = df['solution_text'].tolist()
        labels = df['labels'].tolist()
        
        train_sit, test_sit, train_sol, test_sol, train_lab, test_lab = train_test_split(
            situations, solutions, labels, test_size=0.25, random_state=42, stratify=labels
        )
        
        train_sit, val_sit, train_sol, val_sol, train_lab, val_lab = train_test_split(
            train_sit, train_sol, train_lab, test_size=0.2, random_state=42, stratify=train_lab
        )
        
        print(f"📈 Data splits:")
        print(f"  Training: {len(train_sit)} examples")
        print(f"  Validation: {len(val_sit)} examples")
        print(f"  Test: {len(test_sit)} examples")
        
        return (train_sit, train_sol, val_sit, val_sol, test_sit, test_sol,
                train_lab, val_lab, test_lab, label_mapping, reverse_label_mapping)
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return tuple([None] * 11)


def create_dataset(situations, solutions, labels, tokenizer):
    """Create dataset with tokenized dual sequences"""
    tokenized_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    
    batch_size = 100
    for i in range(0, len(situations), batch_size):
        batch_sit = situations[i:i+batch_size]
        batch_sol = solutions[i:i+batch_size]
        batch_lab = labels[i:i+batch_size]
        
        batch_tokenized = tokenizer(
            batch_sit, batch_sol,
            truncation=True, padding=True, max_length=512,
            return_tensors="pt"
        )
        
        tokenized_data['input_ids'].extend(batch_tokenized['input_ids'])
        tokenized_data['attention_mask'].extend(batch_tokenized['attention_mask'])
        tokenized_data['labels'].extend(batch_lab)
    
    return Dataset.from_dict(tokenized_data)


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def most_frequent_baseline(train_labels, test_labels):
    """Most frequent class baseline"""
    # Find most frequent class in training data
    most_frequent = Counter(train_labels).most_common(1)[0][0]
    
    # Predict most frequent class for all test examples
    predictions = [most_frequent] * len(test_labels)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='binary', zero_division=0
    )
    accuracy = accuracy_score(test_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'predicted_class': most_frequent
    }


def main():
    parser = argparse.ArgumentParser(description='Model comparison: DistilBERT vs RoBERTa vs baseline')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--output-dir', default='./outputs/model-comparison', help='Output directory')
    
    args = parser.parse_args()
    
    print("🚀 Educational Story Analysis Model Comparison")
    print("=" * 60)
    print("Comparing:")
    print("1. Fine-tuned DistilBERT (dual-sequence)")
    print("2. Fine-tuned RoBERTa (dual-sequence)")
    print("3. Most Frequent Class Baseline")
    print("=" * 60)
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple Silicon MPS acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA GPU acceleration")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    
    # Load data
    result = load_and_prepare_data()
    if result[0] is None:
        print("❌ Failed to load data")
        return
    
    (train_sit, train_sol, val_sit, val_sol, test_sit, test_sol,
     train_lab, val_lab, test_lab, label_mapping, reverse_label_mapping) = result
    
    # MODEL 1: FINE-TUNED DISTILBERT
    print(f"\n🤖 MODEL 1: Fine-tuning DistilBERT...")
    
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_mapping), problem_type="single_label_classification"
    )
    model.to(device)
    
    # Create datasets
    train_dataset = create_dataset(train_sit, train_sol, train_lab, tokenizer)
    val_dataset = create_dataset(val_sit, val_sol, val_lab, tokenizer)
    test_dataset = create_dataset(test_sit, test_sol, test_lab, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir + "/fine-tuned",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_ratio=0.1,
        save_total_limit=3,
        report_to=None,
        seed=42,
        dataloader_pin_memory=False,
        remove_unused_columns=False
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print("🚀 Training fine-tuned DistilBERT...")
    trainer.train()
    
    print("📊 Evaluating fine-tuned DistilBERT...")
    finetuned_results = trainer.evaluate(test_dataset)
    
    # MODEL 2: FINE-TUNED ROBERTA
    print(f"\n🤖 MODEL 2: Fine-tuning RoBERTa...")
    
    roberta_model_name = 'FacebookAI/roberta-base'
    roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
    if roberta_tokenizer.pad_token is None:
        roberta_tokenizer.pad_token = roberta_tokenizer.eos_token
    
    roberta_model = AutoModelForSequenceClassification.from_pretrained(
        roberta_model_name, num_labels=len(label_mapping), problem_type="single_label_classification"
    )
    roberta_model.to(device)
    
    # Create datasets with RoBERTa tokenizer
    roberta_train_dataset = create_dataset(train_sit, train_sol, train_lab, roberta_tokenizer)
    roberta_val_dataset = create_dataset(val_sit, val_sol, val_lab, roberta_tokenizer)
    roberta_test_dataset = create_dataset(test_sit, test_sol, test_lab, roberta_tokenizer)
    
    # Training arguments for RoBERTa
    roberta_training_args = TrainingArguments(
        output_dir=args.output_dir + "/roberta-fine-tuned",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_ratio=0.1,
        save_total_limit=3,
        report_to=None,
        seed=42,
        dataloader_pin_memory=False,
        remove_unused_columns=False
    )
    
    # Train RoBERTa
    roberta_trainer = Trainer(
        model=roberta_model,
        args=roberta_training_args,
        train_dataset=roberta_train_dataset,
        eval_dataset=roberta_val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print("🚀 Training fine-tuned RoBERTa...")
    roberta_trainer.train()
    
    print("📊 Evaluating fine-tuned RoBERTa...")
    roberta_results = roberta_trainer.evaluate(roberta_test_dataset)
    
    # MODEL 3: MOST FREQUENT CLASS BASELINE
    print(f"\n🤖 MODEL 3: Most Frequent Class Baseline...")
    baseline_results = most_frequent_baseline(train_lab, test_lab)
    most_frequent_class = reverse_label_mapping[baseline_results['predicted_class']]
    print(f"📊 Baseline always predicts: {most_frequent_class}")
    
    # COMPARISON RESULTS
    print(f"\n🏆 FINAL COMPARISON RESULTS")
    print("=" * 80)
    
    models = [
        ("Fine-tuned DistilBERT", finetuned_results),
        ("Fine-tuned RoBERTa", roberta_results), 
        ("Most Frequent Baseline", baseline_results)
    ]
    
    # Print detailed results
    for model_name, results in models:
        print(f"\n📊 {model_name}:")
        for metric in ['accuracy', 'f1', 'precision', 'recall']:
            key = f'eval_{metric}' if metric in ['accuracy', 'f1', 'precision', 'recall'] and 'eval_' in str(results) else metric
            if key in results:
                value = results[key]
                print(f"  {metric.title()}: {value:.4f} ({value*100:.2f}%)")
    
    # Best model analysis
    print(f"\n🎯 PERFORMANCE RANKING:")
    print("-" * 40)
    
    accuracy_scores = []
    for model_name, results in models:
        key = 'eval_accuracy' if 'eval_accuracy' in results else 'accuracy'
        acc = results[key]
        accuracy_scores.append((model_name, acc))
    
    # Sort by accuracy (descending)
    accuracy_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model_name, acc) in enumerate(accuracy_scores, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"{emoji} {i}. {model_name}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Improvement analysis
    distilbert_acc = finetuned_results['eval_accuracy']
    roberta_acc = roberta_results['eval_accuracy'] 
    baseline_acc = baseline_results['accuracy']
    
    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    print("-" * 40)
    print(f"DistilBERT vs Baseline: {(distilbert_acc - baseline_acc)*100:+.2f} percentage points")
    print(f"RoBERTa vs Baseline: {(roberta_acc - baseline_acc)*100:+.2f} percentage points")
    print(f"RoBERTa vs DistilBERT: {(roberta_acc - distilbert_acc)*100:+.2f} percentage points")
    
    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    comparison_results = {
        'model_comparison': 'DistilBERT vs RoBERTa vs Baseline',
        'fine_tuned_distilbert': finetuned_results,
        'fine_tuned_roberta': roberta_results,
        'most_frequent_baseline': baseline_results,
        'test_set_size': len(test_lab),
        'improvement_analysis': {
            'distilbert_vs_baseline_pp': (distilbert_acc - baseline_acc) * 100,
            'roberta_vs_baseline_pp': (roberta_acc - baseline_acc) * 100,
            'roberta_vs_distilbert_pp': (roberta_acc - distilbert_acc) * 100
        }
    }
    
    with open(Path(args.output_dir) / "comparison_results.json", 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Save both fine-tuned models
    trainer.save_model(args.output_dir + "/distilbert-fine-tuned")
    tokenizer.save_pretrained(args.output_dir + "/distilbert-fine-tuned")
    
    roberta_trainer.save_model(args.output_dir + "/roberta-fine-tuned")
    roberta_tokenizer.save_pretrained(args.output_dir + "/roberta-fine-tuned")
    
    print(f"\n💾 Results saved to: {args.output_dir}")
    print(f"🏁 Model comparison completed!")


if __name__ == "__main__":
    main()

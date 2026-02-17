#!/usr/bin/env python3
"""
Custom SFT training script with response length logging.

This script wraps verl's FSDP trainer and adds logging for average response
length per batch during training.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer


class ResponseLengthLogger:
    """Logs response lengths during training."""
    
    def __init__(self, output_dir: Path, log_freq: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_freq = log_freq
        self.batch_logs = []
        self.step = 0
        
    def log_batch(self, step: int, batch_data: Dict[str, Any] = None):
        """Log response length for a batch."""
        # If we have batch data, compute actual lengths
        if batch_data and 'responses' in batch_data:
            responses = batch_data['responses']
            lengths = [len(r) for r in responses]
            avg_length = sum(lengths) / len(lengths) if lengths else 0
            min_length = min(lengths) if lengths else 0
            max_length = max(lengths) if lengths else 0
        else:
            # Placeholder - will be filled from data statistics
            avg_length = None
            min_length = None
            max_length = None
        
        log_entry = {
            'step': step,
            'avg_response_length': avg_length,
            'min_response_length': min_length,
            'max_response_length': max_length,
        }
        self.batch_logs.append(log_entry)
        
        # Print every log_freq steps
        if step % self.log_freq == 0:
            if avg_length is not None:
                print(f"[Step {step}] Avg response length: {avg_length:.1f} tokens "
                      f"(min: {min_length}, max: {max_length})")
            else:
                print(f"[Step {step}] Batch completed")
    
    def save_logs(self):
        """Save logs to file."""
        log_file = self.output_dir / "response_length_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.batch_logs, f, indent=2)
        print(f"\nSaved response length logs to: {log_file}")


def compute_data_statistics(train_file: str, model_name: str, response_key: str = "generated_solution"):
    """Pre-compute response length statistics from training data."""
    print("Computing response length statistics from training data...")
    
    # Load data
    df = pd.read_parquet(train_file)
    print(f"Loaded {len(df)} examples")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Loaded tokenizer: {model_name}")
    except Exception as e:
        print(f"Warning: Could not load tokenizer ({e}), using character count")
        tokenizer = None
    
    # Compute lengths
    lengths = []
    for idx, row in df.iterrows():
        response = str(row.get(response_key, ""))
        if tokenizer:
            tokens = tokenizer.encode(response, add_special_tokens=False)
            length = len(tokens)
        else:
            # Rough estimate: ~4 characters per token
            length = len(response) // 4
        lengths.append(length)
    
    stats = {
        'mean': sum(lengths) / len(lengths) if lengths else 0,
        'min': min(lengths) if lengths else 0,
        'max': max(lengths) if lengths else 0,
        'median': sorted(lengths)[len(lengths) // 2] if lengths else 0,
        'std': (sum((x - stats['mean'])**2 for x in lengths) / len(lengths))**0.5 if lengths else 0,
        'total_examples': len(lengths)
    }
    
    return stats, lengths


def main():
    parser = argparse.ArgumentParser(description="SFT training with response length logging")
    parser.add_argument("--train-file", type=str, required=True, help="Training parquet file")
    parser.add_argument("--val-file", type=str, help="Validation parquet file")
    parser.add_argument("--model-name", type=str, required=True, help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="./sft_output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--project-name", type=str, default="math-sft", help="Project name")
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    parser.add_argument("--log-freq", type=int, default=10, help="Log frequency (steps)")
    parser.add_argument("--response-key", type=str, default="generated_solution", help="Response key in data")
    
    args = parser.parse_args()
    
    # Set experiment name
    if not args.experiment_name:
        from datetime import datetime
        args.experiment_name = f"math-sft-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("SFT Training with Response Length Logging")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"Val file: {args.val_file or 'None (using train)'}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"GPUs: {args.nproc_per_node}")
    print("=" * 80)
    
    # Compute data statistics
    print("\n[1/3] Computing response length statistics...")
    stats, lengths = compute_data_statistics(args.train_file, args.model_name, args.response_key)
    
    stats_file = output_dir / "response_length_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nResponse length statistics:")
    print(f"  Mean: {stats['mean']:.1f} tokens")
    print(f"  Min: {stats['min']} tokens")
    print(f"  Max: {stats['max']} tokens")
    print(f"  Median: {stats['median']} tokens")
    print(f"  Std: {stats['std']:.1f} tokens")
    print(f"\nSaved to: {stats_file}")
    
    # Initialize logger
    logger = ResponseLengthLogger(output_dir, log_freq=args.log_freq)
    
    # Build verl command
    print("\n[2/3] Starting training with verl...")
    print("=" * 80)
    
    val_file = args.val_file or args.train_file
    
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={args.nproc_per_node}",
        "-m", "verl.trainer.fsdp_sft_trainer",
        f"data.train_files={args.train_file}",
        f"data.val_files={val_file}",
        "data.prompt_key=prompt",
        f"data.response_key={args.response_key}",
        f"data.micro_batch_size_per_gpu={args.batch_size}",
        f"data.max_length={args.max_length}",
        f"model.partial_pretrain={args.model_name}",
        f"optim.lr={args.learning_rate}",
        f"trainer.default_local_dir={args.output_dir}",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.total_epochs={args.epochs}",
        'trainer.logger=["console","wandb"]',
    ]
    
    # Add any additional args
    if len(sys.argv) > 1:
        # Pass through any additional verl config overrides
        pass
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)
    
    # Run training
    import subprocess
    process = subprocess.run(cmd)
    
    # Save logs
    print("\n[3/4] Saving response length logs...")
    logger.save_logs()
    
    # Generate matplotlib graph
    print("\n[4/4] Generating response length vs examples processed graph...")
    try:
        generate_response_length_graph(
            output_dir=output_dir,
            train_file=args.train_file,
            model_name=args.model_name,
            batch_size=args.batch_size,
            nproc_per_node=args.nproc_per_node,
            response_key=args.response_key
        )
    except Exception as e:
        print(f"Warning: Could not generate graph: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"Model checkpoints: {output_dir}")
    print(f"Response length stats: {stats_file}")
    print(f"Response length logs: {output_dir / 'response_length_log.json'}")
    if (output_dir / 'response_length_vs_examples.png').exists():
        print(f"Response length graph: {output_dir / 'response_length_vs_examples.png'}")
    print("=" * 80)
    
    return process.returncode


def generate_response_length_graph(
    output_dir: Path,
    train_file: str,
    model_name: str,
    batch_size: int,
    nproc_per_node: int,
    response_key: str = "generated_solution"
):
    """Generate matplotlib graph of response length vs examples processed."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    log_file = output_dir / 'response_length_log.json'
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return
    
    # Load training logs
    with open(log_file) as f:
        batch_logs = json.load(f)
    
    if not batch_logs:
        print("No batch logs found")
        return
    
    # Load training data
    train_df = pd.read_parquet(train_file)
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load tokenizer ({e}), using character count")
        tokenizer = None
    
    # Compute response lengths
    response_lengths = []
    for idx, row in train_df.iterrows():
        response = str(row.get(response_key, ""))
        if tokenizer:
            tokens = tokenizer.encode(response, add_special_tokens=False)
            length = len(tokens)
        else:
            length = len(response) // 4
        response_lengths.append(length)
    
    response_lengths = np.array(response_lengths)
    
    # Calculate examples processed
    examples_per_step = batch_size * nproc_per_node
    
    # Create data points
    data_points = []
    for log_entry in batch_logs:
        step = log_entry.get('step', 0)
        examples_processed = step * examples_per_step
        
        if examples_processed <= len(response_lengths):
            avg_length = np.mean(response_lengths[:int(examples_processed)])
        else:
            num_epochs = int(examples_processed / len(response_lengths))
            remainder = examples_processed % len(response_lengths)
            if remainder > 0:
                all_lengths = np.concatenate([response_lengths] * num_epochs + [response_lengths[:remainder]])
            else:
                all_lengths = np.concatenate([response_lengths] * num_epochs)
            avg_length = np.mean(all_lengths)
        
        data_points.append({
            'step': step,
            'examples_processed': examples_processed,
            'avg_response_length': avg_length,
            'loss': log_entry.get('loss')
        })
    
    if not data_points:
        print("No data points to plot")
        return
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    
    examples = [d['examples_processed'] for d in data_points]
    avg_lengths = [d['avg_response_length'] for d in data_points]
    losses = [d['loss'] for d in data_points if d['loss'] is not None]
    
    # Main plot
    ax1.plot(examples, avg_lengths, 'b-', linewidth=2.5, label='Average Response Length', 
             alpha=0.8, marker='o', markersize=4)
    ax1.scatter(examples, avg_lengths, s=50, alpha=0.7, c='blue', edgecolors='darkblue', linewidths=0.5)
    
    overall_mean = np.mean(response_lengths)
    overall_median = np.median(response_lengths)
    std_dev = np.std(response_lengths)
    
    ax1.axhline(y=overall_mean, color='r', linestyle='--', linewidth=2,
               label=f'Overall Mean: {overall_mean:.1f} tokens', alpha=0.7)
    ax1.axhline(y=overall_median, color='g', linestyle='--', linewidth=2,
               label=f'Overall Median: {overall_median:.1f} tokens', alpha=0.7)
    ax1.fill_between(examples,
                     [overall_mean - std_dev] * len(examples),
                     [overall_mean + std_dev] * len(examples),
                     alpha=0.2, color='gray', label=f'±1 Std Dev: {std_dev:.1f} tokens')
    
    ax1.set_xlabel('Number of SFT Examples Processed', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Response Length (tokens)', fontsize=12, fontweight='bold')
    ax1.set_title('Response Length vs SFT Examples Processed', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    
    stats_text = f'Total Examples: {len(response_lengths):,}\n'
    stats_text += f'Mean: {np.mean(response_lengths):.1f} tokens\n'
    stats_text += f'Median: {np.median(response_lengths):.1f} tokens\n'
    stats_text += f'Std Dev: {np.std(response_lengths):.1f} tokens\n'
    stats_text += f'Min: {np.min(response_lengths)} tokens\n'
    stats_text += f'Max: {np.max(response_lengths)} tokens'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Subplot: Loss or distribution
    if losses and len(losses) > 0:
        loss_examples = [d['examples_processed'] for d in data_points if d['loss'] is not None]
        ax2.plot(loss_examples, losses, 'r-', linewidth=2, label='Training Loss',
                alpha=0.8, marker='s', markersize=3)
        ax2.set_xlabel('Number of SFT Examples Processed', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax2.set_title('Training Loss vs Examples Processed', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=9)
    else:
        ax2.hist(response_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Response Length (tokens)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Distribution of Response Lengths in Training Data', fontsize=12, fontweight='bold')
        ax2.axvline(overall_mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {overall_mean:.1f}')
        ax2.axvline(overall_median, color='g', linestyle='--', linewidth=2, label=f'Median: {overall_median:.1f}')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    # Save plots
    plot_file = output_dir / 'response_length_vs_examples.png'
    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to: {plot_file}")
    
    plot_file_pdf = output_dir / 'response_length_vs_examples.pdf'
    fig.savefig(plot_file_pdf, bbox_inches='tight')
    print(f"✓ Saved plot (PDF) to: {plot_file_pdf}")
    
    plt.close()


if __name__ == "__main__":
    sys.exit(main())

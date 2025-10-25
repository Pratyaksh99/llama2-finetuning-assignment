"""
Plot training and validation loss curves
CS5242 Assignment 7
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_training_curves(loss_history_file: str = "./llama2-dolly-lora-full/loss_history.json",
                         output_file: str = "training_curves.png"):
    """Plot training and validation loss curves from saved history."""
    
    print("Loading loss history...")
    with open(loss_history_file, 'r') as f:
        history = json.load(f)
    
    # Extract losses
    train_losses = []
    train_steps = []
    eval_losses = []
    eval_steps = []
    
    for entry in history:
        if 'loss' in entry:
            train_losses.append(entry['loss'])
            train_steps.append(entry.get('step', len(train_losses)))
        if 'eval_loss' in entry:
            eval_losses.append(entry['eval_loss'])
            eval_steps.append(entry.get('step', len(eval_losses)))
    
    print(f"Found {len(train_losses)} training loss points")
    print(f"Found {len(eval_losses)} validation loss points")
    
    # Create plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Plot training loss
    plt.plot(train_steps, train_losses, label='Training Loss', color='blue', linewidth=2, alpha=0.7)
    
    # Plot validation loss
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label='Validation Loss', color='red', linewidth=2, marker='o', markersize=6)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('LLaMA-2-7B Fine-Tuning: Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    if train_losses:
        final_train_loss = train_losses[-1]
        plt.annotate(f'Final Train Loss: {final_train_loss:.4f}',
                    xy=(train_steps[-1], final_train_loss),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, color='blue',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    if eval_losses:
        final_eval_loss = eval_losses[-1]
        plt.annotate(f'Final Val Loss: {final_eval_loss:.4f}',
                    xy=(eval_steps[-1], final_eval_loss),
                    xytext=(10, -20), textcoords='offset points',
                    fontsize=10, color='red',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {output_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Training Statistics:")
    print("="*60)
    if train_losses:
        print(f"Initial training loss: {train_losses[0]:.4f}")
        print(f"Final training loss: {train_losses[-1]:.4f}")
        print(f"Loss reduction: {train_losses[0] - train_losses[-1]:.4f} ({(1 - train_losses[-1]/train_losses[0])*100:.1f}%)")
    
    if eval_losses:
        print(f"\nInitial validation loss: {eval_losses[0]:.4f}")
        print(f"Final validation loss: {eval_losses[-1]:.4f}")
        print(f"Loss reduction: {eval_losses[0] - eval_losses[-1]:.4f} ({(1 - eval_losses[-1]/eval_losses[0])*100:.1f}%)")
    print("="*60)
    
    plt.show()


def plot_learning_rate_schedule(loss_history_file: str = "./llama2-dolly-lora-full/loss_history.json",
                                output_file: str = "learning_rate_schedule.png"):
    """Plot learning rate schedule."""
    
    print("\nPlotting learning rate schedule...")
    with open(loss_history_file, 'r') as f:
        history = json.load(f)
    
    lr_values = []
    steps = []
    
    for entry in history:
        if 'learning_rate' in entry:
            lr_values.append(entry['learning_rate'])
            steps.append(entry.get('step', len(lr_values)))
    
    if not lr_values:
        print("No learning rate data found")
        return
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, lr_values, color='green', linewidth=2)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Learning rate plot saved to {output_file}")
    plt.show()


def main():
    """Generate all training plots."""
    
    print("="*60)
    print("Generating Training Visualizations")
    print("="*60)
    print()
    
    # Plot training curves
    plot_training_curves()
    
    # Plot learning rate
    plot_learning_rate_schedule()
    
    print("\n" + "="*60)
    print("✅ All plots generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - training_curves.png")
    print("  - learning_rate_schedule.png")
    print("\nUse these plots in your assignment report.")
    print("="*60)


if __name__ == "__main__":
    main()

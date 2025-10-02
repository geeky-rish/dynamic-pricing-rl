"""
Batch Experiments Runner
========================
Run multiple experiments with different configurations for sensitivity analysis.

Usage:
    python run_experiments.py --experiment sensitivity
    python run_experiments.py --experiment hyperparameter
    python run_experiments.py --experiment scalability

Author: ML Research Team
License: MIT
"""

import argparse
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from env import DynamicPricingEnv
from train import TrainingConfig, run_training_experiment


def run_sensitivity_analysis():
    """
    Test sensitivity to key environment parameters:
    - Inventory levels
    - Arrival rates
    - Seasonality amplitude
    """
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS EXPERIMENTS")
    print("="*80)
    
    base_output = "results/sensitivity"
    os.makedirs(base_output, exist_ok=True)
    
    # Base configuration
    base_config = {
        'episode_length': 90,
        'initial_inventory': 1000,
        'base_price': 50.0,
        'arrival_rate': 20.0,
        'seasonality_amplitude': 0.3
    }
    
    # Experiments: vary one parameter at a time
    experiments = {
        'inventory': {
            'param': 'initial_inventory',
            'values': [500, 750, 1000, 1500, 2000],
            'label': 'Initial Inventory'
        },
        'arrival_rate': {
            'param': 'arrival_rate',
            'values': [10, 15, 20, 30, 40],
            'label': 'Customer Arrival Rate'
        },
        'seasonality': {
            'param': 'seasonality_amplitude',
            'values': [0.0, 0.1, 0.3, 0.5, 0.7],
            'label': 'Seasonality Amplitude'
        }
    }
    
    all_results = {}
    
    for exp_name, exp_config in experiments.items():
        print(f"\n{'-'*80}")
        print(f"Experiment: {exp_name.upper()}")
        print(f"Varying: {exp_config['label']}")
        print(f"{'-'*80}")
        
        exp_results = []
        
        for value in exp_config['values']:
            print(f"\n  Testing {exp_config['label']} = {value}")
            
            # Create modified environment config
            env_config = base_config.copy()
            env_config[exp_config['param']] = value
            
            # Create environment
            env = DynamicPricingEnv(**env_config, seed=42)
            
            # Quick training with PPO only (for speed)
            output_dir = os.path.join(base_output, f"{exp_name}_{value}")
            
            config = TrainingConfig(
                total_timesteps=50000,  # Reduced for sensitivity analysis
                algorithms=['ppo'],
                output_dir=output_dir,
                seed=42
            )
            
            # Override environment parameters
            config.episode_length = env_config['episode_length']
            config.initial_inventory = env_config['initial_inventory']
            config.base_price = env_config['base_price']
            config.arrival_rate = env_config['arrival_rate']
            
            results, models = run_training_experiment(config)
            
            exp_results.append({
                'value': value,
                'reward': results['ppo']['eval_reward']
            })
        
        all_results[exp_name] = {
            'parameter': exp_config['label'],
            'values': [r['value'] for r in exp_results],
            'rewards': [r['reward'] for r in exp_results]
        }
    
    # Save sensitivity results
    results_file = os.path.join(base_output, 'sensitivity_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Plot sensitivity curves
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (exp_name, exp_data) in zip(axes, all_results.items()):
        ax.plot(exp_data['values'], exp_data['rewards'], 
                marker='o', linewidth=2, markersize=8)
        ax.set_xlabel(exp_data['parameter'], fontsize=11)
        ax.set_ylabel('PPO Evaluation Reward', fontsize=11)
        ax.set_title(f"Sensitivity to {exp_data['parameter']}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(base_output, 'sensitivity_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Sensitivity curves saved to {plot_path}")
    plt.close()
    
    print(f"\n✓ Sensitivity analysis complete!")
    print(f"Results saved to {base_output}/")
    
    return all_results


def run_hyperparameter_tuning():
    """
    Test different hyperparameters for PPO:
    - Learning rates
    - Batch sizes
    - Number of steps
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING EXPERIMENTS")
    print("="*80)
    
    base_output = "results/hyperparameter"
    os.makedirs(base_output, exist_ok=True)
    
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from baselines import evaluate_policy
    
    # Create environment
    env = DynamicPricingEnv(episode_length=90, seed=42)
    
    # Hyperparameter grid
    hp_grid = {
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'batch_size': [32, 64, 128],
        'n_steps': [1024, 2048, 4096]
    }
    
    results = []
    
    total_combinations = len(hp_grid['learning_rate']) * len(hp_grid['batch_size']) * len(hp_grid['n_steps'])
    current = 0
    
    print(f"\nTesting {total_combinations} hyperparameter combinations...")
    print("(Using 50K timesteps per config for speed)\n")
    
    for lr in hp_grid['learning_rate']:
        for bs in hp_grid['batch_size']:
            for ns in hp_grid['n_steps']:
                current += 1
                print(f"[{current}/{total_combinations}] LR={lr}, BS={bs}, Steps={ns}")
                
                env_train = Monitor(env)
                
                model = PPO(
                    "MlpPolicy",
                    env_train,
                    learning_rate=lr,
                    batch_size=bs,
                    n_steps=ns,
                    verbose=0,
                    seed=42
                )
                
                model.learn(total_timesteps=50000, progress_bar=True)
                
                eval_reward = evaluate_policy(model, env, n_eval_episodes=10, seed=42)
                
                results.append({
                    'learning_rate': lr,
                    'batch_size': bs,
                    'n_steps': ns,
                    'eval_reward': float(eval_reward)
                })
                
                print(f"  Reward: {eval_reward:.2f}")
    
    # Find best configuration
    best_config = max(results, key=lambda x: x['eval_reward'])
    
    print(f"\n{'='*80}")
    print("BEST HYPERPARAMETERS:")
    print(f"{'='*80}")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  N Steps: {best_config['n_steps']}")
    print(f"  Reward: {best_config['eval_reward']:.2f}")
    print(f"{'='*80}")
    
    # Save results
    results_file = os.path.join(base_output, 'hyperparameter_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'all_results': results,
            'best_config': best_config
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Visualize hyperparameter impact
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Group by each parameter
    for ax, param in zip(axes, ['learning_rate', 'batch_size', 'n_steps']):
        param_values = sorted(set(r[param] for r in results))
        param_rewards = []
        
        for val in param_values:
            rewards = [r['eval_reward'] for r in results if r[param] == val]
            param_rewards.append(rewards)
        
        bp = ax.boxplot(param_rewards, labels=param_values)
        ax.set_xlabel(param.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel('Evaluation Reward', fontsize=11)
        ax.set_title(f"Impact of {param.replace('_', ' ').title()}", 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(base_output, 'hyperparameter_impact.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {plot_path}")
    plt.close()
    
    return results


def run_scalability_test():
    """
    Test scalability with different episode lengths and inventory sizes
    """
    print("\n" + "="*80)
    print("SCALABILITY EXPERIMENTS")
    print("="*80)
    
    base_output = "results/scalability"
    os.makedirs(base_output, exist_ok=True)
    
    # Test configurations
    configs = [
        {'episode_length': 30, 'inventory': 500, 'label': 'Small'},
        {'episode_length': 60, 'inventory': 1000, 'label': 'Medium'},
        {'episode_length': 90, 'inventory': 1500, 'label': 'Large'},
        {'episode_length': 180, 'inventory': 3000, 'label': 'XLarge'},
    ]
    
    results = []
    
    print(f"\nTesting {len(configs)} problem scales...\n")
    
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Testing {config['label']} scale:")
        print(f"  Episode Length: {config['episode_length']} days")
        print(f"  Initial Inventory: {config['inventory']} units")
        
        # Create environment
        env = DynamicPricingEnv(
            episode_length=config['episode_length'],
            initial_inventory=config['inventory'],
            seed=42
        )
        
        # Train PPO
        output_dir = os.path.join(base_output, f"scale_{config['label']}")
        
        train_config = TrainingConfig(
            episode_length=config['episode_length'],
            initial_inventory=config['inventory'],
            total_timesteps=100000,
            algorithms=['ppo'],
            output_dir=output_dir,
            seed=42
        )
        
        import time
        start_time = time.time()
        
        exp_results, models = run_training_experiment(train_config)
        
        training_time = time.time() - start_time
        
        results.append({
            'label': config['label'],
            'episode_length': config['episode_length'],
            'inventory': config['inventory'],
            'reward': exp_results['ppo']['eval_reward'],
            'training_time': training_time
        })
        
        print(f"  Training Time: {training_time:.1f} seconds")
        print(f"  Reward: {exp_results['ppo']['eval_reward']:.2f}\n")
    
    # Save results
    results_file = os.path.join(base_output, 'scalability_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    labels = [r['label'] for r in results]
    rewards = [r['reward'] for r in results]
    times = [r['training_time'] / 60 for r in results]  # Convert to minutes
    
    # Plot 1: Rewards
    axes[0].bar(labels, rewards, color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Evaluation Reward', fontsize=11)
    axes[0].set_title('Performance vs. Problem Scale', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Training time
    axes[1].bar(labels, times, color='coral', alpha=0.7)
    axes[1].set_ylabel('Training Time (minutes)', fontsize=11)
    axes[1].set_title('Computational Cost vs. Problem Scale', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(base_output, 'scalability_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {plot_path}")
    plt.close()
    
    print(f"\n✓ Scalability test complete!")
    print(f"Results saved to {base_output}/")
    
    return results


def run_ablation_study():
    """
    Ablation study: impact of environment features
    """
    print("\n" + "="*80)
    print("ABLATION STUDY")
    print("="*80)
    print("Testing impact of: seasonality, competitor, fairness penalty\n")
    
    base_output = "results/ablation"
    os.makedirs(base_output, exist_ok=True)
    
    from baselines import evaluate_policy
    
    # Test configurations
    configs = {
        'full': {
            'seasonality_amplitude': 0.3,
            'competitor_volatility': 0.1,
            'fairness_weight': 0.05,
            'label': 'Full Model'
        },
        'no_seasonality': {
            'seasonality_amplitude': 0.0,
            'competitor_volatility': 0.1,
            'fairness_weight': 0.05,
            'label': 'No Seasonality'
        },
        'no_competitor': {
            'seasonality_amplitude': 0.3,
            'competitor_volatility': 0.0,
            'fairness_weight': 0.05,
            'label': 'No Competitor'
        },
        'no_fairness': {
            'seasonality_amplitude': 0.3,
            'competitor_volatility': 0.1,
            'fairness_weight': 0.0,
            'label': 'No Fairness Penalty'
        },
        'minimal': {
            'seasonality_amplitude': 0.0,
            'competitor_volatility': 0.0,
            'fairness_weight': 0.0,
            'label': 'Minimal (No Features)'
        }
    }
    
    results = []
    
    for config_name, config in configs.items():
        print(f"Testing: {config['label']}")
        
        # Create environment
        env = DynamicPricingEnv(
            episode_length=90,
            seasonality_amplitude=config['seasonality_amplitude'],
            competitor_volatility=config['competitor_volatility'],
            fairness_weight=config['fairness_weight'],
            seed=42
        )
        
        # Train PPO
        output_dir = os.path.join(base_output, config_name)
        
        train_config = TrainingConfig(
            total_timesteps=100000,
            algorithms=['ppo'],
            output_dir=output_dir,
            seed=42
        )
        
        exp_results, models = run_training_experiment(train_config)
        
        results.append({
            'config': config_name,
            'label': config['label'],
            'reward': exp_results['ppo']['eval_reward'],
            **{k: v for k, v in config.items() if k != 'label'}
        })
        
        print(f"  Reward: {exp_results['ppo']['eval_reward']:.2f}\n")
    
    # Save results
    results_file = os.path.join(base_output, 'ablation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = [r['label'] for r in results]
    rewards = [r['reward'] for r in results]
    
    # Color code by configuration
    colors = ['green', 'orange', 'orange', 'orange', 'red']
    
    bars = ax.barh(labels, rewards, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Evaluation Reward', fontsize=12)
    ax.set_title('Ablation Study: Impact of Environment Features', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, reward in zip(bars, rewards):
        ax.text(reward, bar.get_y() + bar.get_height()/2, 
                f'{reward:.0f}', va='center', ha='left', 
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(base_output, 'ablation_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {plot_path}")
    plt.close()
    
    # Compute feature importance
    full_reward = next(r['reward'] for r in results if r['config'] == 'full')
    
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*80}")
    print(f"Baseline (Full Model): {full_reward:.2f}")
    print(f"{'-'*80}")
    
    for result in results:
        if result['config'] != 'full':
            impact = full_reward - result['reward']
            impact_pct = (impact / full_reward) * 100
            print(f"{result['label']:<30} Impact: {impact:>8.2f} ({impact_pct:>6.1f}%)")
    
    print(f"{'='*80}")
    print(f"\n✓ Ablation study complete!")
    print(f"Results saved to {base_output}/")
    
    return results


def main():
    """Main entry point for experiments"""
    parser = argparse.ArgumentParser(
        description="Run batch experiments for dynamic pricing"
    )
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['sensitivity', 'hyperparameter', 'scalability', 'ablation', 'all'],
        default='all',
        help='Type of experiment to run'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("DYNAMIC PRICING BATCH EXPERIMENTS")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiment Type: {args.experiment}")
    print("="*80)
    
    results = {}
    
    if args.experiment in ['sensitivity', 'all']:
        results['sensitivity'] = run_sensitivity_analysis()
    
    if args.experiment in ['hyperparameter', 'all']:
        results['hyperparameter'] = run_hyperparameter_tuning()
    
    if args.experiment in ['scalability', 'all']:
        results['scalability'] = run_scalability_test()
    
    if args.experiment in ['ablation', 'all']:
        results['ablation'] = run_ablation_study()
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.experiment == 'all':
        print("\n📊 Summary:")
        print("  - Sensitivity Analysis: results/sensitivity/")
        print("  - Hyperparameter Tuning: results/hyperparameter/")
        print("  - Scalability Test: results/scalability/")
        print("  - Ablation Study: results/ablation/")
    
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()
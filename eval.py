"""
Evaluation and Visualization for Dynamic Pricing
=================================================
Comprehensive evaluation metrics:
- Cumulative revenue
- Regret vs. oracle
- Stockout analysis
- Fairness metrics
- Seasonality effects
- Price dynamics visualization

Author: ML Research Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from env import DynamicPricingEnv
from baselines import evaluate_policy

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class DynamicPricingEvaluator:
    """Comprehensive evaluation of dynamic pricing algorithms"""
    
    def __init__(self, env: DynamicPricingEnv):
        self.env = env
        
    def evaluate_policy_detailed(
        self,
        model,
        n_episodes: int = 10,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Detailed evaluation with episode-level metrics
        
        Returns:
            Dictionary with comprehensive metrics
        """
        all_metrics = defaultdict(list)
        episode_trajectories = []
        
        for ep in range(n_episodes):
            # Reset environment
            obs, _ = self.env.reset(seed=seed + ep if seed else None)
            
            # Episode tracking
            trajectory = {
                'prices': [],
                'sales': [],
                'inventory': [],
                'competitor_prices': [],
                'rewards': [],
                'revenue': [],
                'days': []
            }
            
            episode_reward = 0.0
            episode_revenue = 0.0
            episode_sales = 0
            stockout_occurred = False
            prices_charged = []
            
            terminated, truncated = False, False
            step = 0
            
            while not (terminated or truncated):
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Track metrics
                price = info['our_price']
                prices_charged.append(price)
                
                trajectory['prices'].append(price)
                trajectory['sales'].append(info['sales'])
                trajectory['inventory'].append(info['inventory'])
                trajectory['competitor_prices'].append(info['competitor_price'])
                trajectory['rewards'].append(reward)
                trajectory['revenue'].append(info['revenue'])
                trajectory['days'].append(step)
                
                episode_reward += reward
                episode_revenue += info['revenue']
                episode_sales += info['sales']
                
                if info['inventory'] == 0:
                    stockout_occurred = True
                
                step += 1
            
            # Compute episode metrics
            all_metrics['episode_reward'].append(episode_reward)
            all_metrics['episode_revenue'].append(episode_revenue)
            all_metrics['episode_sales'].append(episode_sales)
            all_metrics['stockout'].append(1 if stockout_occurred else 0)
            all_metrics['avg_price'].append(np.mean(prices_charged))
            all_metrics['price_std'].append(np.std(prices_charged))
            all_metrics['episode_length'].append(step)
            
            episode_trajectories.append(trajectory)
        
        # Aggregate statistics
        results = {
            'mean_reward': float(np.mean(all_metrics['episode_reward'])),
            'std_reward': float(np.std(all_metrics['episode_reward'])),
            'mean_revenue': float(np.mean(all_metrics['episode_revenue'])),
            'std_revenue': float(np.std(all_metrics['episode_revenue'])),
            'mean_sales': float(np.mean(all_metrics['episode_sales'])),
            'stockout_rate': float(np.mean(all_metrics['stockout'])),
            'avg_price': float(np.mean(all_metrics['avg_price'])),
            'price_variability': float(np.mean(all_metrics['price_std'])),
            'trajectories': episode_trajectories,
            'raw_metrics': dict(all_metrics)
        }
        
        return results
    
    def compute_oracle_benchmark(
        self,
        n_episodes: int = 10,
        price_grid_size: int = 50
    ) -> float:
        """
        Compute clairvoyant oracle performance (with perfect foresight)
        Uses dynamic programming to find optimal pricing strategy
        
        Note: This is a simplified oracle that assumes perfect knowledge
        of customer arrivals and preferences
        """
        # For simplicity, we'll use the best static price as oracle
        # In practice, a true oracle would solve the stochastic DP
        
        from baselines import StaticPricingBaseline
        oracle = StaticPricingBaseline(
            self.env,
            price_grid=np.linspace(0.5, 2.0, price_grid_size),
            n_eval_episodes=n_episodes
        )
        oracle.train(verbose=False)
        
        return oracle.best_reward
    
    def compute_regret(
        self,
        model,
        oracle_reward: float,
        n_episodes: int = 10
    ) -> Dict:
        """Compute regret vs. oracle"""
        results = self.evaluate_policy_detailed(model, n_episodes=n_episodes)
        
        policy_reward = results['mean_reward']
        regret = oracle_reward - policy_reward
        relative_regret = regret / abs(oracle_reward) if oracle_reward != 0 else 0
        
        return {
            'absolute_regret': float(regret),
            'relative_regret': float(relative_regret),
            'policy_reward': float(policy_reward),
            'oracle_reward': float(oracle_reward)
        }
    
    def analyze_fairness(self, trajectories: List[Dict]) -> Dict:
        """Analyze price fairness across customers"""
        all_prices = []
        
        for traj in trajectories:
            all_prices.extend(traj['prices'])
        
        if len(all_prices) == 0:
            return {'fairness_score': 0.0}
        
        prices_array = np.array(all_prices)
        
        # Compute fairness metrics
        price_cv = np.std(prices_array) / np.mean(prices_array) if np.mean(prices_array) > 0 else 0
        price_range = np.max(prices_array) - np.min(prices_array)
        
        # Gini coefficient (inequality measure)
        sorted_prices = np.sort(prices_array)
        n = len(sorted_prices)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_prices)) / (n * np.sum(sorted_prices)) - (n + 1) / n
        
        return {
            'price_cv': float(price_cv),
            'price_range': float(price_range),
            'gini_coefficient': float(gini),
            'fairness_score': float(1 - gini)  # Higher is more fair
        }


def plot_learning_curves(results_dict: Dict, output_path: Optional[str] = None):
    """Plot learning curves for all algorithms"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Evaluation rewards over time
    ax = axes[0]
    
    for algo_name, results in results_dict.items():
        if 'history' in results:
            # Bandit algorithms
            episodes = results['history']['episodes']
            rewards = results['history']['rewards']
            
            # Smooth with moving average
            window = min(10, len(rewards) // 10)
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(episodes[:len(smoothed)], smoothed, label=results['name'], linewidth=2)
            else:
                ax.plot(episodes, rewards, label=results['name'], linewidth=2)
    
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Learning Curves (Bandit Algorithms)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final evaluation comparison
    ax = axes[1]
    
    algo_names = []
    eval_rewards = []
    
    for algo_name, results in results_dict.items():
        algo_names.append(results['name'])
        eval_rewards.append(results['eval_reward'])
    
    bars = ax.bar(range(len(algo_names)), eval_rewards, color=sns.color_palette("husl", len(algo_names)))
    ax.set_xticks(range(len(algo_names)))
    ax.set_xticklabels(algo_names, rotation=45, ha='right')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Final Evaluation Performance')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, eval_rewards)):
        ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.0f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_episode_trajectories(
    trajectories: List[Dict],
    model_name: str,
    output_path: Optional[str] = None
):
    """Plot detailed episode trajectories"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Use first trajectory for detailed view
    traj = trajectories[0]
    days = traj['days']
    
    # Plot 1: Prices over time
    ax = axes[0, 0]
    ax.plot(days, traj['prices'], label='Our Price', linewidth=2, color='blue')
    ax.plot(days, traj['competitor_prices'], label='Competitor Price', 
            linewidth=2, color='red', linestyle='--')
    ax.set_xlabel('Day')
    ax.set_ylabel('Price ($)')
    ax.set_title(f'{model_name}: Price Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Sales and Inventory
    ax = axes[0, 1]
    ax2 = ax.twinx()
    
    l1 = ax.bar(days, traj['sales'], alpha=0.6, label='Sales', color='green')
    l2 = ax2.plot(days, traj['inventory'], label='Inventory', 
                  linewidth=2, color='orange', marker='o', markersize=3)
    
    ax.set_xlabel('Day')
    ax.set_ylabel('Units Sold', color='green')
    ax2.set_ylabel('Inventory Level', color='orange')
    ax.tick_params(axis='y', labelcolor='green')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax.set_title(f'{model_name}: Sales and Inventory')
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines = [l1] + l2
    labels = ['Sales', 'Inventory']
    ax.legend(lines, labels, loc='upper right')
    
    # Plot 3: Revenue over time
    ax = axes[1, 0]
    cumulative_revenue = np.cumsum(traj['revenue'])
    ax.plot(days, cumulative_revenue, linewidth=2, color='purple')
    ax.fill_between(days, 0, cumulative_revenue, alpha=0.3, color='purple')
    ax.set_xlabel('Day')
    ax.set_ylabel('Cumulative Revenue ($)')
    ax.set_title(f'{model_name}: Cumulative Revenue')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Rewards over time
    ax = axes[1, 1]
    cumulative_rewards = np.cumsum(traj['rewards'])
    ax.plot(days, cumulative_rewards, linewidth=2, color='darkgreen')
    ax.fill_between(days, 0, cumulative_rewards, alpha=0.3, color='darkgreen')
    ax.set_xlabel('Day')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title(f'{model_name}: Cumulative Reward')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_seasonality_analysis(
    trajectories: List[Dict],
    output_path: Optional[str] = None
):
    """Analyze impact of seasonality on sales"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Aggregate sales by day of week
    day_of_week_sales = defaultdict(list)
    day_of_week_prices = defaultdict(list)
    
    for traj in trajectories:
        for day, sales, price in zip(traj['days'], traj['sales'], traj['prices']):
            dow = day % 7
            day_of_week_sales[dow].append(sales)
            day_of_week_prices[dow].append(price)
    
    # Plot 1: Sales by day of week
    ax = axes[0]
    days = sorted(day_of_week_sales.keys())
    avg_sales = [np.mean(day_of_week_sales[d]) for d in days]
    std_sales = [np.std(day_of_week_sales[d]) for d in days]
    
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.bar(days, avg_sales, yerr=std_sales, capsize=5, alpha=0.7, color='skyblue')
    ax.set_xticks(days)
    ax.set_xticklabels([day_names[d] for d in days])
    ax.set_ylabel('Average Sales')
    ax.set_title('Sales by Day of Week (Seasonality Effect)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Prices by day of week
    ax = axes[1]
    avg_prices = [np.mean(day_of_week_prices[d]) for d in days]
    std_prices = [np.std(day_of_week_prices[d]) for d in days]
    
    ax.bar(days, avg_prices, yerr=std_prices, capsize=5, alpha=0.7, color='coral')
    ax.set_xticks(days)
    ax.set_xticklabels([day_names[d] for d in days])
    ax.set_ylabel('Average Price ($)')
    ax.set_title('Pricing Strategy by Day of Week')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Seasonality analysis saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison_heatmap(
    results_dict: Dict,
    metrics: List[str] = None,
    output_path: Optional[str] = None
):
    """Create heatmap comparing algorithms across metrics"""
    if metrics is None:
        metrics = ['eval_reward', 'mean_revenue', 'stockout_rate', 'fairness_score']
    
    # Extract available metrics
    algo_names = []
    metric_values = defaultdict(list)
    
    for algo_name, results in results_dict.items():
        algo_names.append(results['name'])
        
        for metric in metrics:
            if metric in results:
                metric_values[metric].append(results[metric])
            else:
                metric_values[metric].append(0)
    
    # Create matrix
    matrix = []
    available_metrics = []
    
    for metric in metrics:
        if metric in metric_values and len(metric_values[metric]) > 0:
            # Normalize to [0, 1] for visualization
            values = np.array(metric_values[metric])
            if np.max(values) != np.min(values):
                normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
            else:
                normalized = values
            matrix.append(normalized)
            available_metrics.append(metric)
    
    if len(matrix) == 0:
        print("No metrics available for heatmap")
        return
    
    matrix = np.array(matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(algo_names)))
    ax.set_yticks(np.arange(len(available_metrics)))
    ax.set_xticklabels(algo_names, rotation=45, ha='right')
    ax.set_yticklabels(available_metrics)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Performance', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(available_metrics)):
        for j in range(len(algo_names)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Algorithm Comparison Across Metrics')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison heatmap saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_evaluation_report(
    models_dict: Dict,
    env: DynamicPricingEnv,
    output_dir: str = "results",
    n_eval_episodes: int = 10
):
    """Generate comprehensive evaluation report with all visualizations"""
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE EVALUATION REPORT")
    print("="*80)
    
    evaluator = DynamicPricingEvaluator(env)
    
    # Load training results if available
    results_file = os.path.join(output_dir, 'training_results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            training_results = json.load(f)
    else:
        training_results = {}
    
    # Detailed evaluation for each model
    detailed_results = {}
    
    print("\nEvaluating models...")
    for algo_name, model in models_dict.items():
        print(f"\n  Evaluating {algo_name}...")
        
        detailed = evaluator.evaluate_policy_detailed(
            model,
            n_episodes=n_eval_episodes,
            seed=42
        )
        
        # Add fairness analysis
        fairness = evaluator.analyze_fairness(detailed['trajectories'])
        detailed.update(fairness)
        
        detailed_results[algo_name] = detailed
        
        # Merge with training results
        if algo_name in training_results:
            training_results[algo_name].update({
                'mean_reward': detailed['mean_reward'],
                'mean_revenue': detailed['mean_revenue'],
                'stockout_rate': detailed['stockout_rate'],
                'fairness_score': detailed.get('fairness_score', 0)
            })
    
    # Compute oracle benchmark
    print("\n  Computing oracle benchmark...")
    oracle_reward = evaluator.compute_oracle_benchmark(n_episodes=n_eval_episodes)
    print(f"  Oracle reward: {oracle_reward:.2f}")
    
    # Compute regret for each model
    print("\n  Computing regret metrics...")
    for algo_name, model in models_dict.items():
        regret_info = evaluator.compute_regret(model, oracle_reward, n_episodes=n_eval_episodes)
        detailed_results[algo_name].update(regret_info)
        
        if algo_name in training_results:
            training_results[algo_name].update(regret_info)
    
    # Generate visualizations
    print("\n  Generating visualizations...")
    
    # 1. Learning curves
    plot_learning_curves(
        training_results,
        output_path=os.path.join(output_dir, 'learning_curves.png')
    )
    
    # 2. Episode trajectories for each model
    for algo_name, results in detailed_results.items():
        if 'trajectories' in results and len(results['trajectories']) > 0:
            model_name = training_results.get(algo_name, {}).get('name', algo_name)
            plot_episode_trajectories(
                results['trajectories'],
                model_name,
                output_path=os.path.join(output_dir, f'trajectory_{algo_name}.png')
            )
    
    # 3. Seasonality analysis (using best model)
    best_algo = max(detailed_results.items(), key=lambda x: x[1]['mean_reward'])
    plot_seasonality_analysis(
        best_algo[1]['trajectories'],
        output_path=os.path.join(output_dir, 'seasonality_analysis.png')
    )
    
    # 4. Comparison heatmap
    plot_comparison_heatmap(
        training_results,
        output_path=os.path.join(output_dir, 'comparison_heatmap.png')
    )
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, 'detailed_evaluation.json')
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(detailed_results)
    
    # Remove trajectories (too large for JSON)
    for algo_name in serializable_results:
        if 'trajectories' in serializable_results[algo_name]:
            del serializable_results[algo_name]['trajectories']
        if 'raw_metrics' in serializable_results[algo_name]:
            del serializable_results[algo_name]['raw_metrics']
    
    with open(detailed_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n  Detailed results saved to {detailed_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Algorithm':<20} {'Reward':<12} {'Revenue':<12} {'Regret':<12} {'Fairness':<10}")
    print("-"*80)
    
    for algo_name, results in detailed_results.items():
        model_name = training_results.get(algo_name, {}).get('name', algo_name)
        reward = results['mean_reward']
        revenue = results['mean_revenue']
        regret = results.get('relative_regret', 0) * 100  # As percentage
        fairness = results.get('fairness_score', 0)
        
        print(f"{model_name:<20} {reward:<12.2f} ${revenue:<11.2f} {regret:<11.1f}% {fairness:<10.3f}")
    
    print("-"*80)
    print(f"{'Oracle Benchmark':<20} {oracle_reward:<12.2f} {'N/A':<12} {'0.0%':<12} {'N/A':<10}")
    print("="*80)
    
    # Identify best model
    best_algo_name = max(detailed_results.items(), key=lambda x: x[1]['mean_reward'])
    best_model_name = training_results.get(best_algo_name[0], {}).get('name', best_algo_name[0])
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   - Mean Reward: {best_algo_name[1]['mean_reward']:.2f}")
    print(f"   - Mean Revenue: ${best_algo_name[1]['mean_revenue']:.2f}")
    print(f"   - Relative Regret: {best_algo_name[1].get('relative_regret', 0)*100:.1f}%")
    print(f"   - Fairness Score: {best_algo_name[1].get('fairness_score', 0):.3f}")
    print(f"   - Stockout Rate: {best_algo_name[1]['stockout_rate']*100:.1f}%")
    
    print("\n" + "="*80)
    print(f"All visualizations saved to {output_dir}/")
    print("="*80)
    
    return detailed_results


def main():
    """Main evaluation script"""
    import argparse
    from train import TrainingConfig
    
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory with trained models')
    parser.add_argument('--n-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    # Load environment
    env = DynamicPricingEnv(episode_length=90, seed=42)
    
    # Load trained models
    models = {}
    
    # Try to load deep RL models
    try:
        from stable_baselines3 import PPO, SAC, DDPG
        
        model_files = {
            'ppo': os.path.join(args.results_dir, 'ppo_final.zip'),
            'sac': os.path.join(args.results_dir, 'sac_final.zip'),
            'ddpg': os.path.join(args.results_dir, 'ddpg_final.zip')
        }
        
        for algo_name, model_path in model_files.items():
            if os.path.exists(model_path):
                if algo_name == 'ppo':
                    models[algo_name] = PPO.load(model_path)
                elif algo_name == 'sac':
                    models[algo_name] = SAC.load(model_path)
                elif algo_name == 'ddpg':
                    models[algo_name] = DDPG.load(model_path)
                print(f"Loaded {algo_name.upper()} model from {model_path}")
    except ImportError:
        print("Stable-Baselines3 not available, skipping deep RL models")
    
    if len(models) == 0:
        print("No trained models found. Please run train.py first.")
        return
    
    # Generate evaluation report
    results = generate_evaluation_report(
        models,
        env,
        output_dir=args.results_dir,
        n_eval_episodes=args.n_episodes
    )
    
    return results


if __name__ == "__main__":
    main()
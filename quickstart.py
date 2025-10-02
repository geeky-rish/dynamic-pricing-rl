"""
Quick-Start Example: Dynamic Pricing with PPO
==============================================
Trains PPO for 100K steps and compares with static baseline.

Usage:
    python quickstart.py

Author: ML Research Team
License: MIT
"""

import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from env import DynamicPricingEnv
from baselines import StaticPricingBaseline, evaluate_policy

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback
    SB3_AVAILABLE = True
except ImportError:
    print("ERROR: Stable-Baselines3 not installed!")
    print("Install with: pip install stable-baselines3 torch")
    SB3_AVAILABLE = False
    exit(1)

import matplotlib.pyplot as plt
import seaborn as sns


def quick_start_demo():
    """
    Quick demonstration of dynamic pricing with PPO vs. static baseline
    """
    print("\n" + "="*80)
    print("QUICK-START: DYNAMIC PRICING WITH PPO")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Configuration
    TOTAL_TIMESTEPS = 100000
    N_EVAL_EPISODES = 20
    SEED = 42
    OUTPUT_DIR = "results/quickstart"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  - Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  - Evaluation Episodes: {N_EVAL_EPISODES}")
    print(f"  - Random Seed: {SEED}")
    print(f"  - Output Directory: {OUTPUT_DIR}")
    
    # Step 1: Create Environment
    print("\n" + "-"*80)
    print("STEP 1: Creating Environment")
    print("-"*80)
    
    env = DynamicPricingEnv(
        episode_length=90,
        initial_inventory=1000,
        base_price=50.0,
        arrival_rate=20.0,
        seed=SEED
    )
    
    print("✓ Environment created successfully!")
    print(f"  - Episode Length: {env.episode_length} days")
    print(f"  - Initial Inventory: {env.initial_inventory} units")
    print(f"  - Base Price: ${env.base_price}")
    print(f"  - Expected Customers/Day: {env.arrival_rate}")
    print(f"  - Observation Space: {env.observation_space.shape}")
    print(f"  - Action Space: {env.action_space.shape}")
    
    # Step 2: Train Static Baseline
    print("\n" + "-"*80)
    print("STEP 2: Training Static Pricing Baseline")
    print("-"*80)
    
    static_model = StaticPricingBaseline(env, n_eval_episodes=10)
    static_model.train(verbose=True)
    
    static_eval_reward = evaluate_policy(
        static_model,
        env,
        n_eval_episodes=N_EVAL_EPISODES,
        seed=SEED
    )
    
    print(f"\n✓ Static Baseline Trained!")
    print(f"  - Best Price Multiplier: {static_model.best_price:.3f}")
    print(f"  - Evaluation Reward: {static_eval_reward:.2f}")
    
    # Step 3: Train PPO
    print("\n" + "-"*80)
    print("STEP 3: Training PPO Agent")
    print("-"*80)
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps...")
    print("This will take approximately 5-10 minutes on CPU...\n")
    
    # Wrap environment
    env_train = Monitor(env)
    
    # Create PPO model
    ppo_model = PPO(
        "MlpPolicy",
        env_train,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=SEED
    )
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(OUTPUT_DIR, 'ppo_best'),
        log_path=os.path.join(OUTPUT_DIR, 'ppo_logs'),
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # Train
    ppo_model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True
        #progress_bar=False  # Disable progress bar for compatibility
    )
    
    print("\n✓ PPO Training Complete!")
    
    # Save final model
    model_path = os.path.join(OUTPUT_DIR, 'ppo_final')
    ppo_model.save(model_path)
    print(f"  - Model saved to {model_path}")
    
    # Step 4: Evaluate PPO
    print("\n" + "-"*80)
    print("STEP 4: Evaluating PPO Agent")
    print("-"*80)
    
    ppo_eval_reward = evaluate_policy(
        ppo_model,
        env,
        n_eval_episodes=N_EVAL_EPISODES,
        seed=SEED
    )
    
    print(f"✓ PPO Evaluation Complete!")
    print(f"  - Evaluation Reward: {ppo_eval_reward:.2f}")
    
    # Step 5: Detailed Comparison
    print("\n" + "-"*80)
    print("STEP 5: Detailed Comparison")
    print("-"*80)
    
    # Run detailed episode for visualization
    def run_detailed_episode(model, model_name):
        """Run one episode and track all metrics"""
        obs, _ = env.reset(seed=SEED)
        
        trajectory = {
            'days': [],
            'prices': [],
            'sales': [],
            'inventory': [],
            'rewards': [],
            'revenue': [],
            'competitor_prices': []
        }
        
        episode_reward = 0.0
        terminated, truncated = False, False
        step = 0
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            trajectory['days'].append(step)
            trajectory['prices'].append(info['our_price'])
            trajectory['sales'].append(info['sales'])
            trajectory['inventory'].append(info['inventory'])
            trajectory['rewards'].append(reward)
            trajectory['revenue'].append(info['revenue'])
            trajectory['competitor_prices'].append(info['competitor_price'])
            
            episode_reward += reward
            step += 1
        
        return trajectory, episode_reward
    
    print("Running detailed episodes...")
    static_traj, static_reward = run_detailed_episode(static_model, "Static")
    ppo_traj, ppo_reward = run_detailed_episode(ppo_model, "PPO")
    
    print(f"\n  Static Pricing:")
    print(f"    - Total Reward: {static_reward:.2f}")
    print(f"    - Total Revenue: ${sum(static_traj['revenue']):.2f}")
    print(f"    - Total Sales: {sum(static_traj['sales'])} units")
    print(f"    - Avg Price: ${np.mean(static_traj['prices']):.2f}")
    
    print(f"\n  PPO Agent:")
    print(f"    - Total Reward: {ppo_reward:.2f}")
    print(f"    - Total Revenue: ${sum(ppo_traj['revenue']):.2f}")
    print(f"    - Total Sales: {sum(ppo_traj['sales'])} units")
    print(f"    - Avg Price: ${np.mean(ppo_traj['prices']):.2f}")
    
    # Calculate improvement
    reward_improvement = ((ppo_reward - static_reward) / abs(static_reward)) * 100
    revenue_improvement = ((sum(ppo_traj['revenue']) - sum(static_traj['revenue'])) / 
                           sum(static_traj['revenue'])) * 100
    
    print(f"\n  Improvement:")
    print(f"    - Reward: {reward_improvement:+.1f}%")
    print(f"    - Revenue: {revenue_improvement:+.1f}%")
    
    # Step 6: Visualizations
    print("\n" + "-"*80)
    print("STEP 6: Generating Visualizations")
    print("-"*80)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Price Strategies
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(static_traj['days'], static_traj['prices'], 
             label='Static Pricing', linewidth=2, color='blue', alpha=0.7)
    ax1.plot(ppo_traj['days'], ppo_traj['prices'], 
             label='PPO Pricing', linewidth=2, color='red', alpha=0.7)
    ax1.plot(ppo_traj['days'], ppo_traj['competitor_prices'], 
             label='Competitor', linewidth=2, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Day', fontsize=11)
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_title('Pricing Strategies Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Revenue
    ax2 = fig.add_subplot(gs[1, 0])
    static_cum_revenue = np.cumsum(static_traj['revenue'])
    ppo_cum_revenue = np.cumsum(ppo_traj['revenue'])
    ax2.plot(static_traj['days'], static_cum_revenue, 
             label='Static', linewidth=2, color='blue')
    ax2.plot(ppo_traj['days'], ppo_cum_revenue, 
             label='PPO', linewidth=2, color='red')
    ax2.fill_between(ppo_traj['days'], static_cum_revenue, ppo_cum_revenue,
                     where=(ppo_cum_revenue >= static_cum_revenue),
                     color='green', alpha=0.2, label='PPO Advantage')
    ax2.set_xlabel('Day', fontsize=11)
    ax2.set_ylabel('Cumulative Revenue ($)', fontsize=11)
    ax2.set_title('Cumulative Revenue Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Daily Sales
    ax3 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(static_traj['days']))
    width = 0.35
    ax3.bar(x - width/2, static_traj['sales'], width, 
            label='Static', alpha=0.7, color='blue')
    ax3.bar(x + width/2, ppo_traj['sales'], width, 
            label='PPO', alpha=0.7, color='red')
    ax3.set_xlabel('Day', fontsize=11)
    ax3.set_ylabel('Units Sold', fontsize=11)
    ax3.set_title('Daily Sales Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    # Only show every 10th day label to avoid crowding
    ax3.set_xticks(x[::10])
    ax3.set_xticklabels(static_traj['days'][::10])
    
    # Plot 4: Inventory Levels
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(static_traj['days'], static_traj['inventory'], 
             label='Static', linewidth=2, color='blue')
    ax4.plot(ppo_traj['days'], ppo_traj['inventory'], 
             label='PPO', linewidth=2, color='red')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Day', fontsize=11)
    ax4.set_ylabel('Inventory Level', fontsize=11)
    ax4.set_title('Inventory Management', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Performance Summary (Bar Chart)
    ax5 = fig.add_subplot(gs[2, 1])
    metrics = ['Reward', 'Revenue', 'Sales']
    static_metrics = [
        static_reward,
        sum(static_traj['revenue']),
        sum(static_traj['sales'])
    ]
    ppo_metrics = [
        ppo_reward,
        sum(ppo_traj['revenue']),
        sum(ppo_traj['sales'])
    ]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax5.bar(x_pos - width/2, static_metrics, width, 
                    label='Static', alpha=0.7, color='blue')
    bars2 = ax5.bar(x_pos + width/2, ppo_metrics, width, 
                    label='PPO', alpha=0.7, color='red')
    
    ax5.set_ylabel('Value', fontsize=11)
    ax5.set_title('Performance Summary', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(metrics, fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add percentage improvement labels
    for i, (s_val, p_val) in enumerate(zip(static_metrics, ppo_metrics)):
        improvement = ((p_val - s_val) / abs(s_val)) * 100
        color = 'green' if improvement > 0 else 'red'
        ax5.text(i, max(s_val, p_val), f'{improvement:+.1f}%',
                ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
    
    plt.suptitle('Quick-Start Demo: PPO vs Static Pricing', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    # Save figure
    plot_path = os.path.join(OUTPUT_DIR, 'quickstart_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {plot_path}")
    plt.close()
    
    # Additional plot: Price distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.hist(static_traj['prices'], bins=20, alpha=0.6, 
            label='Static', color='blue', edgecolor='black')
    ax.hist(ppo_traj['prices'], bins=20, alpha=0.6, 
            label='PPO', color='red', edgecolor='black')
    ax.axvline(np.mean(static_traj['prices']), color='blue', 
               linestyle='--', linewidth=2, label=f"Static Mean: ${np.mean(static_traj['prices']):.2f}")
    ax.axvline(np.mean(ppo_traj['prices']), color='red', 
               linestyle='--', linewidth=2, label=f"PPO Mean: ${np.mean(ppo_traj['prices']):.2f}")
    
    ax.set_xlabel('Price ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Price Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    dist_path = os.path.join(OUTPUT_DIR, 'price_distribution.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"✓ Price distribution saved to {dist_path}")
    plt.close()
    
    # Summary Report
    print("\n" + "="*80)
    print("QUICK-START SUMMARY")
    print("="*80)
    
    print(f"\n📊 RESULTS:")
    print(f"  Static Pricing: {static_eval_reward:.2f} reward")
    print(f"  PPO Agent:      {ppo_eval_reward:.2f} reward")
    print(f"  Improvement:    {((ppo_eval_reward - static_eval_reward) / abs(static_eval_reward) * 100):+.1f}%")
    
    print(f"\n💰 KEY METRICS:")
    print(f"  {'Metric':<20} {'Static':<15} {'PPO':<15} {'Change':<10}")
    print(f"  {'-'*60}")
    print(f"  {'Avg Reward':<20} {static_eval_reward:<15.2f} {ppo_eval_reward:<15.2f} "
          f"{((ppo_eval_reward - static_eval_reward) / abs(static_eval_reward) * 100):>9.1f}%")
    print(f"  {'Total Revenue':<20} ${sum(static_traj['revenue']):<14.2f} "
          f"${sum(ppo_traj['revenue']):<14.2f} {revenue_improvement:>9.1f}%")
    print(f"  {'Total Sales':<20} {sum(static_traj['sales']):<15} "
          f"{sum(ppo_traj['sales']):<15} "
          f"{((sum(ppo_traj['sales']) - sum(static_traj['sales'])) / sum(static_traj['sales']) * 100):>9.1f}%")
    
    print(f"\n🎯 INSIGHTS:")
    
    if ppo_eval_reward > static_eval_reward:
        print(f"  ✅ PPO successfully learned a dynamic pricing strategy!")
        print(f"  ✅ Achieved {((ppo_eval_reward - static_eval_reward) / abs(static_eval_reward) * 100):.1f}% "
              f"improvement over static pricing")
    else:
        print(f"  ⚠️  PPO did not outperform static baseline")
        print(f"  💡 Try training longer or adjusting hyperparameters")
    
    # Analyze PPO strategy
    price_std_static = np.std(static_traj['prices'])
    price_std_ppo = np.std(ppo_traj['prices'])
    
    print(f"\n  📈 PPO Strategy Analysis:")
    print(f"     - Price Variability: {price_std_ppo:.2f} (Static: {price_std_static:.2f})")
    
    if price_std_ppo > price_std_static * 1.5:
        print(f"     - PPO uses dynamic pricing (adapts to conditions)")
    else:
        print(f"     - PPO converged to near-static strategy")
    
    # Inventory management
    final_inv_static = static_traj['inventory'][-1]
    final_inv_ppo = ppo_traj['inventory'][-1]
    
    print(f"\n     - Final Inventory: {final_inv_ppo} units (Static: {final_inv_static})")
    
    if final_inv_ppo > final_inv_static:
        print(f"     - PPO was more conservative (less stockout risk)")
    else:
        print(f"     - PPO was more aggressive (better inventory turnover)")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  - Model: {OUTPUT_DIR}/ppo_final.zip")
    print(f"  - Logs: {OUTPUT_DIR}/ppo_logs/")
    print(f"  - Plots: {OUTPUT_DIR}/quickstart_comparison.png")
    print(f"  - Distribution: {OUTPUT_DIR}/price_distribution.png")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"  1. Train all baselines: python train.py")
    print(f"  2. Full evaluation: python eval.py")
    print(f"  3. Experiment with hyperparameters")
    print(f"  4. Extend to real datasets (Retailrocket, FreshRetailNet)")
    
    print(f"\n" + "="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return {
        'static_reward': static_eval_reward,
        'ppo_reward': ppo_eval_reward,
        'improvement': ((ppo_eval_reward - static_eval_reward) / abs(static_eval_reward)) * 100
    }


if __name__ == "__main__":
    if not SB3_AVAILABLE:
        print("Error: Stable-Baselines3 is required for this demo")
        exit(1)
    
    results = quick_start_demo()
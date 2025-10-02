"""
Training Script for Dynamic Pricing with Deep RL
=================================================
Trains and compares multiple baselines:
- Static Pricing
- LinUCB
- Thompson Sampling
- PPO (Stable-Baselines3)
- SAC (Stable-Baselines3)
- DDPG (Stable-Baselines3)

Author: ML Research Team
License: MIT
"""

import numpy as np
import argparse
import json
import os
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from env import DynamicPricingEnv
from baselines import (
    StaticPricingBaseline, 
    LinUCBPricing, 
    ThompsonSamplingPricing,
    train_bandit_baseline,
    evaluate_policy
)

# Deep RL imports
try:
    from stable_baselines3 import PPO, SAC, DDPG
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    print("Warning: Stable-Baselines3 not installed. Deep RL baselines will be skipped.")
    print("Install with: pip install stable-baselines3")
    SB3_AVAILABLE = False


class TrainingConfig:
    """Configuration for training experiments"""
    def __init__(
        self,
        # Environment parameters
        episode_length: int = 90,
        initial_inventory: int = 1000,
        base_price: float = 50.0,
        arrival_rate: float = 20.0,
        
        # Training parameters
        total_timesteps: int = 100000,
        n_bandit_episodes: int = 100,
        eval_freq: int = 5000,
        n_eval_episodes: int = 10,
        
        # Algorithm selection
        algorithms: list = None,
        
        # Output
        output_dir: str = "results",
        seed: int = 42
    ):
        # Environment
        self.episode_length = episode_length
        self.initial_inventory = initial_inventory
        self.base_price = base_price
        self.arrival_rate = arrival_rate
        
        # Training
        self.total_timesteps = total_timesteps
        self.n_bandit_episodes = n_bandit_episodes
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        # Algorithms to train (default: all)
        if algorithms is None:
            self.algorithms = ['static', 'linucb', 'thompson', 'ppo', 'sac', 'ddpg']
        else:
            self.algorithms = algorithms
        
        # Output
        self.output_dir = output_dir
        self.seed = seed
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)


def train_static_baseline(env: DynamicPricingEnv, config: TrainingConfig) -> Dict:
    """Train static pricing baseline"""
    print("\n" + "="*60)
    print("Training Static Pricing Baseline")
    print("="*60)
    
    model = StaticPricingBaseline(env, n_eval_episodes=config.n_eval_episodes)
    grid_results = model.train(verbose=True)
    
    # Evaluate best price
    eval_reward = evaluate_policy(model, env, n_eval_episodes=config.n_eval_episodes)
    
    results = {
        'name': 'Static Pricing',
        'best_price': float(model.best_price),
        'best_reward': float(model.best_reward),
        'eval_reward': float(eval_reward),
        'grid_results': [(float(p), float(r)) for p, r in grid_results]
    }
    
    print(f"Final Evaluation Reward: {eval_reward:.2f}")
    
    return results, model


def train_linucb_baseline(env: DynamicPricingEnv, config: TrainingConfig) -> Dict:
    """Train LinUCB baseline"""
    print("\n" + "="*60)
    print("Training LinUCB Baseline")
    print("="*60)
    
    model = LinUCBPricing(env, n_actions=10, alpha=1.0)
    history = train_bandit_baseline(
        model, 
        n_episodes=config.n_bandit_episodes,
        eval_every=max(1, config.n_bandit_episodes // 20),
        verbose=True
    )
    
    # Final evaluation
    eval_reward = evaluate_policy(model, env, n_eval_episodes=config.n_eval_episodes)
    
    results = {
        'name': 'LinUCB',
        'history': history,
        'eval_reward': float(eval_reward),
        'final_train_reward': float(history['rewards'][-1])
    }
    
    print(f"Final Evaluation Reward: {eval_reward:.2f}")
    
    return results, model


def train_thompson_baseline(env: DynamicPricingEnv, config: TrainingConfig) -> Dict:
    """Train Thompson Sampling baseline"""
    print("\n" + "="*60)
    print("Training Thompson Sampling Baseline")
    print("="*60)
    
    model = ThompsonSamplingPricing(env, n_actions=10)
    history = train_bandit_baseline(
        model,
        n_episodes=config.n_bandit_episodes,
        eval_every=max(1, config.n_bandit_episodes // 20),
        verbose=True
    )
    
    # Final evaluation
    eval_reward = evaluate_policy(model, env, n_eval_episodes=config.n_eval_episodes)
    
    results = {
        'name': 'Thompson Sampling',
        'history': history,
        'eval_reward': float(eval_reward),
        'final_train_reward': float(history['rewards'][-1])
    }
    
    print(f"Final Evaluation Reward: {eval_reward:.2f}")
    
    return results, model


def train_ppo_baseline(env: DynamicPricingEnv, config: TrainingConfig) -> Dict:
    """Train PPO (Stable-Baselines3)"""
    print("\n" + "="*60)
    print("Training PPO Baseline")
    print("="*60)
    
    # Wrap environment with Monitor
    env_train = Monitor(env)
    
    # Create PPO model
    model = PPO(
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
        seed=config.seed
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(config.output_dir, 'ppo_best'),
        log_path=os.path.join(config.output_dir, 'ppo_logs'),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True
    )
    
    # Train
    print(f"Training for {config.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Final evaluation
    eval_reward = evaluate_policy(model, env, n_eval_episodes=config.n_eval_episodes)
    
    results = {
        'name': 'PPO',
        'eval_reward': float(eval_reward),
        'total_timesteps': config.total_timesteps
    }
    
    print(f"Final Evaluation Reward: {eval_reward:.2f}")
    
    # Save model
    model_path = os.path.join(config.output_dir, 'ppo_final')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return results, model


def train_sac_baseline(env: DynamicPricingEnv, config: TrainingConfig) -> Dict:
    """Train SAC (Stable-Baselines3)"""
    print("\n" + "="*60)
    print("Training SAC Baseline")
    print("="*60)
    
    # Wrap environment with Monitor
    env_train = Monitor(env)
    
    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env_train,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        verbose=1,
        seed=config.seed
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(config.output_dir, 'sac_best'),
        log_path=os.path.join(config.output_dir, 'sac_logs'),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True
    )
    
    # Train
    print(f"Training for {config.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Final evaluation
    eval_reward = evaluate_policy(model, env, n_eval_episodes=config.n_eval_episodes)
    
    results = {
        'name': 'SAC',
        'eval_reward': float(eval_reward),
        'total_timesteps': config.total_timesteps
    }
    
    print(f"Final Evaluation Reward: {eval_reward:.2f}")
    
    # Save model
    model_path = os.path.join(config.output_dir, 'sac_final')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return results, model


def train_ddpg_baseline(env: DynamicPricingEnv, config: TrainingConfig) -> Dict:
    """Train DDPG (Stable-Baselines3)"""
    print("\n" + "="*60)
    print("Training DDPG Baseline")
    print("="*60)
    
    # Wrap environment with Monitor
    env_train = Monitor(env)
    
    # Create DDPG model
    model = DDPG(
        "MlpPolicy",
        env_train,
        learning_rate=1e-3,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        seed=config.seed
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(config.output_dir, 'ddpg_best'),
        log_path=os.path.join(config.output_dir, 'ddpg_logs'),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True
    )
    
    # Train
    print(f"Training for {config.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Final evaluation
    eval_reward = evaluate_policy(model, env, n_eval_episodes=config.n_eval_episodes)
    
    results = {
        'name': 'DDPG',
        'eval_reward': float(eval_reward),
        'total_timesteps': config.total_timesteps
    }
    
    print(f"Final Evaluation Reward: {eval_reward:.2f}")
    
    # Save model
    model_path = os.path.join(config.output_dir, 'ddpg_final')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return results, model


def run_training_experiment(config: TrainingConfig):
    """Run complete training experiment"""
    print("\n" + "="*80)
    print("DYNAMIC PRICING WITH DEEP REINFORCEMENT LEARNING")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Random Seed: {config.seed}")
    print(f"Algorithms: {', '.join(config.algorithms)}")
    print("="*80)
    
    # Create environment
    env = DynamicPricingEnv(
        episode_length=config.episode_length,
        initial_inventory=config.initial_inventory,
        base_price=config.base_price,
        arrival_rate=config.arrival_rate,
        seed=config.seed
    )
    
    # Dictionary to store all results
    all_results = {}
    all_models = {}
    
    # Train each algorithm
    if 'static' in config.algorithms:
        results, model = train_static_baseline(env, config)
        all_results['static'] = results
        all_models['static'] = model
    
    if 'linucb' in config.algorithms:
        results, model = train_linucb_baseline(env, config)
        all_results['linucb'] = results
        all_models['linucb'] = model
    
    if 'thompson' in config.algorithms:
        results, model = train_thompson_baseline(env, config)
        all_results['thompson'] = results
        all_models['thompson'] = model
    
    if SB3_AVAILABLE:
        if 'ppo' in config.algorithms:
            results, model = train_ppo_baseline(env, config)
            all_results['ppo'] = results
            all_models['ppo'] = model
        
        if 'sac' in config.algorithms:
            results, model = train_sac_baseline(env, config)
            all_results['sac'] = results
            all_models['sac'] = model
        
        if 'ddpg' in config.algorithms:
            results, model = train_ddpg_baseline(env, config)
            all_results['ddpg'] = results
            all_models['ddpg'] = model
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    for algo_name, results in all_results.items():
        print(f"{results['name']:20s}: Eval Reward = {results['eval_reward']:10.2f}")
    
    # Find best algorithm
    best_algo = max(all_results.items(), key=lambda x: x[1]['eval_reward'])
    print(f"\n🏆 Best Algorithm: {best_algo[1]['name']} "
          f"(Reward: {best_algo[1]['eval_reward']:.2f})")
    
    # Save results
    results_file = os.path.join(config.output_dir, 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return all_results, all_models


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train Dynamic Pricing algorithms"
    )
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total timesteps for deep RL (default: 100000)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Episodes for bandit algorithms (default: 100)')
    parser.add_argument('--algorithms', nargs='+', 
                       default=['static', 'linucb', 'thompson', 'ppo', 'sac', 'ddpg'],
                       help='Algorithms to train')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        total_timesteps=args.timesteps,
        n_bandit_episodes=args.episodes,
        algorithms=args.algorithms,
        output_dir=args.output,
        seed=args.seed
    )
    
    # Run training
    results, models = run_training_experiment(config)
    
    return results, models


if __name__ == "__main__":
    main()
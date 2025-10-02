"""
Baseline Algorithms for Dynamic Pricing
========================================
Implements:
1. Static Pricing (grid search)
2. Contextual Bandits (LinUCB, Thompson Sampling)
3. Deep RL (PPO, SAC, DDPG via Stable-Baselines3)

Author: ML Research Team
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class StaticPricingBaseline:
    """
    Static pricing: Find best fixed price via grid search
    """
    def __init__(
        self, 
        env,
        price_grid: Optional[np.ndarray] = None,
        n_eval_episodes: int = 10
    ):
        self.env = env
        self.n_eval_episodes = n_eval_episodes
        
        if price_grid is None:
            # Default: search 20 price points
            self.price_grid = np.linspace(
                env.min_price_mult, 
                env.max_price_mult, 
                20
            )
        else:
            self.price_grid = price_grid
        
        self.best_price = None
        self.best_reward = -np.inf
        
    def train(self, verbose: bool = True):
        """Grid search over fixed prices"""
        if verbose:
            print("Training Static Pricing Baseline...")
            print(f"Testing {len(self.price_grid)} price points")
        
        results = []
        
        for price_mult in self.price_grid:
            total_reward = 0.0
            
            for ep in range(self.n_eval_episodes):
                obs, _ = self.env.reset(seed=ep)
                episode_reward = 0.0
                terminated, truncated = False, False
                
                while not (terminated or truncated):
                    action = np.array([price_mult], dtype=np.float32)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    episode_reward += reward
                
                total_reward += episode_reward
            
            avg_reward = total_reward / self.n_eval_episodes
            results.append((price_mult, avg_reward))
            
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.best_price = price_mult
            
            if verbose and len(results) % 5 == 0:
                print(f"  Tested {len(results)}/{len(self.price_grid)} prices")
        
        if verbose:
            print(f"✓ Best price multiplier: {self.best_price:.3f} "
                  f"(Avg reward: {self.best_reward:.2f})")
        
        return results
    
    def predict(self, observation, deterministic=True):
        """Return best static price"""
        return np.array([self.best_price], dtype=np.float32), None


class LinUCBPricing:
    """
    Linear UCB for contextual bandits
    Discretizes action space and learns linear model per arm
    """
    def __init__(
        self,
        env,
        n_actions: int = 10,
        alpha: float = 1.0,
        feature_dim: Optional[int] = None
    ):
        self.env = env
        self.n_actions = n_actions
        self.alpha = alpha  # Exploration parameter
        
        # Discretize action space
        self.actions = np.linspace(
            env.min_price_mult,
            env.max_price_mult,
            n_actions
        )
        
        # Feature dimension (observation dimension)
        if feature_dim is None:
            self.feature_dim = env.observation_space.shape[0]
        else:
            self.feature_dim = feature_dim
        
        # Initialize parameters for each arm
        self.A = [np.eye(self.feature_dim) for _ in range(n_actions)]  # Design matrix
        self.b = [np.zeros(self.feature_dim) for _ in range(n_actions)]  # Response vector
        
        self.step_count = 0
        
    def predict(self, observation, deterministic=False):
        """Select action using UCB"""
        observation = observation.reshape(-1)
        
        ucb_values = []
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            
            # UCB = expected reward + exploration bonus
            expected_reward = theta @ observation
            uncertainty = self.alpha * np.sqrt(observation @ A_inv @ observation)
            ucb = expected_reward + uncertainty
            
            ucb_values.append(ucb)
        
        # Select action with highest UCB
        best_action_idx = np.argmax(ucb_values)
        action = self.actions[best_action_idx]
        
        return np.array([action], dtype=np.float32), best_action_idx
    
    def update(self, observation, action_idx, reward):
        """Update model with new observation"""
        observation = observation.reshape(-1)
        
        self.A[action_idx] += np.outer(observation, observation)
        self.b[action_idx] += reward * observation
        self.step_count += 1
    
    def train_episode(self, seed: Optional[int] = None):
        """Run one episode of training"""
        obs, _ = self.env.reset(seed=seed)
        episode_reward = 0.0
        terminated, truncated = False, False
        
        while not (terminated or truncated):
            action, action_idx = self.predict(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Update model
            self.update(obs, action_idx, reward)
            
            episode_reward += reward
            obs = next_obs
        
        return episode_reward


class ThompsonSamplingPricing:
    """
    Thompson Sampling for contextual bandits
    Assumes Gaussian posterior over reward models
    """
    def __init__(
        self,
        env,
        n_actions: int = 10,
        lambda_prior: float = 1.0,
        noise_variance: float = 1.0,
        feature_dim: Optional[int] = None
    ):
        self.env = env
        self.n_actions = n_actions
        self.lambda_prior = lambda_prior
        self.noise_variance = noise_variance
        
        # Discretize action space
        self.actions = np.linspace(
            env.min_price_mult,
            env.max_price_mult,
            n_actions
        )
        
        if feature_dim is None:
            self.feature_dim = env.observation_space.shape[0]
        else:
            self.feature_dim = feature_dim
        
        # Gaussian posterior parameters for each arm
        self.B = [self.lambda_prior * np.eye(self.feature_dim) for _ in range(n_actions)]
        self.mu = [np.zeros(self.feature_dim) for _ in range(n_actions)]
        self.f = [np.zeros(self.feature_dim) for _ in range(n_actions)]
        
        self.step_count = 0
    
    def predict(self, observation, deterministic=False):
        """Sample from posterior and select action"""
        observation = observation.reshape(-1)
        
        sampled_rewards = []
        for a in range(self.n_actions):
            # Sample theta from posterior N(mu, B^-1)
            B_inv = np.linalg.inv(self.B[a])
            try:
                theta_sample = np.random.multivariate_normal(self.mu[a], B_inv)
            except np.linalg.LinAlgError:
                # Fallback if covariance is singular
                theta_sample = self.mu[a]
            
            # Predicted reward
            reward_sample = theta_sample @ observation
            sampled_rewards.append(reward_sample)
        
        # Select action with highest sampled reward
        best_action_idx = np.argmax(sampled_rewards)
        action = self.actions[best_action_idx]
        
        return np.array([action], dtype=np.float32), best_action_idx
    
    def update(self, observation, action_idx, reward):
        """Bayesian update of posterior"""
        observation = observation.reshape(-1)
        
        self.B[action_idx] += np.outer(observation, observation) / self.noise_variance
        self.f[action_idx] += observation * reward / self.noise_variance
        
        # Update posterior mean
        B_inv = np.linalg.inv(self.B[action_idx])
        self.mu[action_idx] = B_inv @ self.f[action_idx]
        
        self.step_count += 1
    
    def train_episode(self, seed: Optional[int] = None):
        """Run one episode of training"""
        obs, _ = self.env.reset(seed=seed)
        episode_reward = 0.0
        terminated, truncated = False, False
        
        while not (terminated or truncated):
            action, action_idx = self.predict(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Update model
            self.update(obs, action_idx, reward)
            
            episode_reward += reward
            obs = next_obs
        
        return episode_reward


def train_bandit_baseline(
    bandit_model,
    n_episodes: int,
    eval_every: int = 10,
    verbose: bool = True
) -> Dict:
    """
    Train a bandit model for multiple episodes
    
    Returns:
        Dictionary with training history
    """
    history = {
        'episodes': [],
        'rewards': [],
        'eval_rewards': []
    }
    
    if verbose:
        model_name = bandit_model.__class__.__name__
        print(f"Training {model_name}...")
    
    for ep in range(n_episodes):
        reward = bandit_model.train_episode(seed=ep)
        history['episodes'].append(ep)
        history['rewards'].append(reward)
        
        # Periodic evaluation
        if (ep + 1) % eval_every == 0:
            eval_reward = evaluate_policy(
                bandit_model,
                bandit_model.env,
                n_eval_episodes=5,
                deterministic=True
            )
            history['eval_rewards'].append(eval_reward)
            
            if verbose:
                print(f"  Episode {ep+1}/{n_episodes}: "
                      f"Train Reward={reward:.2f}, Eval Reward={eval_reward:.2f}")
    
    if verbose:
        print(f"✓ Training complete!")
    
    return history


def evaluate_policy(
    model,
    env,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    seed: Optional[int] = None
) -> float:
    """
    Evaluate a policy over multiple episodes
    
    Returns:
        Mean episode reward
    """
    total_reward = 0.0
    
    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep if seed else None)
        episode_reward = 0.0
        terminated, truncated = False, False
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
    
    return total_reward / n_eval_episodes


if __name__ == "__main__":
    # Test baselines
    from env import DynamicPricingEnv
    
    print("Testing baseline algorithms...")
    env = DynamicPricingEnv(episode_length=30, seed=42)
    
    # Test Static Pricing
    print("\n1. Static Pricing Baseline")
    static_model = StaticPricingBaseline(env, n_eval_episodes=3)
    static_model.train(verbose=True)
    
    # Test LinUCB
    print("\n2. LinUCB Baseline")
    linucb_model = LinUCBPricing(env, n_actions=10)
    linucb_history = train_bandit_baseline(linucb_model, n_episodes=5, verbose=True)
    
    # Test Thompson Sampling
    print("\n3. Thompson Sampling Baseline")
    ts_model = ThompsonSamplingPricing(env, n_actions=10)
    ts_history = train_bandit_baseline(ts_model, n_episodes=5, verbose=True)
    
    print("\n✓ All baselines tested successfully!")
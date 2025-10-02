"""
Dynamic Pricing Environment with Deep RL
========================================
A Gym-style environment simulating e-commerce dynamic pricing with:
- Poisson customer arrivals
- Heterogeneous demand (MNL model)
- Inventory constraints
- Seasonality patterns
- Competitor dynamics
- Fairness penalties

Author: ML Research Team
License: MIT
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class DynamicPricingEnv(gym.Env):
    """
    Dynamic Pricing Environment for E-Commerce
    
    State Space:
        - Current inventory level (normalized)
        - Time in episode (day of season, normalized)
        - Day of week (one-hot encoded)
        - Competitor's price (normalized)
        - Recent sales velocity (moving average)
        - Price elasticity estimate (adaptive)
    
    Action Space:
        - Continuous price multiplier [0.5, 2.0] of base price
    
    Reward:
        - Revenue from sales
        - Penalty for stockouts
        - Fairness penalty (variance in prices across cohorts)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        episode_length: int = 90,  # Days in episode
        initial_inventory: int = 1000,
        base_price: float = 50.0,
        base_cost: float = 30.0,
        arrival_rate: float = 20.0,  # Mean customers per day
        seasonality_amplitude: float = 0.3,
        seasonality_period: int = 7,  # Weekly seasonality
        competitor_volatility: float = 0.1,
        stockout_penalty: float = 100.0,
        fairness_weight: float = 0.05,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        # Environment parameters
        self.episode_length = episode_length
        self.initial_inventory = initial_inventory
        self.base_price = base_price
        self.base_cost = base_cost
        self.arrival_rate = arrival_rate
        self.seasonality_amplitude = seasonality_amplitude
        self.seasonality_period = seasonality_period
        self.competitor_volatility = competitor_volatility
        self.stockout_penalty = stockout_penalty
        self.fairness_weight = fairness_weight
        
        # Price bounds (as multipliers of base price)
        self.min_price_mult = 0.5
        self.max_price_mult = 2.0
        
        # State tracking
        self.current_step = 0
        self.inventory = initial_inventory
        self.competitor_price = base_price  # Initialize at base
        self.sales_history = []  # Track recent sales
        self.price_history = []  # For fairness penalty
        self.cumulative_revenue = 0.0
        self.total_sales = 0
        
        # Random number generator
        self.np_random = np.random.RandomState(seed)
        
        # Define action space: continuous price multiplier
        self.action_space = spaces.Box(
            low=self.min_price_mult,
            high=self.max_price_mult,
            shape=(1,),
            dtype=np.float32
        )
        
        # Define observation space
        # [inventory_norm, time_norm, day_of_week(7), competitor_price_norm, 
        #  sales_velocity, price_elasticity]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12,),  # 1 + 1 + 7 + 1 + 1 + 1
            dtype=np.float32
        )
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        self.current_step = 0
        self.inventory = self.initial_inventory
        self.competitor_price = self.base_price
        self.sales_history = []
        self.price_history = []
        self.cumulative_revenue = 0.0
        self.total_sales = 0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step (one day) in the environment
        
        Args:
            action: Price multiplier [0.5, 2.0]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Extract price from action
        price_multiplier = np.clip(action[0], self.min_price_mult, self.max_price_mult)
        our_price = self.base_price * price_multiplier
        
        # Store price for fairness calculation
        self.price_history.append(our_price)
        
        # Calculate seasonality effect
        seasonality_factor = self._get_seasonality_factor()
        
        # Simulate customer arrivals (Poisson process)
        adjusted_arrival_rate = self.arrival_rate * seasonality_factor
        num_customers = self.np_random.poisson(adjusted_arrival_rate)
        
        # Simulate sales using MNL demand model
        sales, revenue, cohort_purchases = self._simulate_sales(
            num_customers, our_price, self.competitor_price
        )
        
        # Update inventory
        self.inventory = max(0, self.inventory - sales)
        self.total_sales += sales
        self.sales_history.append(sales)
        
        # Keep only recent history (last 7 days)
        if len(self.sales_history) > 7:
            self.sales_history.pop(0)
        if len(self.price_history) > 30:
            self.price_history.pop(0)
        
        # Calculate reward
        profit = revenue - (sales * self.base_cost)
        stockout_penalty = self.stockout_penalty if self.inventory == 0 else 0.0
        fairness_penalty = self._calculate_fairness_penalty(cohort_purchases)
        
        reward = profit - stockout_penalty - fairness_penalty
        self.cumulative_revenue += revenue
        
        # Update competitor price (random walk + sinusoidal)
        self._update_competitor_price()
        
        # Check termination conditions
        self.current_step += 1
        terminated = (self.inventory == 0)  # Stockout
        truncated = (self.current_step >= self.episode_length)  # End of episode
        
        # Get new observation
        obs = self._get_observation()
        info = self._get_info()
        info.update({
            'sales': sales,
            'revenue': revenue,
            'profit': profit,
            'our_price': our_price,
            'competitor_price': self.competitor_price,
            'stockout_penalty': stockout_penalty,
            'fairness_penalty': fairness_penalty
        })
        
        return obs, reward, terminated, truncated, info
    
    def _simulate_sales(
        self, 
        num_customers: int, 
        our_price: float, 
        competitor_price: float
    ) -> Tuple[int, float, Dict]:
        """
        Simulate customer purchases using Multinomial Logit (MNL) demand model
        
        Each customer has:
        - Price sensitivity (drawn from distribution)
        - Product preference (attractiveness)
        
        Returns:
            sales: Number of units sold
            revenue: Total revenue
            cohort_purchases: Dict mapping cohort to purchase count (for fairness)
        """
        if num_customers == 0 or self.inventory == 0:
            return 0, 0.0, {}
        
        sales = 0
        revenue = 0.0
        cohort_purchases = {'low_sensitivity': [], 'high_sensitivity': []}
        
        # Define customer cohorts with different price sensitivities
        # Low sensitivity: price_sens ~ N(0.02, 0.005)
        # High sensitivity: price_sens ~ N(0.05, 0.01)
        
        for _ in range(num_customers):
            if self.inventory - sales <= 0:
                break
            
            # Randomly assign customer to cohort
            is_high_sensitivity = self.np_random.rand() > 0.5
            
            if is_high_sensitivity:
                price_sensitivity = max(0.01, self.np_random.normal(0.05, 0.01))
                cohort = 'high_sensitivity'
            else:
                price_sensitivity = max(0.005, self.np_random.normal(0.02, 0.005))
                cohort = 'low_sensitivity'
            
            # MNL utility: U = attractiveness - price_sensitivity * price
            our_attractiveness = 3.0  # Base attractiveness of our product
            competitor_attractiveness = 2.5
            
            our_utility = our_attractiveness - price_sensitivity * our_price
            competitor_utility = competitor_attractiveness - price_sensitivity * competitor_price
            no_purchase_utility = 0.0  # Outside option
            
            # Compute choice probabilities (softmax)
            utilities = np.array([our_utility, competitor_utility, no_purchase_utility])
            exp_utilities = np.exp(utilities)
            probs = exp_utilities / exp_utilities.sum()
            
            # Customer makes choice
            choice = self.np_random.choice(3, p=probs)
            
            if choice == 0:  # Customer buys from us
                sales += 1
                revenue += our_price
                cohort_purchases[cohort].append(our_price)
        
        return sales, revenue, cohort_purchases
    
    def _calculate_fairness_penalty(self, cohort_purchases: Dict) -> float:
        """
        Calculate penalty for price discrimination across cohorts
        Uses coefficient of variation of prices charged
        """
        all_prices = []
        for prices in cohort_purchases.values():
            all_prices.extend(prices)
        
        if len(all_prices) < 2:
            return 0.0
        
        prices_array = np.array(all_prices)
        price_std = np.std(prices_array)
        price_mean = np.mean(prices_array)
        
        if price_mean == 0:
            return 0.0
        
        # Coefficient of variation as fairness metric
        cv = price_std / price_mean
        penalty = self.fairness_weight * cv * self.base_price
        
        return penalty
    
    def _get_seasonality_factor(self) -> float:
        """Calculate demand seasonality multiplier"""
        # Weekly sinusoidal pattern
        day_in_cycle = self.current_step % self.seasonality_period
        phase = 2 * np.pi * day_in_cycle / self.seasonality_period
        factor = 1.0 + self.seasonality_amplitude * np.sin(phase)
        return max(0.1, factor)  # Ensure positive
    
    def _update_competitor_price(self):
        """Update competitor's price using random walk + sinusoidal trend"""
        # Random walk component
        random_change = self.np_random.normal(0, self.competitor_volatility * self.base_price)
        
        # Sinusoidal trend (competitor has their own strategy)
        trend_phase = 2 * np.pi * self.current_step / 30  # Monthly cycle
        trend = 0.05 * self.base_price * np.sin(trend_phase)
        
        self.competitor_price += random_change + trend
        
        # Keep within reasonable bounds
        self.competitor_price = np.clip(
            self.competitor_price, 
            self.base_price * 0.6, 
            self.base_price * 1.8
        )
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector"""
        # Inventory (normalized)
        inv_norm = self.inventory / self.initial_inventory
        
        # Time in episode (normalized)
        time_norm = self.current_step / self.episode_length
        
        # Day of week (one-hot)
        day_of_week = self.current_step % 7
        day_one_hot = np.zeros(7)
        day_one_hot[day_of_week] = 1.0
        
        # Competitor price (normalized)
        comp_price_norm = self.competitor_price / (self.base_price * 2.0)
        
        # Sales velocity (recent average, normalized)
        if len(self.sales_history) > 0:
            sales_velocity = np.mean(self.sales_history) / self.arrival_rate
        else:
            sales_velocity = 0.5
        
        # Estimated price elasticity (simple heuristic)
        if len(self.sales_history) >= 2 and len(self.price_history) >= 2:
            price_change = self.price_history[-1] - self.price_history[-2]
            sales_change = self.sales_history[-1] - self.sales_history[-2]
            if price_change != 0 and self.sales_history[-2] != 0 and self.price_history[-2] != 0:
                elasticity = -(sales_change / self.sales_history[-2]) / (price_change / self.price_history[-2])
                elasticity_norm = np.clip(elasticity / 3.0, 0, 1)
            else:
                elasticity_norm = 0.5
        else:
            elasticity_norm = 0.5
        
        obs = np.array([
            inv_norm,
            time_norm,
            *day_one_hot,
            comp_price_norm,
            sales_velocity,
            elasticity_norm
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> Dict:
        """Return auxiliary information"""
        return {
            'step': self.current_step,
            'inventory': self.inventory,
            'cumulative_revenue': self.cumulative_revenue,
            'total_sales': self.total_sales,
        }
    
    def render(self):
        """Render environment state (optional)"""
        if self.current_step % 10 == 0:
            print(f"Day {self.current_step}: Inventory={self.inventory}, "
                  f"Revenue=${self.cumulative_revenue:.2f}, Sales={self.total_sales}")


if __name__ == "__main__":
    # Test the environment
    env = DynamicPricingEnv(seed=42)
    obs, info = env.reset()
    
    print("Environment initialized successfully!")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a few steps with random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}: Action={action[0]:.3f}, Reward={reward:.2f}, "
              f"Sales={info['sales']}, Inventory={info['inventory']}")
        
        if terminated or truncated:
            break
    
    print("\nEnvironment test completed!")
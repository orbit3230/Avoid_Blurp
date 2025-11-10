import kymnasium as kym
import gymnasium as gym
import pickle
import numpy as np
from typing import Dict, Any
import tensorflow as tf
from tensorflow import keras

# On-Policy Monte Carlo Control (can be changed)

# Method only for Manual Play
# kym.avoid_blurp.ManualPlayWrapper("kymnasium/AvoidBlurp-Normal-v0", debug=True).play()

# ---------- Helper Functions ----------
# TODO
# ---------------------------------

# ---------- Agent Class ----------
class Agent(kym.Agent) :
    # 생성자 메소드
    def __init__(self, model: keras.models.Model, seed: int = None) :
        self.model = model
        self.seed = seed
        self.random = np.random.default_rng(self.seed)

    def save(self, path: str) :
        pass  # TODO

    @classmethod
    def load(cls, path: str, n: int, m: int, alpha: float, gamma: float, epsilon: float, planning_steps: int) -> "kym.Agent" :
        pass  # TODO

    def act(self, observation: Any, info: Dict) -> Any :
        pass  # TODO
# ---------- End of Agent Class ----------

# ---------- Training & Testing ----------
def train() :
    # Environment
    env = gym.make(
        id = "kymnasium/AvoidBlurp-Normal-v0",
        render_mode = "rgb_array",
        bgm = False,
        obs_type = "custom"
    )
    observation, info = env.reset()
    agent = Agent(model=None, seed=42)  # TODO
    agent.save(f"./moka_avoid_blurp.pkl")
    env.close()

def test() :
    env = gym.make(
        id = "kymnasium/AvoidBlurp-Normal-v0",
        render_mode = "huma",
        bgm = True,
        obs_type = "custom"
    )
    for _ in range(10) :
        agent = Agent.load("./moka_avoid_blurp.pkl")
        observation, info = env.reset()
        done = False
        while not done :
            action = agent.act(observation, info)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.close()
# ---------- End of Training & Testing ----------
    
if __name__ == "__main__" :
    train()
    test()
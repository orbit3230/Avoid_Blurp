import kymnasium as kym
import gymnasium as gym
import numpy as np
from typing import Dict, Any
import tensorflow as tf
from tensorflow import keras

# On-Policy Monte Carlo Control (can be changed)

# Method only for Manual Play
# kym.avoid_blurp.ManualPlayWrapper("kymnasium/AvoidBlurp-Normal-v0", debug=True).play()

# ---------- Helper Functions ----------
# revised : player width/height, enemy width/height are fixed. So does not need to be included.
def observation_to_input(observation) :
    # Max values for normalization
    SCREEN_WIDTH = 600.0
    SCREEN_HEIGHT = 750.0
    PLAYER_MAX_VELOCITY = 10.0
    ENEMY_MAX_VELOCITY = 20.0
    ENEMY_MAX_ACCELERATION = 0.20
    
    player_obs = observation["player"]  # shape: (3, )
    player_features = np.array([
        player_obs[0] / SCREEN_WIDTH,
        player_obs[1] / SCREEN_HEIGHT,
        player_obs[4] / PLAYER_MAX_VELOCITY
    ])
    enemies_observation = observation["enemies"]
    enemies_features = enemies_observation[:, [0, 1, 4, 5]].astype(np.float32)
    for i in range(enemies_features.shape[0]) :
        enemies_features[i, 0] /= SCREEN_WIDTH
        enemies_features[i, 1] /= SCREEN_HEIGHT
        enemies_features[i, 2] /= ENEMY_MAX_VELOCITY
        enemies_features[i, 3] /= ENEMY_MAX_ACCELERATION
    
    
    input_obs = np.concatenate([player_features, enemies_features.flatten()])  # shape: (123, )
    return input_obs

def reward_shaping(truncated, terminated) :
    if truncated : return -100  # Failed
    if terminated : return 100  # Succeeded
    return 0.1  # Living reward
# ---------------------------------

# ---------- Agent Class ----------
class Agent(kym.Agent) :
    # 생성자 메소드
    def __init__(self, model: keras.models.Model, seed: int, gamma: float, epsilon: float) :
        self.model = model
        self.seed = seed
        self.gamma = gamma
        self.epsilon = epsilon
        keras.utils.set_random_seed(self.seed)
        self.random = np.random.default_rng(self.seed)
        
    # ---------- Agent Helper Methods ----------
    def action_selection(self, state) :
        if(self.random.random() < self.epsilon) :
            return self.random.choice([0, 1, 2])  # Exploration
        state_tensor = keras.ops.expand_dims(state, axis = 0) 
        Q = self.model(state_tensor).numpy()
        return self.random.choice(np.flatnonzero(Q == np.max((Q))))  # Exploitation (with tie-breaking)
    # -------------------------------------

    def save(self, path: str) :
        self.model.save(path)

    @classmethod
    def load(cls, path: str, seed: int, gamma: float, epsilon: float) -> "kym.Agent" :
        model = keras.models.load_model(path)
        return cls(model, seed, gamma, epsilon)

    def act(self, observation: Any, info: Dict) -> Any :
        state = observation_to_input(observation)
        return self.action_selection(state)
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
    
    # Hyperparameters
    gamma = 0.99
    epsilon_min = 0.1
    epsilon_start = 1.0
    epsilon_decay_rate = 0.99965
    episodes = 10000
    seed_ = 42
    
    # Neural Network Model Creation
    model = keras.models.Sequential([
        keras.layers.Input(shape = (123, )),  # player(3) + blurps(4*30) = 123
        keras.layers.Dense(
            units = 64,
            activation = keras.activations.relu,
            kernel_initializer = keras.initializers.HeNormal(seed = seed_),
        ),
        keras.layers.Dense(
            units = 64,
            activation = keras.activations.relu,
            kernel_initializer = keras.initializers.HeNormal(seed = seed_),
        ),
        keras.layers.Dense(
            units = 3,  # Number of actions (0: Stop, 1: Left, 2: Right)
            activation = keras.activations.linear,
            kernel_initializer = keras.initializers.HeNormal(seed = seed_),
        )
    ])
    
    # Agent Initialization
    agent = Agent(model, seed_, gamma, epsilon_start)
    
    # optimizer & loss function
    objective_function = keras.losses.Huber(delta = 1.0)
    optimizer = keras.optimizers.Adam(learning_rate = 0.00025, clipnorm = 1.0)
    
    # Training Loop
    for episode in range(episodes) :
        history = []  # (state, action, reward)
        observation, info = env.reset()
        done = False
        total_reward = 0.0
        # Each Episode
        while not done :
            state = observation_to_input(observation)
            # Epsilon-Greedy Action Selection
            action = agent.action_selection(state)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated
            reward = reward_shaping(truncated, terminated)
            total_reward += reward
            history.append((state, action, reward))
            observation = next_observation
        # Epsilon Decay
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay_rate)
        # Monte Carlo Update
        G = 0.0  # Return
        for(state, action, reward) in reversed(history) :
            G = reward + agent.gamma * G
            # Model update (by GradientTape)
            with tf.GradientTape() as tape :
                state_tensor = keras.ops.expand_dims(state, axis = 0)
                Q = model(state_tensor)
                Q_action = Q[:, action]
                G_tensor = tf.convert_to_tensor([G], dtype = tf.float32)
                loss = objective_function(G_tensor, Q_action)
            # Calculate gradients and Model update
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"Episode {episode + 1}/{episodes} completed. | Epsilon: {agent.epsilon:.4f} | Total Reward: {total_reward:.2f}", end="\r")      
    
    agent.save("./moka_v1.keras")
    env.close()

def test() :
    env = gym.make(
        id = "kymnasium/AvoidBlurp-Normal-v0",
        render_mode = "human",
        bgm = True,
        obs_type = "custom"
    )
    agent = Agent.load("./moka_v1.keras", seed = 42, gamma = 0.99, epsilon = 0.0)
    for _ in range(10) :    
        observation, info = env.reset()
        done = False
        while not done :
            action = agent.act(observation, info)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        time_elapsed = info.get("time_elapsed", 0.0)
        print(f"Time: {time_elapsed:.2f} sec", end = ' ')
        if terminated : print("=> Succeeded")
        else : print("=> Failed")
    
    env.close()
# ---------- End of Training & Testing ----------
    
if __name__ == "__main__" :
    train()
    test()
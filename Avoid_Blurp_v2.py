import kymnasium as kym
import gymnasium as gym
import numpy as np
from typing import Dict, Any
import tensorflow as tf
from tensorflow import keras

# SARSA (TD)

# Method only for Manual Play
# kym.avoid_blurp.ManualPlayWrapper("kymnasium/AvoidBlurp-Normal-v0", debug=True).play()

# ---------- Helper Functions ----------
def observation_to_input(observation) :
    player_obs = observation["player"]  # shape: (5, )
    enemies_obs = observation["enemies"].flatten()  # shape: (30, 6)
    input_obs = np.concatenate([player_obs, enemies_obs])  # shape: (185, )
    return input_obs

def reward_shaping(truncated, terminated) :
    if truncated : return -100  # Failed
    if terminated : return 100  # Succeeded
    # return 0.1  # Living reward
    return 0.0  # No living reward
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
    epsilon_decay_rate = 0.9995
    episodes = 10000
    seed_ = 42
    
    # Neural Network Model Creation
    model = keras.models.Sequential([
        keras.layers.Input(shape = (185, )),  # player(5) + blurps(6*30) = 185
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
            kernel_initializer = keras.initializers.GlorotNormal(seed = seed_),
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
        # Each Episode
        while not done :
            state = observation_to_input(observation)
            # Epsilon-Greedy Action Selection
            action = agent.action_selection(state)
            next_observation, reward, truncated, terminated, info = env.step(action)
            done = truncated or terminated
            reward = reward_shaping(truncated, terminated)
            # history.append((state, action, reward))
            # --- SARSA Update ---
            next_state = observation_to_input(next_observation)
            next_action = agent.action_selection(next_state)
            with tf.GradientTape() as tape :
                state_tensor = keras.ops.expand_dims(state, axis = 0)
                Q_s_a = model(state_tensor)[0, action]  # Q(s, a)
                if done : target = reward
                else :
                    next_state_tensor = keras.ops.expand_dims(next_state, axis = 0)
                    Q_snext_anext = model(next_state_tensor)[0, next_action]  # Q(s', a')
                    target = reward + agent.gamma * Q_snext_anext
                target_tensor = tf.convert_to_tensor([target], dtype = tf.float32)
                loss = objective_function(target_tensor, tf.convert_to_tensor([Q_s_a], dtype = tf.float32))
            # --- END of SARSA Update ---
            # Gradient update
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            observation = next_observation
            state = next_state
            action = next_action
            
        # Epsilon Decay
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay_rate)
        # Monte Carlo Update
        # G = 0.0  # Return
        # for(state, action, reward) in reversed(history) :
        #     G = reward + agent.gamma * G
        #     # Model update (by GradientTape)
        #     with tf.GradientTape() as tape :
        #         state_tensor = keras.ops.expand_dims(state, axis = 0)
        #         Q = model(state_tensor)
        #         Q_action = Q[:, action]
        #         G_tensor = tf.convert_to_tensor([G], dtype = tf.float32)
        #         loss = objective_function(G_tensor, Q_action)
        #     # Calculate gradients and Model update
        #     gradients = tape.gradient(loss, model.trainable_variables)
        #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"Episode {episode + 1}/{episodes} completed. | Epsilon: {agent.epsilon:.4f}", end="\r")      
    
    agent.save("./moka.keras")
    env.close()

def test() :
    env = gym.make(
        id = "kymnasium/AvoidBlurp-Normal-v0",
        render_mode = "human",
        bgm = True,
        obs_type = "custom"
    )
    agent = Agent.load("./moka.keras", seed = 42, gamma = 0.99, epsilon = 0.0)
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
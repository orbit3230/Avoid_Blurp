import kymnasium as kym
import gymnasium as gym
import numpy as np
from typing import Dict, Any
import collections
import random
import tensorflow as tf
from tensorflow import keras

# DDQN (Double Deep Q-Network from v7)
# Patch Note v10
# 1. Hyperparameter Tuning
# 2. Network Architecture Change (64 -> 256 units)
# 3. Replay Buffer Size Increase (50000 -> 100000)
# 4. Batch Size Increase (32 -> 64)
# 5. Target Network Update Frequency Increase (3000 -> 8000)
# 6. Epsilon Decay Strategy Change (Linear Decay over 90% episodes)
# 7. Minimum Replay Size to Start Training Increase (1000 -> 2000)
# 8. Reward Shaping Change (+/-10 for success/failure, +0.01 per step)
# 9. Optimizer Learning Rate Change (0.00025 -> 0.0001)

# Method only for Manual Play
# kym.avoid_blurp.ManualPlayWrapper("kymnasium/AvoidBlurp-Normal-v0", debug=True).play()

# ---------- Helper Functions ----------
# Desending order by enemy y position
def sort_enemies(enemies_observation) :
    sorted_indices = np.argsort(enemies_observation[:, 1])[::-1]  # [::-1] for descending order
    return enemies_observation[sorted_indices]

# revised : player width/height, enemy width/height are fixed. So does not need to be included.
def observation_to_input(observation) :
    # Max values for normalization
    SCREEN_WIDTH = 600.0
    SCREEN_HEIGHT = 750.0
    PLAYER_MAX_VELOCITY = 10.0
    ENEMY_MAX_VELOCITY = 20.0
    ENEMY_MAX_ACCELERATION = 0.20
    
    player_obs = observation["player"]
    player_features = np.array([
        player_obs[0] / SCREEN_WIDTH,
        player_obs[1] / SCREEN_HEIGHT,
        player_obs[4] / PLAYER_MAX_VELOCITY
    ])
    enemies_observation = sort_enemies(np.array(observation["enemies"]))
    enemies_features = enemies_observation[:, [0, 1, 4, 5]].astype(np.float32)
    for i in range(enemies_features.shape[0]) :
        enemies_features[i, 0] /= SCREEN_WIDTH
        enemies_features[i, 1] /= SCREEN_HEIGHT
        enemies_features[i, 2] /= ENEMY_MAX_VELOCITY
        enemies_features[i, 3] /= ENEMY_MAX_ACCELERATION
    
    input_obs = np.concatenate([player_features, enemies_features.flatten()])  # shape: (123, )
    return input_obs

def reward_shaping(truncated, terminated) :
    if(truncated) : return -10   # Failed
    if(terminated) : return 10   # Succeeded
    return 0.01
# ---------------------------------

# ---------- Replay Buffer Class ----------
# # Store Experience (S, A, R, S') & Sample Mini-batch
class ReplayBuffer :
    def __init__(self, max_size, batch_size, seed) :
        self.buffer = collections.deque(maxlen = max_size)
        self.batch_size = batch_size
        self.random = random.Random(seed)
        
    # Give bigger weight for terminated & truncated experiences
    def store_transition(self, state, action, reward, next_state, done) :
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample_batch(self) :
        batch = self.random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action, dtype = np.int32),
            np.array(reward, dtype = np.float32),
            np.array(next_state),
            np.array(done, dtype = np.float32)
        )
        
    def __len__(self) : return len(self.buffer)
# ------------------------------------

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
    epsilon_min = 0.05
    epsilon_start = 1.0
    # epsilon_decay_rate = 0.99967
    episodes = 10000
    epsilon_linear_decay = (epsilon_start - epsilon_min) / (episodes * 0.9)
    full_exploration = episodes * 0.1
    seed_ = 42
    # Hyperparameters for Replay Buffer
    replay_buffer_size = 100000
    batch_size = 64
    min_replay_size = 2000  # min buffer size to start training
    # Hyperparameters for Target Network
    target_update_frequency = 8000
    train_step_counter = 0
    
    # Neural Network Model Creation
    model = keras.models.Sequential([
        keras.layers.Input(shape = (123, )),  # player(3) + blurps(4*30) = 123
        keras.layers.Dense(
            units = 256,
            activation = keras.activations.relu,
            kernel_initializer = keras.initializers.HeNormal(seed = seed_),
        ),
        keras.layers.Dense(
            units = 256,
            activation = keras.activations.relu,
            kernel_initializer = keras.initializers.HeNormal(seed = seed_),
        ),
        keras.layers.Dense(
            units = 3,  # Number of actions (0: Stop, 1: Left, 2: Right)
            activation = keras.activations.linear,
            kernel_initializer = keras.initializers.GlorotNormal(seed = seed_),
        )
    ])
    
    # Target Network Creation
    target_model = keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())
    
    # Agent Initialization
    agent = Agent(model, seed_, gamma, epsilon_start)
    
    # optimizer & loss function
    objective_function = keras.losses.Huber(delta = 1.0)
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001, clipnorm = 1.0)
    
    # Replay Buffer Initialization
    replay_buffer = ReplayBuffer(replay_buffer_size, batch_size, seed_)
    
    # Training Loop
    for episode in range(episodes) :
        observation, info = env.reset()
        state = observation_to_input(observation)
        # action = agent.action_selection(state)
        done = False
        total_reward = 0.0
        # Each Episode
        while not done :
            action = agent.action_selection(state)
            # (A) -> (R, S')
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated
            reward = reward_shaping(truncated, terminated)
            total_reward += reward
            # (S') -> (A')
            next_state = observation_to_input(next_observation)
            replay_buffer.store_transition(state, action, reward, next_state, done)
            state = next_state
            if(len(replay_buffer) >= min_replay_size) :
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = replay_buffer.sample_batch()
                with tf.GradientTape() as tape :
                    # --- target Q-Value Calculation (DDQN) ---
                    Q_snext_main = model(next_states_batch)  # Q(s') from main model
                    best_actions = tf.argmax(Q_snext_main, axis = 1, output_type = tf.int32)  # Best actions from main model
                    Q_snext_target = target_model(next_states_batch)  # Q(s') from target model
                    batch_indices = tf.range(batch_size, dtype = tf.int32)
                    action_indices = tf.stack([batch_indices, best_actions], axis = 1)
                    ddqn_Q_snext = tf.gather_nd(Q_snext_target, action_indices)  # Q(s', a') from target model, which main model selected
                    target_Q_values = rewards_batch + agent.gamma * ddqn_Q_snext * (1.0 - dones_batch)
                    # --- Current Q-Value Calculation ---
                    Q_s = model(states_batch)  # Q(s) for all actions
                    batch_indices = tf.range(batch_size, dtype = tf.int32)
                    actions_indices = tf.stack([batch_indices, actions_batch], axis = 1)
                    Q_s_a = tf.gather_nd(Q_s, actions_indices)  # Q(s, a) for taken actions
                    # Loss Calculation
                    loss = objective_function(target_Q_values, Q_s_a)
                    
                # Gradient update
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                # Target Network Update
                train_step_counter += 1
                if(train_step_counter % target_update_frequency == 0) : target_model.set_weights(model.get_weights())
            
        # Epsilon Linear Decay
        # agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay_rate)
        if(episode < full_exploration) : pass
        else : agent.epsilon = max(epsilon_min, agent.epsilon - epsilon_linear_decay)
        print(f"Episode {episode + 1}/{episodes} completed. | Total Reward: {total_reward:.2f} | Alive Time: {info.get('time_elapsed', 0.0):.2f} sec | Epsilon: {agent.epsilon:.4f}", end="\r")
    
    agent.save("./moka_v9.keras")
    env.close()

def test() :
    env = gym.make(
        id = "kymnasium/AvoidBlurp-Normal-v0",
        render_mode = "human",
        bgm = True,
        obs_type = "custom"
    )
    agent = Agent.load("./moka_v9.keras", seed = 42, gamma = 0.99, epsilon = 0.0)
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
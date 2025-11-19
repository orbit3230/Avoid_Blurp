import kymnasium as kym
from Avoid_Blurp_v10 import Agent

agent = Agent.load("./moka_v10.keras", seed = 42, gamma = 0.99, epsilon = 0.0)

kym.evaluate(
    env_id = "kymnasium/AvoidBlurp-Normal-v0",
    agent = agent,
    render_mode = "human",
    bgm = True,
    obs_type = "custom"
)
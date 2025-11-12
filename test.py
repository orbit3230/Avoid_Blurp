import kymnasium as kym
from Avoid_Blurp import Agent

agent = Agent.load("./moka.keras", seed = 42, gamma = 0.99, epsilon = 0.0)

kym.evaluate(
    env_id = "kymnasium/AvoidBlurp-Normal-v0",
    agent = agent,
    render_mode = "human",
    bgm = True
)
from gym.envs.registration import register

register(
    id='CrowdSim-v0',
    entry_point='crowd_sim.envs:CrowdSim',
)

# register(
#     id='PyMiniSimEnv-v0',
#     entry_point='crowd_sim.envs:PyMiniSimEnv',
# )

register(
    id='SocNav-v0',
    entry_point='crowd_sim.envs.pms.envs.environments:SocialNavEnv'
)

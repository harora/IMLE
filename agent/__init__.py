from agent.agent_ae import PointAEAgent, PointVAEAgent
from agent.agent_gan import MainAgent
from agent.agent_imle import MainAgentImle


def get_agent(config):
    if config.module == 'ae':
        return PointAEAgent(config)
    elif config.module == 'vae':
        return PointVAEAgent(config)
    elif config.module == 'gan':
        return MainAgent(config)
    elif config.module == 'imle_gan':
        return MainAgentImle(config)
    else:
        raise ValueError

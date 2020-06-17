#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
env = gym.make("CartPole-v1")


# In[2]:


import json
import random

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments, run
from ray.tune.experiment import Experiment
from ray.tune.registry import register_env

from ray.tune.schedulers import PopulationBasedTraining


# In[3]:


# number of parallel workers
N_CPUS = 4
# number of rollouts per training iteration
N_ROLLOUTS = 1

ray.init(num_cpus=N_CPUS)#, object_store_memory=1000000000)


# In[4]:


def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config


# In[5]:


pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=4,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "vf_clip_param": lambda: random.uniform(20000, 50000),
            "lr": [5e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(N_CPUS*HORIZON, 2*N_CPUS*HORIZON),
        },
        custom_explore_fn=explore)


# In[6]:


# The algorithm or model to train. This may refer to "
#      "the name of a built-on algorithm (e.g. RLLib's DQN "
#      "or PPO), or a user-defined trainable function or "
# #      "class registered in the tune registry.")
alg_run = "DQN"

# HORIZON = 200
BATCH_SIZE = 200

agent_cls = get_agent_class(alg_run)
config = agent_cls._default_config.copy()
config["num_workers"] = N_CPUS - 1  # number of parallel workers
config["num_envs_per_worker"] = 1  # number of parallel workers
# config["num_gpus"] = 0.1
config["train_batch_size"] = 1000  # batch size
config["sample_batch_size"] = 200  # batch size
config["gamma"] = 0.99  # discount rate
config["model"].update({"fcnet_hiddens": [256]})  # size of hidden layers in network
config["log_level"] = "DEBUG"
config["n_step"] = 2
# config["noisy"] = True
config["num_atoms"] = 2

# save the flow params for replay
# flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
#                        indent=4)  # generating a string version of flow_params
# config['env_config']['flow_params'] = flow_json  # adding the flow_params to config dict
# config['env_config']['run'] = alg_run

# Call the utility function make_create_env to be able to 
# register the Flow env for this experiment
# create_env, gym_name = make_create_env(params=flow_params, version=0)

config["env"] = "CartPole-v1"
# Register as rllib env with Gym
# register_env(gym_name, create_env)


# In[7]:


exp = Experiment("cart_pole_tests", **{
        "run": alg_run,
        "config": {
            **config
        },
        "checkpoint_freq": 5,  # number of iterations between checkpoints
        "checkpoint_at_end": True,  # generate a checkpoint at the end
        "max_failures": 5,
        "stop": {  # stopping conditions
            "episode_reward_mean": 200,  # number of iterations to stop after
        },
        "num_samples": 1})


# In[8]:


trials = run_experiments(exp)


# In[ ]:





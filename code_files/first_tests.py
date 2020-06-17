#!/usr/bin/env python
# coding: utf-8

# In[1]:

# import os
# os.chdir('../../flow')
# the TestEnv environment is used to simply simulate the network
from flow.envs import TestEnv

# the Experiment class is used for running simulations
from flow.core.experiment import Experiment

# the base network class
from flow.networks import Network
from flow.envs import Env

# all other imports are standard
from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import EnvParams
from flow.core.params import TrafficLightParams
from flow.controllers import IDMController
from flow.core.params import SumoCarFollowingParams

# create some default parameters parameters
HORIZON=2000
env_params = EnvParams(horizon=HORIZON)
initial_config = InitialConfig()


# In[2]:


le_dir = "/home/valentin/Schreibtisch/personal_sumo_files"


# In[3]:


from flow.core.params import SumoParams

sim_params = SumoParams(render=False, sim_step=1, restart_instance=True)


# In[4]:


vehicles=VehicleParams()


# In[5]:


from flow.core.params import InFlows

inflow = InFlows()

inflow.add(veh_type="human",
           edge="gneE3",
           probability=0.03)
inflow.add(veh_type="human",
           edge="-460515716",
           probability=0.03)
inflow.add(veh_type="human",
           edge="-672687133",
           probability=0.03)
inflow.add(veh_type="human",
           edge="gneE34",
           probability=0.03)
inflow.add(veh_type="human",
           edge="gneE18",
           probability=0.03)
inflow.add(veh_type="human",
           edge="-385995923",
           probability=0.03)
inflow.add(veh_type="human",
           edge="gneE26",
           probability=0.03)
inflow.add(veh_type="human",
           edge="gneE27",
           probability=0.03)
inflow.add(veh_type="human",
           edge="gneE30",
           probability=0.03)
inflow.add(veh_type="human",
           edge="660768672#2",
           probability=0.03)
inflow.add(veh_type="human",
           edge="gneE47",
           probability=0.03)
inflow.add(veh_type="human",
           edge="gneE43",
           probability=0.03)
inflow.add(veh_type="human",
           edge="gneE7",
           probability=0.03)


# In[6]:


inflow.get()


# In[7]:


import os

net_params = NetParams(
    inflows=inflow,
    template={
        # network geometry features
        "net": os.path.join(le_dir, "lemgo.net.xml"),
        # features associated with the properties of drivers
        "vtype": os.path.join(le_dir, "vtypes.add.xml"),
        # features associated with the routes vehicles take
        "rou": os.path.join(le_dir, "testfile.rou.xml"),
        "det": os.path.join(le_dir, "lemgo.add.xml")
    }
)


# ## Create custom network with lane area detectors

# #### 3.2.3 Running the Modified Simulation
# 
# Finally, the fully imported simulation can be run as follows. 
# 
# **Warning**: the network takes time to initialize while the departure positions and times and vehicles are specified.

# In[ ]:


# create the network
network = Network(
    name="template",
    net_params=net_params,
    vehicles=vehicles
)

# create the environment
env = TestEnv(
    env_params=env_params,
    sim_params=sim_params,
    network=network
)


# In[8]:


# This is the custom environment
# Needs to be important in order to work properly in flow
from flow.envs.simple_env import SimpleEnv
env_name = SimpleEnv


# In[9]:


# Creating flow_params. Make sure the dictionary keys are as specified. 
flow_params = dict(
    # name of the experiment
    exp_tag="first_exp",
    # name of the flow environment the experiment is running on
    env_name=env_name,
    # name of the network class the experiment uses
    network=Network,
    # simulator that is used by the experiment
    simulator='traci',
    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=sim_params,
    # environment related parameters (see flow.core.params.EnvParams)
    env=env_params,
    # network-related parameters (see flow.core.params.NetParams and
    # the network's documentation or ADDITIONAL_NET_PARAMS component)
    net=net_params,
    # vehicles to be placed in the network at the start of a rollout 
    # (see flow.core.vehicles.Vehicles)
    veh=VehicleParams(),
    # (optional) parameters affecting the positioning of vehicles upon 
    # initialization/reset (see flow.core.params.InitialConfig)
    initial=initial_config
)


# In[10]:


import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments, run
from ray.tune.experiment import Experiment
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

from ray.tune.schedulers import PopulationBasedTraining


# In[11]:


# number of parallel workers
N_CPUS = 4
# number of rollouts per training iteration
N_ROLLOUTS = 1

ray.init(num_cpus=N_CPUS)


# In[12]:


def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config


# In[13]:


pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=5,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "vf_clip_param": lambda: random.uniform(20000, 50000),
            "lr": [5e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(3*HORIZON, 6*HORIZON),
        },
        custom_explore_fn=explore)


# In[14]:


# The algorithm or model to train. This may refer to "
#      "the name of a built-on algorithm (e.g. RLLib's DQN "
#      "or PPO), or a user-defined trainable function or "
#      "class registered in the tune registry.")
alg_run = "PPO"

BATCH_SIZE = HORIZON

agent_cls = get_agent_class(alg_run)
config = agent_cls._default_config.copy()
config["num_workers"] = N_CPUS - 1  # number of parallel workers
config["train_batch_size"] = BATCH_SIZE * 3  # batch size
config["gamma"] = 0.999  # discount rate
config["model"].update({"fcnet_hiddens": [16, 16]})  # size of hidden layers in network
config["use_gae"] = True  # using generalized advantage estimation
config["lambda"] = 0.97  
config["log_level"] = "WARN"
config["sgd_minibatch_size"] = min(16 * 1024, config["train_batch_size"])  # stochastic gradient descent
config["kl_target"] = 0.02  # target KL divergence
config["num_sgd_iter"] = 10  # number of SGD iterations
config["horizon"] = HORIZON  # rollout horizon
config["vf_clip_param"] = 40000.0

# save the flow params for replay
flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                       indent=4)  # generating a string version of flow_params
config['env_config']['flow_params'] = flow_json  # adding the flow_params to config dict
config['env_config']['run'] = alg_run

# Call the utility function make_create_env to be able to 
# register the Flow env for this experiment
create_env, gym_name = make_create_env(params=flow_params, version=0)

config["env"] = gym_name
# Register as rllib env with Gym
register_env(gym_name, create_env)


# In[15]:


# The algorithm or model to train. This may refer to "
#      "the name of a built-on algorithm (e.g. RLLib's DQN "
#      "or PPO), or a user-defined trainable function or "
#      "class registered in the tune registry.")
#alg_run = "A3C"
#
#BATCH_SIZE = HORIZON * N_ROLLOUTS
#
#agent_cls = get_agent_class(alg_run)
#config = agent_cls._default_config.copy()
#config["num_workers"] = N_CPUS - 1  # number of parallel workers
#config["train_batch_size"] = BATCH_SIZE  # batch size
#config["sample_batch_size"] = BATCH_SIZE  # batch size
#config["gamma"] = 0.999  # discount rate
#config["model"].update({"fcnet_hiddens": [16, 16]})  # size of hidden layers in network
#config["horizon"] = HORIZON  # rollout horizon
#
## save the flow params for replay
#flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
#                       indent=4)  # generating a string version of flow_params
#config['env_config']['flow_params'] = flow_json  # adding the flow_params to config dict
#config['env_config']['run'] = alg_run
#
## Call the utility function make_create_env to be able to 
## register the Flow env for this experiment
#create_env, gym_name = make_create_env(params=flow_params, version=0)
#
## Register as rllib env with Gym
#register_env(gym_name, create_env)


# In[16]:


exp = Experiment(flow_params["exp_tag"], **{
        "run": alg_run,
        "config": {
            **config
        },
        "checkpoint_freq": 5,  # number of iterations between checkpoints
        "checkpoint_at_end": True,  # generate a checkpoint at the end
        "max_failures": 5,
        "stop": {  # stopping conditions
            "training_iteration": 10,  # number of iterations to stop after
        },
        "num_samples": 8})


# In[ ]:


trials = run_experiments(exp, scheduler=pbt)


# In[ ]:






import sys
import os

from flow.envs import TestEnv
from flow.core.experiment import Experiment
from flow.networks import Network
from flow.envs import Env

from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import EnvParams
from flow.core.params import TrafficLightParams
from flow.controllers import IDMController
from flow.core.params import SumoCarFollowingParams

HORIZON=2000
env_params = EnvParams(horizon=HORIZON)
initial_config = InitialConfig()

le_dir = "/home/valentin/Schreibtisch/personal_sumo_files"

from flow.core.params import SumoParams

sim_params = SumoParams(render=False, sim_step=1, restart_instance=True)

vehicles=VehicleParams()

from flow.core.params import InFlows

inflow = InFlows()

inflow.add(veh_type="human",
           edge="right_east",
           probability=0.16)
inflow.add(veh_type="human",
           edge="right_south",
           probability=0.16)
inflow.add(veh_type="human",
           edge="right_north",
           probability=0.16)
inflow.add(veh_type="human",
           edge="left_north",
           probability=0.16)
inflow.add(veh_type="human",
           edge="left_south",
           probability=0.16)
inflow.add(veh_type="human",
           edge="left_west",
           probability=0.16)

net_params = NetParams(
    inflows=inflow,
    template={
        # network geometry features
        "net": os.path.join(le_dir, "lemgo_small.net.xml"),
        # features associated with the properties of drivers
        "vtype": os.path.join(le_dir, "vtypes.add.xml"),
        # features associated with the routes vehicles take
        "rou": os.path.join(le_dir, "lemgo_small2_out.rou.xml"),
        "det": os.path.join(le_dir, "lemgo_small.add.xml")
    }
)


from flow.envs.simple_env import SimpleEnv
env_name = SimpleEnv

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


# In[21]:


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

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

from ray.tune.schedulers import PopulationBasedTraining


# In[22]:


# number of parallel workers
N_CPUS = 1
N_GPUS = 0
# number of rollouts per training iteration
N_ROLLOUTS = 1

ray.init(num_cpus=N_CPUS, num_gpus=N_GPUS)#, object_store_memory=1000000000)


# In[23]:


def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config


# In[24]:


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


alg_run = "A2C"

BATCH_SIZE = HORIZON * N_ROLLOUTS
 
agent_cls = get_agent_class(alg_run)
config = agent_cls._default_config.copy()
config["num_workers"] = N_CPUS - 1  # number of parallel workers
config["num_gpus"] = N_GPUS
config["train_batch_size"] = BATCH_SIZE * N_CPUS # batch size
config["sample_batch_size"] = BATCH_SIZE  # batch size
config["gamma"] = 0.999  # discount rate
config["model"].update({"fcnet_hiddens": [100, 100]})  # size of hidden layers in network
config["horizon"] = HORIZON  # rollout horizon
# 
# # save the flow params for replay
flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)  # generating a string version of flow_params
config['env_config']['flow_params'] = flow_json  # adding the flow_params to config dict
config['env_config']['run'] = alg_run
# 
# # Call the utility function make_create_env to be able to 
# # register the Flow env for this experiment
create_env, gym_name = make_create_env(params=flow_params, version=0)
# 
config["env"] = gym_name
# # Register as rllib env with Gym
register_env(gym_name, create_env)


# In[27]:


exp = Experiment(flow_params["exp_tag"], **{
        "run": alg_run,
        "config": {
            **config
        },
        "checkpoint_freq": 5,  # number of iterations between checkpoints
        "checkpoint_at_end": True,  # generate a checkpoint at the end
        "max_failures": 5,
        "stop": {  # stopping conditions
            "training_iteration": 4,  # number of iterations to stop after
        },
        "num_samples": 1})


# In[28]:


trials = run_experiments(exp)



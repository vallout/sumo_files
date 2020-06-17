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
from ray import tune

HORIZON=3000
env_params = EnvParams(horizon=HORIZON)
initial_config = InitialConfig()

le_dir = "/Schreibtisch/wrapper/Schreibtisch/sumo_files/personal_sumo_files"


from flow.core.params import SumoParams

sim_params = SumoParams(render=False, sim_step=1, restart_instance=True)

vehicles=VehicleParams()


from flow.core.params import InFlows

inflow = InFlows()

inflow.add(veh_type="human",
           edge="right_east",
           probability=0.04)
inflow.add(veh_type="human",
           edge="right_south",
           probability=0.04)
inflow.add(veh_type="human",
           edge="right_north",
           probability=0.04)
inflow.add(veh_type="human",
           edge="left_north",
           probability=0.04)
inflow.add(veh_type="human",
           edge="left_south",
           probability=0.04)
inflow.add(veh_type="human",
           edge="left_west",
           probability=0.04)

net_params = NetParams(
    inflows=inflow,
    template={
        # network geometry features
        "net": os.path.join(le_dir, "lemgo_small.net.xml"),
        # features associated with the properties of drivers
        "vtype": os.path.join(le_dir, "vtypes.add.xml"),
        # features associated with the routes vehicles take
        "rou": os.path.join(le_dir, "lemgo_small2_out.rou.xml"),
        # features associated with the detectors used
        "det": os.path.join(le_dir, "lemgo_small.add.xml")
    }
)

# This is the custom environment
from flow.envs.simple_env import SimpleEnv
env_name = SimpleEnv

# Creating flow_params. Make sure the dictionary keys are as specified. 
flow_params = dict(
    # name of the experiment
    exp_tag="train_batch_scaling_apex",
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


# number of parallel workers
N_CPUS = 80
# number of gpus used in general
N_GPUS = 8
# number of rollouts per training iteration
N_ROLLOUTS = 2

ray.init(num_cpus=N_CPUS, num_gpus=N_GPUS)

def explore(config):
    # do nothing
    return config

pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=2,
        resample_probability=0.5,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lr": [5e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "train_batch_size": [32, 64, 128, 256],
            "sample_batch_size": [4, 6, 8, 12, 16]
        },
        custom_explore_fn=explore
    )


alg_run = "APEX"

BATCH_SIZE = HORIZON * N_ROLLOUTS

configs = []

agent_cls = get_agent_class(alg_run)
config16 = agent_cls._default_config.copy()
config16["lr"] = 0.01
config16["num_workers"] = 60  # number of parallel workers
config16["num_gpus"] = 8
config16["train_batch_size"] = 16  # batch size
config16["sample_batch_size"] = 50  # batch size
config16["gamma"] = 0.997  # discount rate
config16["model"].update({"fcnet_hiddens": [128]})  # size of hidden layers in network
config16["horizon"] = HORIZON  # rollout horizon

# save the flow params for replay
flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                       indent=4)  # generating a string version of flow_params
config16['env_config']['flow_params'] = flow_json  # adding the flow_params to config dict
config16['env_config']['run'] = alg_run

# Call the utility function make_create_env to be able to 
# register the Flow env for this experiment
create_env, gym_name = make_create_env(params=flow_params, version=0)

config16["env"] = gym_name
# Register as rllib env with Gym
register_env(gym_name, create_env)

configs.append(config16)
config32 = config16.copy()
config32["train_batch_size"] = 32
configs.append(config32)
config64 = config16.copy()
config64["train_batch_size"] = 64
configs.append(config64)
config128 = config16.copy()
config128["train_batch_size"] = 128
configs.append(config128)
config256 = config16.copy()
config256["train_batch_size"] = 256
configs.append(config256)
config512 = config16.copy()
config512["train_batch_size"] = 512
configs.append(config512)
config1024 = config16.copy()
config1024["train_batch_size"] = 1024
configs.append(config1024)
config2048 = config16.copy()
config2048["train_batch_size"] = 2048
configs.append(config2048)


for config in configs:
    exp = Experiment("train_batch_scaling_apex_{}".format(config["train_batch_size"]), **{
            "run": alg_run,
            "config": {
                **config
            },
            "checkpoint_freq": 2,  # number of iterations between checkpoints
            "checkpoint_at_end": True,  # generate a checkpoint at the end
            "max_failures": 5,
            "keep_checkpoints_num": 4,
            "stop": {  # stopping conditions
                "timesteps_total": 2400000,  # number of iterations to stop after
            },
            "num_samples": 10, 
            "local_dir": "/tmp/ray_results"
        })

    trials = run_experiments(exp)




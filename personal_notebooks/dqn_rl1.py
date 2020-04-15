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

HORIZON=2000
env_params = EnvParams(horizon=HORIZON)
initial_config = InitialConfig()

le_dir = "/Schreibtisch/wrapper/Schreibtisch/personal_sumo_files"


from flow.core.params import SumoParams

sim_params = SumoParams(render=False, sim_step=1, restart_instance=True)

vehicles=VehicleParams()


from flow.core.params import InFlows

inflow = InFlows()

inflow.add(veh_type="human",
           edge="right_east",
           probability=0.10)
inflow.add(veh_type="human",
           edge="right_south",
           probability=0.10)
inflow.add(veh_type="human",
           edge="right_north",
           probability=0.10)
inflow.add(veh_type="human",
           edge="left_north",
           probability=0.10)
inflow.add(veh_type="human",
           edge="left_south",
           probability=0.10)
inflow.add(veh_type="human",
           edge="left_west",
           probability=0.10)

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
    exp_tag="second_exp",
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
N_GPUS = 0
# number of rollouts per training iteration
N_ROLLOUTS = 1

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


alg_run = "DQN"

BATCH_SIZE = HORIZON * N_ROLLOUTS

agent_cls = get_agent_class(alg_run)
config = agent_cls._default_config.copy()
config["lr"] = tune.grid_search([5e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
config["num_workers"] = 0  # number of parallel workers
config["num_gpus"] = 0
config["train_batch_size"] = 32  # batch size
config["sample_batch_size"] = 4  # batch size
config["gamma"] = 0.999  # discount rate
config["model"].update({"fcnet_hiddens": tune.grid_search([[256], [128, 128], [128], [256, 256], [64]])})  # size of hidden layers in network
config["log_level"] = "DEBUG"
config["horizon"] = HORIZON  # rollout horizon
config["timesteps_per_iteration"] = BATCH_SIZE

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

exp = Experiment(flow_params["exp_tag"], **{
        "run": alg_run,
        "config": {
            **config
        },
        "checkpoint_freq": 5,  # number of iterations between checkpoints
        "checkpoint_at_end": True,  # generate a checkpoint at the end
        "max_failures": 5,
        "stop": {  # stopping conditions
            "training_iteration": 100,  # number of iterations to stop after
        },
        "num_samples": 1, 
        "local_dir": "/tmp/ray_results"
        })


trials = run_experiments(exp)




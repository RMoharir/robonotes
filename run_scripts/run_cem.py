"""
Runs the random agent.

Example usage:
    python ./run_scripts/run_ppo.py --exp_name ppo --total_timesteps 2000000 --save_model
    python ./run_scripts/run_ppo.py --exp_name ppo --load_model_path run_logs/ppo_09-11-2022_15-07-35/ppo.zip --save_midi
"""
import os
import time

from env import RoboNotesEnv
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import Model, CategoricalMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.cem import CEM, CEM_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env

import tqdm

MIDI_SAVEDIR = "./samples/cem/"

# Define the model (categorical model) for the CEM agent using mixin
# - Policy: takes as input the environment's observation/state and returns an action
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.linear_layer_1 = nn.Linear(self.num_observations, 64)
        self.linear_layer_2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, self.num_actions)

    def compute(self, states, taken_actions, role):
        x = F.relu(self.linear_layer_1(states))
        x = F.relu(self.linear_layer_2(x))
        return self.output_layer(x)

def sample_test_trajectory(trainer, env, evaluation_timesteps=10000):
        """Evaluate the agents sequentially
        This method executes the following steps in loop:
        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """

        # reset env
        states = trainer.env.reset()

        for timestep in tqdm.tqdm(range(trainer.initial_timestep, evaluation_timesteps)):

            # compute actions
            with torch.no_grad():
                actions, _, _ = trainer.agents.act(states, timestep=timestep, timesteps=trainer.timesteps)

            # step the environments
            next_states, rewards, dones, infos = trainer.env.step(actions)

            with torch.no_grad():
                # write data to TensorBoard
                super(type(trainer.agents), trainer.agents).record_transition(states=states,
                                                                        actions=actions,
                                                                        rewards=rewards,
                                                                        next_states=next_states,
                                                                        dones=dones,
                                                                        infos=infos,
                                                                        timestep=timestep,
                                                                        timesteps=trainer.timesteps)
                super(type(trainer.agents), trainer.agents).post_interaction(timestep=timestep, timesteps=trainer.timesteps)

                # reset environments
                if dones.any():
                    env.render()
                    states = trainer.env.reset()
                else:
                    states.copy_(next_states)

        # close the environment
        trainer.env.close()

def run_cem(args, log_dir=None, exp_name=None):
    midi_savedir = MIDI_SAVEDIR if args.save_midi else None
    
    env = RoboNotesEnv(max_trajectory_len=args.max_trajectory_len, midi_savedir=midi_savedir)
    env = wrap_env(env)

    device = env.device

    # Instantiate a RandomMemory (without replacement) as experience replay memory
    memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=device, replacement=False)

    # Instantiate the agent's model (function approximator).
    # CEM requires 1 model, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.cem.html#spaces-and-models
    models_cem = {}
    models_cem["policy"] = Policy(env.observation_space, env.action_space, device)

    # Initialize the models' parameters (weights and biases) using a Gaussian distribution
    for model in models_cem.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    # Configure and instantiate the agent.
    # Only modify some of the default configuration, visit its documentation to see all the options
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.cem.html#configuration-and-hyperparameters
    cfg_cem = CEM_DEFAULT_CONFIG.copy()
    cfg_cem["rollouts"] = args.rollouts
    cfg_cem["learning_rate"] = args.learning_rate
    cfg_cem["random_timesteps"] = args.random_timesteps
    cfg_cem["learning_starts"] = args.learning_starts
    # logging to TensorBoard and write checkpoints each 1000 and 5000 timesteps respectively
    cfg_cem["experiment"]["directory"] = log_dir 
    cfg_cem["experiment"]["experiment_name"] = exp_name 
    cfg_cem["experiment"]["write_interval"] = args.log_interval
    cfg_cem["experiment"]["checkpoint_interval"] = args.checkpoint_interval

    agent_cem = CEM(models=models_cem, 
                    memory=memory, 
                    cfg=cfg_cem, 
                    observation_space=env.observation_space, 
                    action_space=env.action_space,
                    device=device)


    # Configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": args.timesteps, "headless": True}
    trainer = SequentialTrainer(env=env, agents=[agent_cem], cfg=cfg_trainer)

    # start training
    trainer.train()

    print("Finished training.")


def eval_cem(args, log_dir=None, exp_name=None, eval_exp_name=None) :
    midi_savedir = MIDI_SAVEDIR if args.save_midi else None
    
    env = RoboNotesEnv(max_trajectory_len=args.max_trajectory_len, midi_savedir=midi_savedir)
    env = wrap_env(env)

    device = env.device

    # Instantiate a RandomMemory (without replacement) as experience replay memory
    memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=device, replacement=False)

    # Instantiate the agent's model (function approximator).
    # CEM requires 1 model, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.cem.html#spaces-and-models
    models_cem = {}
    models_cem["policy"] = Policy(env.observation_space, env.action_space, device)

    # Initialize the models' parameters (weights and biases) using a Gaussian distribution
    for model in models_cem.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    # Configure and instantiate the agent.
    # Only modify some of the default configuration, visit its documentation to see all the options
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.cem.html#configuration-and-hyperparameters
    cfg_cem = CEM_DEFAULT_CONFIG.copy()
    cfg_cem["rollouts"] = args.rollouts
    cfg_cem["learning_rate"] = args.learning_rate
    cfg_cem["random_timesteps"] = args.random_timesteps
    cfg_cem["learning_starts"] = args.learning_starts
    # logging to TensorBoard and write checkpoints each 1000 and 5000 timesteps respectively
    cfg_cem["experiment"]["directory"] = log_dir 
    cfg_cem["experiment"]["experiment_name"] = eval_exp_name 
    cfg_cem["experiment"]["write_interval"] = args.log_interval
    cfg_cem["experiment"]["checkpoint_interval"] = args.checkpoint_interval

    agent_cem = CEM(models=models_cem, 
                    memory=memory, 
                    cfg=cfg_cem, 
                    observation_space=env.observation_space, 
                    action_space=env.action_space,
                    device=device)


    # Configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": args.timesteps, "headless": True}
    trainer = SequentialTrainer(env=env, agents=[agent_cem], cfg=cfg_trainer)

    agent_cem.load(log_dir + "/" + exp_name + "/checkpoints/best_agent.pt")

    sample_test_trajectory(trainer, env)

    print("Finished evaluating.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="name the experiment, for log dir")

    # train parameters
    parser.add_argument("--timesteps", type=int, default=100000, required=False,
                        help="number of timesteps (env steps) to train")
    parser.add_argument("--rollouts", type=float, default=16, required=False,
                        help="number of rollouts before updating")
    parser.add_argument("--percentile", type=float, default=0.70, required=False,
                        help="percentile to compute the reward bound [0, 1]")  

    parser.add_argument("--discount_factor", type=float, default=0.99, required=False,
                        help="discount factor (gamma)")  

    parser.add_argument("--learning_rate", type=float, default=1e-2, required=False,
                        help="learning rate")
    parser.add_argument("--learning_rate_scheduler", type=float, default=None, required=False,
                        help="learning rate scheduler class (see torch.optim.lr_scheduler)")
    parser.add_argument("--learning_rate_scheduler_kwargs", type=float, default={}, required=False,
                        help="learning rate scheduler's kwargs")

    parser.add_argument("--state_preprocessor", type=int, default=None, required=False,
                        help="state preprocessor class (see skrl.resources.preprocessors)")
    parser.add_argument("--state_preprocessor_kwargs", type=int, default={}, required=False,
                        help="state preprocessor's kwargs")

    parser.add_argument("--random_timesteps", type=int, default=0, required=False,
                        help="random exploration steps")
    parser.add_argument("--learning_starts", type=float, default=0, required=False,
                        help="learning starts after this many steps")

    parser.add_argument("--rewards_shaper", type=int, default=None, required=False,
                        help="rewards shaping function: Callable(reward, timestep, timesteps) -> reward")

    # env parameters
    parser.add_argument("--max_trajectory_len", type=int, default=20, required=False,
                        help="Length of music composition (number of beats)")

    # other parameters
    parser.add_argument("--log_interval", type=int, default=1, required=False,
                        help="number of timesteps before logging to tensorboard")
    parser.add_argument("--checkpoint_interval", type=int, default=5000, required=False,
                        help="number of timesteps before writing checkpoint")
    parser.add_argument("--test_interval", type=int, default=100000, required=False,
                        help="number of timesteps before testing")

    parser.add_argument("--save_midi", action="store_true", help="Whether or not to save MIDI output")
    parser.add_argument("--save_model", action="store_true", help="Whether or not to save trained cem model")
    parser.add_argument("--load_model_path", type=str,
                        help="If provided, path of the cem model to load.", required=False)
    args = parser.parse_args()
    print("\nARGS: ", args, "\n")

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../run_logs')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    exp_name = args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")

    print("\nLOGGING TO: ", data_path, "\n")

    run_cem(args, log_dir=data_path, exp_name=exp_name)

    eval_exp_name = 'eval_' + exp_name

    eval_cem(args, log_dir=data_path, exp_name=exp_name, eval_exp_name=eval_exp_name)
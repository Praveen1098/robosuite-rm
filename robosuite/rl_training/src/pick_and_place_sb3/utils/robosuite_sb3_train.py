# Training file for connecting robosuite and Stable Baselines3 algorithms.

import json
from datetime import datetime
from typing import Any, Union, Tuple

import gymnasium
import numpy as np
import robosuite
from robosuite import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from stable_baselines3.common.logger import configure


def make_vec_env_(
        env_kwargs: dict[str, Any] | None = None,
        n_envs: int = 1,
        multiproc: bool = False,
        seed: int | None = None
        ) -> VecEnv:
    """
    Make vectorized robosuite environments
    Args:
        env_kwargs: Robosuite environment arguments to pass to the environment constructor.
        n_envs: The number of vectorized environments you want to run in parallel.
        multiproc: Activates use of 'SubprocVecEnv' instead of 'DummyVecEnv'.
        seed: The initial seed for the random number generator.
    """

    vec_env_cls = SubprocVecEnv if multiproc and n_envs > 1 else None
    def make_env():
        controller_name = env_kwargs.pop("controller")
        arm_controller_config = load_part_controller_config(default_controller=controller_name)
        robot = env_kwargs["robots"][0]
        controller_configs = refactor_composite_controller_config(
            arm_controller_config, robot, ["right"]
        )
        env = robosuite.make(**env_kwargs,
                             controller_configs=controller_configs)
        env = GymWrapper(env)
        return env
    
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=vec_env_cls)
    env.seed(seed)
    return env


def make_algorithm_off_policy(
        env: gymnasium.Env,
        algorithm: str,
        buffer_size: int,
        learning_rate,
        tau,
        gamma,
        train_freq: Union[int, Tuple[int, str]],
        ent_coef: Union[str, float],
        batch_size: int,
        target_update_interval,
        verbose) -> BaseAlgorithm:

    """
    Initialize off policy algorithm model

    :params env: Gymnasium environment.
    :params algorithm: name for the algorithm, "SAC", "TD3".
    :params buffer_size: size for the replay buffer.

    :return: stable baselines 3 BaseAlgorithm.
    """

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
    if algorithm == "SAC":
        model = SAC("MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    buffer_size=int(buffer_size),
                    learning_starts=100,
                    batch_size=batch_size,
                    action_noise=action_noise,
                    tau=tau,
                    gamma=gamma,
                    ent_coef=ent_coef,
                    target_update_interval=target_update_interval,
                    verbose=verbose,
                    device="auto")
    elif algorithm == "TD3":
        model = TD3("MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    buffer_size=int(buffer_size),
                    action_noise=action_noise,
                    tau=tau,
                    policy_delay=2,
                    verbose=verbose)
    else:
        raise ValueError("Not Supported Algorithm")

    return model


def make_algorithm_on_policy(
        env: gymnasium.Env,
        algorithm: str,
        learning_rate,
        n_steps,
        batch_size: int,
        gamma,
        ent_coef: Union[str, float],
        vf_coef,
        max_grad_norm,
        gae_lambda,
        clip_range,
        verbose) -> BaseAlgorithm:

    """
    Initialize algorithm model for on-policy

    :params env: Gym environment.
    :params algorithm: name for the algorithm, "PPO".
    :params buffer_size: size for the replay buffer.

    :return: stable baselines 3 BaseAlgorithm.
    """

    if algorithm == "PPO":
        model = PPO("MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    gamma=gamma,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    verbose=verbose)

    else:
        raise ValueError("Not Supported Algorithm")

    return model


def get_now():
    return datetime.now().replace(microsecond=0).isoformat().replace(":", "_")


class RobosuiteSb3Trainer:
    """
    Base Class for using Stable Baselines 3 for robosuite environments
    Stable Baselines 3  https://stable-baselines3.readthedocs.io/en/master/
    Robosuite           https://robosuite.ai/

    :params logdir: the directory for logging.
    :params algorithm: the name of the algorithm for training.
    :params replay_buffer_size: size for the replay buffer.
    :params num_steps: total number of steps for training.
    :params seed: random seed.
    :params no_reward_shaping: Set to True to not use reward shaping by default.
    """
    def __init__(
            self,
            logdir: str = "tmp",
            algorithm: str = "SAC",
            replay_buffer_size: int = 1_000_000,
            num_steps: int = 2_000_000,
            seed: int = 69,
            stopping_criteria: str = "None",
            has_stop_on_reward_threshold: bool = False,
            has_stop_on_no_model_improvement: bool = False,
            has_checkpoint_callback: bool = False,
            eval_freq: int = 1000,
            n_envs: int = 8,
            multiproc: bool = False,
            n_eval_episodes: int = 50,
            reward_threshold: float = 1.0,
            algorithm_kwargs: dict = None,
            robosuite_env_kwargs: dict = None,
    ):

        self.model = None
        self.model_path = None
        self.env = None
        self.logdir = logdir
        self.algorithm = algorithm
        self.replay_buffer_size = replay_buffer_size
        self.num_steps = num_steps
        self.seed = seed
        self.stopping_criteria = stopping_criteria
        self.has_stop_on_reward_threshold = has_stop_on_reward_threshold
        self.has_stop_on_no_model_improvement = has_stop_on_no_model_improvement
        self.has_checkpoint_callback = has_checkpoint_callback
        self.eval_freq = eval_freq
        self.n_envs = n_envs
        self.multiproc = multiproc
        self.n_eval_episodes = n_eval_episodes
        self.reward_threshold = reward_threshold
        self.algorithm_kwargs = algorithm_kwargs
        self.robosuite_env_kwargs = robosuite_env_kwargs

    def make(self):
        """
        Function to initialize environment and model for experiments.
        Must call before training.
        """
        self.env = make_vec_env_(self.robosuite_env_kwargs, self.n_envs, self.multiproc, self.seed)
        self.model_path =  f"{self.logdir}/{experiment_name}/SEED{self.seed}/{get_now()}"

    def train(self):
        """
        Train on environments ("Lift", "Door", etc.) with
        algorithms ("PPO", "SAC", "TD3", etc).
        Model checkpoints and weights are saved in "<logdir>/<experiment_name>/<algorithm>/SEED<seed>/<current Date and time>"
        """
        if self.algorithm == "PPO":
            self.model = make_algorithm_on_policy(env=self.env,
                                                  algorithm=self.algorithm,
                                                  learning_rate=self.algorithm_kwargs['learning_rate'],
                                                  n_steps=self.algorithm_kwargs['n_steps'],
                                                  batch_size=self.algorithm_kwargs['batch_size'],
                                                  gamma=self.algorithm_kwargs['gamma'],
                                                  ent_coef=self.algorithm_kwargs['ent_coef'],
                                                  vf_coef=self.algorithm_kwargs['vf_coef'],
                                                  max_grad_norm=self.algorithm_kwargs['max_grad_norm'],
                                                  gae_lambda=self.algorithm_kwargs['gae_lambda'],
                                                  clip_range=self.algorithm_kwargs['clip_range'],
                                                  verbose=self.algorithm_kwargs['verbose'])
        elif self.algorithm == "SAC" or self.algorithm == "TD3":
            self.model = make_algorithm_off_policy(env=self.env,
                                                   algorithm=self.algorithm,
                                                   buffer_size=self.replay_buffer_size,
                                                   learning_rate=self.algorithm_kwargs['learning_rate'], 
                                                   tau=self.algorithm_kwargs['tau'],
                                                   gamma=self.algorithm_kwargs['gamma'], 
                                                   train_freq=self.algorithm_kwargs['train_freq'],
                                                   ent_coef=self.algorithm_kwargs['ent_coef'],
                                                   batch_size=self.algorithm_kwargs['batch_size'],
                                                   target_update_interval=self.algorithm_kwargs['target_update_interval'],
                                                   verbose=self.algorithm_kwargs['verbose'])
        else:
            raise ValueError("Not Supported Algorithm")

        logger = configure(self.model_path, ["stdout", "csv", "tensorboard"])
        # based on variant argument stop training on reward threshold or no model improvement
        callback_on_best = None
        stop_train_callback = None
        if self.has_stop_on_reward_threshold:
            callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=self.reward_threshold, verbose=1)
        if self.has_stop_on_no_model_improvement:
            stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=25,
                                                                   verbose=1)

        eval_callback = EvalCallback(eval_env=self.env,
                                    n_eval_episodes=self.n_eval_episodes,
                                    callback_on_new_best=callback_on_best,
                                    callback_after_eval=stop_train_callback,
                                    log_path=self.model_path,
                                    best_model_save_path=self.model_path,
                                    eval_freq=max(self.eval_freq // self.n_envs, 1),
                                    deterministic=True, render=False)
        checkpoint_callback = CheckpointCallback(save_freq=100000,
                                                 save_path=self.model_path,
                                                 name_prefix="rl_model")
        if self.has_checkpoint_callback:
            callback_list = [checkpoint_callback, eval_callback]
        else:
            callback_list = [eval_callback]

        self.model.set_random_seed(seed=self.seed)
        self.model.set_logger(logger)

        # Set model parameters for fine-tuning pre-trained policy
        # self.model.set_parameters('f"{self.previouslogdir}/SACparameters.zip')

        self.model.learn(total_timesteps=self.num_steps, callback=callback_list)
        self.model.save(f"{self.model_path}parameters")


def main(variant_dic):

    learn = RobosuiteSb3Trainer(**variant_dic)
    learn.make()
    learn.train()


if __name__ == "__main__":
    # load json file
    f = open('Path To pick_and_place_milk_1_item.json')
    variant = json.load(f)

    # Name to designate experimental differences not included in variant file
    experiment_name = "pick_and_place_milk_1_item"

    # Notify user we're starting run
    print('\n\n')
    print('------------- Running Experiment {} in {} Environment using {} Algorithm --------------'.format(
          experiment_name, variant['robosuite_env_kwargs']['env_name'], variant['algorithm']))

    print('\n\n')

    # Execute run
    main(variant)
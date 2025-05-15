import json
import os
import sys
from csv import DictWriter

import cv2
import imageio
import numpy as np
import robosuite as suite
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import SAC, PPO


class RobosuiteSb3Rollout:
    """
    Base Class for running rollout from experiments using Stable Baselines 3 for robosuite environments
    Stable Baselines 3  https://stable-baselines3.readthedocs.io/en/master/
    Robosuite           https://robosuite.ai/

    :params logdir: the directory for logging.
    :params reuse_logdir: the directory for teacher policy in Policy Reuse.
    :params robot: the name for robosuite robots, like "Panda", "Sawyer", "LBR IIWA 7",
        "Jaco", "Kinova Gen3", "UR5e".
    :params env_name: the name for the environment.
    :params algorithm: the name of the algorithm for training.
    :params buffersize: size for the replay buffer.
    :params num_steps: total number of steps for training.
    :params mu: parameter mu in Policy Reuse.
    :params max_reuse_steps: maximum number of steps for per episode in Policy Reuse.
    :params seed: random seed.
    :params no_reward_shaping: Set to True to not use reward shaping by default.
    """
    def __init__(self,
                 logdir: str = "default_log",
                 curriculum: str = "default",
                 algorithm: str = "SAC",
                 robosuite_env_kwargs: dict = None, 
                 camera_height=512, 
                 camera_width=512,
                 **kwargs):
    
        self.logdir = logdir
        self.algorithm = algorithm
        self.curriculum = curriculum
        self.env_name = robosuite_env_kwargs['env_name']
        self.robots = robosuite_env_kwargs['robots']
        self.robot = robosuite_env_kwargs['robots'][0]
        self.controller = robosuite_env_kwargs.pop("controller")
        self.camera = "frontview"
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.ignore_done = robosuite_env_kwargs['ignore_done']
        self.horizon = robosuite_env_kwargs['horizon']
        self.control_freq = robosuite_env_kwargs['control_freq']
        self.use_camera_obs = True
        self.robosuite_env_kwargs = robosuite_env_kwargs

    # Function to create a dictionary of obs_keys, obd pairs
    def make_keys(self):
        keys = []
        keys += ["object-state"]
        # Iterate over all robots to add to state
        for idx in range(len(self.robots)):
            keys += ["robot{}_proprio-state".format(idx)]
        return keys

    # Helper function to flatten obs dictionary to np array
    @staticmethod
    def flatten_obs(obs_dict, keys):
        ob_lst = []
        for key in keys:
            if key in obs_dict:
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def generate_rollout_video(
            self,
            model_path: str,
            episodes: int = 10,
    ):

        """
        Generate video of policy rollouts of a manipulation task.
        : params model_path: directory to where the model is saved.
        : params episodes: number of episodes to run for video generation of policy rollouts.
        """
        hard_reset = self.robosuite_env_kwargs.pop('hard_reset')
        # horizon = self.robosuite_env_kwargs.pop('horizon')
        self.robosuite_env_kwargs['horizon'] = 500

        # load the desired controller
        arm_controller_config = load_part_controller_config(default_controller=self.controller)
        controller_configs = refactor_composite_controller_config(
            arm_controller_config, self.robot, ["right"]
        )    
        # Pass in camera arguments
        self.robosuite_env_kwargs['has_offscreen_renderer'] = True
        self.robosuite_env_kwargs['use_camera_obs'] = True
        self.robosuite_env_kwargs['camera_names'] = self.camera
        self.robosuite_env_kwargs['camera_heights'] = self.camera_height
        self.robosuite_env_kwargs['camera_widths'] = self.camera_width
        self.robosuite_env_kwargs['hard_reset'] = True

        env = suite.make(
            **self.robosuite_env_kwargs,
            controller_configs=controller_configs,
        )

        # Grab name of this rollout combo
        video_name = "{}{}".format(model_path, self.curriculum)
        # Calculate appropriate fps
        fps = int(self.control_freq)
        # Define video writer
        video_writer = imageio.get_writer("{}.mp4".format(video_name), fps=fps)

        next_eps_frame = imageio.v2.imread('Path to episode_slide_resized.jpg')

        model = None

        if self.algorithm == "SAC":
            model = SAC.load(f"{model_path}best_model.zip")
        elif self.algorithm == "PPO":
            model = PPO.load(f"{model_path}best_model.zip")

        for eps in range(episodes):

            observations = env.reset()
            keys = ['object-state', 'robot0_proprio-state']
            observations = self.flatten_obs(observations, keys)
            print(f" Recording Episode number {eps+1}")

            while True:
                action, _states = model.predict(observations, deterministic=True)
                next_observations, reward, done, info = env.step(action)
                print("Reward: ", reward)
                print("Done: ", done)

                frame = next_observations[self.camera + "_image"]
                frame = cv2.flip(frame, 0)
                video_writer.append_data(frame)

                if done:
                    break
                observations = self.flatten_obs(next_observations, keys)

            for i in range(10):
                video_writer.append_data(next_eps_frame)

        video_writer.close()

    def render_rollout_onscreen(
            self,
            model_path: str,
            episodes: int = 10,
    ):

        """
        Generate onscreen render of policy rollouts of a manipulation task.
        : params model_path: directory to where the model is saved.
        : params episodes: number of episodes to run for video generation of policy rollouts.
        """
        hard_reset = self.robosuite_env_kwargs.pop('hard_reset')
        horizon = self.robosuite_env_kwargs.pop('horizon')
        # self.robosuite_env_kwargs['horizon'] = 200

        # load the desired controller
        arm_controller_config = load_part_controller_config(default_controller=self.controller)
        controller_configs = refactor_composite_controller_config(
            arm_controller_config, self.robot, ["right"]
        )    

        # Pass in camera arguments
        self.robosuite_env_kwargs['has_offscreen_renderer'] = False
        self.robosuite_env_kwargs['use_camera_obs'] = False
        self.robosuite_env_kwargs['hard_reset'] = True
        self.robosuite_env_kwargs['has_renderer'] = True
        self.robosuite_env_kwargs['render_camera'] = self.camera

        env = suite.make(
            **self.robosuite_env_kwargs,
            controller_configs=controller_configs,
        )

        model = None

        if self.algorithm == "SAC":
            model = SAC.load(f"{model_path}best_model.zip")
        elif self.algorithm == "PPO":
            model = PPO.load(f"{model_path}best_model.zip")

        for eps in range(episodes):

            observations = env.reset()
            keys = ['object-state', 'robot0_proprio-state']
            observations = self.flatten_obs(observations, keys)
            print(f" Running Episode number {eps+1}")

            try:
                while True:
                    action, _states = model.predict(observations, deterministic=True)
                    next_observations, reward, done, info = env.step(action)
                    print("Action: ", action)
                    print("Reward: ", reward)
                    print("Done: ", done)
                    # print("Observations: ", next_observations)

                    env.render()

                    if done:
                        break

                    observations = self.flatten_obs(next_observations, keys)
            except KeyboardInterrupt:
                sys.exit("\nStopping Rollout and Shutting Down.")


def rollout(experiment_dic, model_path, has_record_rollout, episodes):
    run_rollout = RobosuiteSb3Rollout(**experiment_dic)
    if has_record_rollout:
        run_rollout.generate_rollout_video(model_path, episodes)
    else:
        run_rollout.render_rollout_onscreen(model_path, episodes)


if __name__ == "__main__":
    # load path to variant file
    f = open('Path to pick_and_place_milk_1_item.json')
    experiment = json.load(f)

    # Notify user we're starting run
    print('\n\n')
    print('------------- Running {} Environment using the best model policy --------------'.format(experiment['robosuite_env_kwargs']['env_name']))

    print('\n\n')

    # best model path
    best_model_path  = 'Path to /logs/tmp/SAC/pick_and_place_milk_1_item/SEED17/2025-05-02T11_52_15/'

    # set recording a rollout or rendering a rollout
    has_record_rollout = False

    # Set the number of episodes
    number_of_episodes = 10

    rollout(experiment, best_model_path, has_record_rollout, number_of_episodes)

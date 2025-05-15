import numpy as np
from collections import OrderedDict

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.utils.observables import Observable, sensor

'''
class RewardMachine:
    """
    Finite-state reward machine for structured rewards (6 states per object):
    0=init,1=reaching,2=grasping,3=lifting,4=hovering,5=placed
    """
    def __init__(self, num_objects):
        # State and object counts
        self.num_states = 6
        self.num_objects = num_objects
        # History must exist before reset
        self.history = []
        # Initialize states and completion flags
        self.reset()
        # Transition rewards matrix (from state i to j)
        self.transition_rewards = [
            [0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0] * 6,
        ]
        # State rewards for being in each state
        self.state_rewards = [0.0, 0.1, 0.35, 0.5, 0.7, 1.0]

    def reset(self):
        # Initialize or reset states and completion flags
        self.states = [0] * self.num_objects
        self.completed = [False] * self.num_objects
        # Clear history
        self.history = []

    def transition(self, idx, next_state):
        # Bounds check
        if idx < 0 or idx >= self.num_objects:
            return 0.0
        curr = self.states[idx]
        # Only allow forward progress
        if next_state <= curr:
            return 0.0
        # Record transition
        self.history.append((idx, curr, next_state))
        # Update state and completion
        self.states[idx] = next_state
        if next_state == self.num_states - 1:
            self.completed[idx] = True
        return self.transition_rewards[curr][next_state]

    def state_reward(self, idx):
        # Bounds check
        if idx < 0 or idx >= self.num_objects:
            return 0.0
        return self.state_rewards[self.states[idx]]

    def is_completed(self, idx):
        if idx < 0 or idx >= self.num_objects:
            return False
        return self.completed[idx]

    def get_state(self, idx):
        if idx < 0 or idx >= self.num_objects:
            return 0
        return self.states[idx]


class PickPlaceRewardMachine(PickPlace):
    """PickPlace with an integrated reward machine for structured shaping."""

    def __init__(
        self,
        robots,
        reward_scale=1.0,
        reward_shaping=False,
        use_rm_transitions=True,
        use_rm_states=True,
        **kwargs
    ):
        # Placeholder before super init
        self.reward_machine = None
        super().__init__(
            robots=robots,
            reward_scale=reward_scale,
            reward_shaping=reward_shaping,
            **kwargs
        )
        # Determine number of objects
        self.num_objects = 1 if self.single_object_mode in {1, 2} else len(self.objects)
        # Initialize reward machine
        self.reward_machine = RewardMachine(num_objects=self.num_objects)
        self.use_rm_transitions = use_rm_transitions
        self.use_rm_states = use_rm_states
        # Transition thresholds aligned with staged_rewards()
        self.reach_th = 0.05
        self.grasp_th = 0.01
        self.lift_th = 0.25
        self.hover_th = 0.15

        # Print observation keys for debugging
        print("Observation space keys:", list(self.observation_spec().keys()))

    def _reset_internal(self):
        super()._reset_internal()
        # Reset reward machine state if it exists
        if self.reward_machine is not None:
            self.reward_machine.reset()

    def reward(self, action=None):
        # Update success flags
        self._check_success()
        # Get staged shaping signals
        r_reach, r_grasp, r_lift, r_hover = self.staged_rewards()
        total = 0.0
        # Iterate through objects
        for i in range(self.num_objects):
            if self.reward_machine.is_completed(i):
                # Optionally add final state reward
                if self.use_rm_states:
                    total += self.reward_machine.state_reward(i)
                continue
            curr = self.reward_machine.get_state(i)
            # Determine next state
            next_state = curr
            if self.objects_in_bins[i]:
                next_state = 5
            elif r_hover > self.hover_th and curr >= 3:
                next_state = 4
            elif r_lift > self.lift_th and curr >= 2:
                next_state = 3
            elif r_grasp > self.grasp_th and curr >= 1:
                next_state = 2
            elif r_reach > self.reach_th and curr == 0:
                next_state = 1
            # Apply transition reward
            if next_state > curr and self.use_rm_transitions:
                total += self.reward_machine.transition(i, next_state)
            # Apply state reward
            if self.use_rm_states:
                total += self.reward_machine.state_reward(i)
        # Scale and normalize
        if self.reward_scale is not None:
            total *= self.reward_scale
            if self.single_object_mode == 0:
                total /= self.num_objects
        return total

    def _get_observation(self):
        obs = super()._get_observation()
        # Append RM states when object observations are enabled
        if self.use_object_obs and self.reward_machine is not None:
            states = np.array([self.reward_machine.get_state(i) for i in range(self.num_objects)], dtype=np.int32)
            obs["rm_states"] = states
        return obs

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs and self.reward_machine is not None:
            @sensor(modality="object")
            def rm_states_sensor(obs_cache):
                return np.array(self.reward_machine.states, dtype=np.int32)
            observables["rm_states"] = Observable(
                name="rm_states",
                sensor=rm_states_sensor,
                sampling_rate=self.control_freq,
                enabled=True,
                active=True,
            )
        return observables
'''

class RewardMachine:
    """
    Finite-state reward machine for structured rewards (6 states per object):
    0=init,1=reaching,2=grasping,3=lifting,4=hovering,5=placed
    """
    def __init__(self, num_objects):
        self.num_states = 6
        self.num_objects = num_objects
        self.reset()
        self.transition_rewards = [
            [0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0] * 6,
        ]
        self.state_rewards = [0.0, 0.1, 0.35, 0.5, 0.7, 1.0]

    def reset(self):
        self.states = [0] * self.num_objects
        self.completed = [False] * self.num_objects
        self.history = []

    def transition(self, idx, next_state):
        if idx < 0 or idx >= self.num_objects:
            return 0.0
        curr = self.states[idx]
        if next_state <= curr:
            return 0.0
        self.history.append((idx, curr, next_state))
        self.states[idx] = next_state
        if next_state == self.num_states - 1:
            self.completed[idx] = True
        return self.transition_rewards[curr][next_state]

    def state_reward(self, idx):
        if idx < 0 or idx >= self.num_objects:
            return 0.0
        return self.state_rewards[self.states[idx]]

    def is_completed(self, idx):
        return 0 <= idx < self.num_objects and self.completed[idx]

    def get_state(self, idx):
        return self.states[idx] if 0 <= idx < self.num_objects else 0


class PickPlaceRewardMachine(PickPlace):
    """PickPlace with an integrated reward machine for structured shaping."""

    def __init__(
        self,
        robots,
        reward_scale=1.0,
        reward_shaping=False,
        use_rm_transitions=True,
        use_rm_states=True,
        **kwargs
    ):
        self.use_rm_transitions = use_rm_transitions
        self.use_rm_states = use_rm_states
        self.reach_th = 0.05
        self.grasp_th = 0.01
        self.lift_th = 0.25
        self.hover_th = 0.15

        # Temporary placeholder
        self.reward_machine = RewardMachine(num_objects=4)

        super().__init__(
            robots=robots,
            reward_scale=reward_scale,
            reward_shaping=reward_shaping,
            **kwargs
        )

        self.num_objects = 1 if self.single_object_mode in {1, 2} else len(self.objects)
        self.reward_machine = RewardMachine(num_objects=self.num_objects)

        # Print for debugging
        print("Observation space keys:", list(self.observation_spec().keys()))

    def _reset_internal(self):
        #self._print_flattened_observation_keys()
        super()._reset_internal()
        if self.reward_machine:
            self.reward_machine.reset()

    def reward(self, action=None):
        self._check_success()
        r_reach, r_grasp, r_lift, r_hover = self.staged_rewards()
        total = 0.0
        for i in range(self.num_objects):
            if self.reward_machine.is_completed(i):
                if self.use_rm_states:
                    total += self.reward_machine.state_reward(i)
                continue
            curr = self.reward_machine.get_state(i)
            next_state = curr
            if self.objects_in_bins[i]:
                next_state = 5
            elif r_hover > self.hover_th and curr >= 3:
                next_state = 4
            elif r_lift > self.lift_th and curr >= 2:
                next_state = 3
            elif r_grasp > self.grasp_th and curr >= 1:
                next_state = 2
            elif r_reach > self.reach_th and curr == 0:
                next_state = 1
            if next_state > curr and self.use_rm_transitions:
                total += self.reward_machine.transition(i, next_state)
            if self.use_rm_states:
                total += self.reward_machine.state_reward(i)
        if self.reward_scale is not None:
            total *= self.reward_scale
            if self.single_object_mode == 0:
                total /= self.num_objects
        return total

    #def _print_flattened_observation_keys(self):
    #    flat_obs = self._flatten_obs(self._get_observation())
    #    if isinstance(flat_obs, dict):
    #        print("[DEBUG] Flattened observation keys:", list(flat_obs.keys()))
    #    else:
    #        print("[DEBUG] Flattened observation shape:", flat_obs.shape)

    #def _get_observation(self):
    #    obs = super()._get_observation()
    #    if self.use_object_obs and self.reward_machine:
    #        obs["rm_states"] = np.array(
    #            [self.reward_machine.get_state(i) for i in range(self.num_objects)], dtype=np.int32
    #        )
    #    return obs
"""
Reward Machines for Robosuite PickPlace Task

This module implements reward machines for the PickPlace task in the robosuite
simulation framework. It creates a wrapper around the robosuite environment
that uses a reward machine to determine rewards.

Based on the reward machines framework: https://github.com/RodrigoToroIcarte/reward_machines
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import robosuite as suite
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.wrappers import GymWrapper
from typing import Dict, List, Tuple, Optional, Set, Union, Any


class RewardMachine:
    """
    A reward machine implementation for robosuite tasks.
    
    A reward machine is a finite state machine that defines the rewards for an agent
    based on its current state and transitions.
    """
    
    def __init__(self, states: List[int], initial_state: int, terminal_states: List[int]):
        """
        Initialize a reward machine.
        
        Args:
            states: List of states in the reward machine
            initial_state: Initial state of the reward machine
            terminal_states: List of terminal states
        """
        self.states = states
        self.initial_state = initial_state
        self.current_state = initial_state
        self.terminal_states = terminal_states
        self.transitions = {}  # (state, event) -> (next_state, reward)
        
    def add_transition(self, source: int, event: str, target: int, reward: float):
        """
        Add a transition to the reward machine.
        
        Args:
            source: Source state
            event: Event triggering the transition
            target: Target state
            reward: Reward for taking this transition
        """
        # Verify source and target states exist
        if source not in self.states:
            raise ValueError(f"Source state {source} is not in the defined states list")
        if target not in self.states:
            raise ValueError(f"Target state {target} is not in the defined states list")
            
        self.transitions[(source, event)] = (target, reward)
        
    def get_next_state(self, event: str) -> int:
        """
        Get the next state given an event.
        
        Args:
            event: Event triggering a transition
            
        Returns:
            Next state after transition
        """
        if (self.current_state, event) in self.transitions:
            next_state, _ = self.transitions[(self.current_state, event)]
            return next_state
        return self.current_state
        
    def get_reward(self, event: str) -> float:
        """
        Get the reward for a transition given an event.
        
        Args:
            event: Event triggering a transition
            
        Returns:
            Reward for this transition
        """
        if (self.current_state, event) in self.transitions:
            _, reward = self.transitions[(self.current_state, event)]
            return reward
        return 0.0
        
    def step(self, event: str) -> float:
        """
        Update the current state and return the reward for this transition.
        
        Args:
            event: Event triggering a transition
            
        Returns:
            Reward for this transition
        """
        reward = self.get_reward(event)
        self.current_state = self.get_next_state(event)
        return reward
        
    def reset(self):
        """Reset the reward machine to its initial state."""
        self.current_state = self.initial_state
        
    def is_terminal(self) -> bool:
        """Check if the current state is a terminal state."""
        return self.current_state in self.terminal_states


class PickPlaceEventDetector:
    """
    Event detector for the PickPlace task in robosuite.
    This class detects events that are used as inputs to the reward machine.
    Based on the official robosuite PickPlace implementation which contains
    bread, milk, cereal, and can objects.
    """
    
    def __init__(self, env):
        """
        Initialize the event detector.
        
        Args:
            env: Robosuite PickPlace environment
        """
        self.env = env
        self.object_grasped = False
        self.reached_target = False
        self.prev_object_pos = None
        
        # Standard PickPlace objects in robosuite
        self.default_objects = ["bread", "milk", "cereal", "can"]
        
        # Try to access the underlying robosuite environment
        self.pick_place_env = self._get_robosuite_env(env)
        
        # Determine object names based on environment configuration
        if self.pick_place_env is not None:
            if hasattr(self.pick_place_env, 'objects'):
                # Handle both dictionary and list type for objects
                if isinstance(self.pick_place_env.objects, dict):
                    self.object_names = list(self.pick_place_env.objects.keys())
                elif isinstance(self.pick_place_env.objects, list):
                    # For list objects, try to extract names if they have an 'name' attribute
                    try:
                        self.object_names = [obj.name for obj in self.pick_place_env.objects 
                                           if hasattr(obj, 'name')]
                    except AttributeError:
                        # Fallback to default if objects don't have name attribute
                        self.object_names = self.default_objects
                else:
                    self.object_names = self.default_objects
            else:
                self.object_names = self.default_objects
                
            if hasattr(self.pick_place_env, 'target_bins'):
                self.target_bins = self.pick_place_env.target_bins
            else:
                self.target_bins = None
        else:
            self.object_names = self.default_objects
            self.target_bins = None
        
        # Thresholds for event detection
        self.grasp_threshold = 0.01  # Distance threshold for grasping detection
        self.target_threshold = 0.05  # Distance threshold for target detection
    
    def _get_robosuite_env(self, env):
        """
        Safely extract the underlying robosuite environment from potential wrappers.
        
        Args:
            env: Potentially wrapped environment
            
        Returns:
            Robosuite environment or None if not found
        """
        # Try to navigate through potential wrapper layers
        current = env
        
        # Attempt to access the environment through common wrapper patterns
        for _ in range(10):  # Limit depth to avoid infinite loops
            if current is None:
                return None
                
            # Check if this is a robosuite PickPlace environment
            if isinstance(current, PickPlace):
                return current
                
            # Try different attribute patterns used by wrappers
            if hasattr(current, 'env'):
                current = current.env
            elif hasattr(current, '_env'):
                current = current._env
            elif hasattr(current, 'sim_env'):
                current = current.sim_env
            else:
                # No more recognized wrapper patterns
                break
                
        return None
        
    def detect_events(self, obs: Dict) -> List[str]:
        """
        Detect events based on the current observation.
        
        Args:
            obs: Current observation from the environment
            
        Returns:
            List of detected events
        """
        events = []
        
        # Extract relevant state information from observation
        object_pos = self._get_object_position(obs)
        gripper_pos = self._get_gripper_position(obs)
        target_pos = self._get_target_position(obs)
        gripper_state = self._get_gripper_state(obs)
        
        # Skip event detection if we couldn't extract necessary information
        if object_pos is None or gripper_pos is None or target_pos is None:
            return events
        
        # Store initial object position if this is the first step
        if self.prev_object_pos is None:
            self.prev_object_pos = object_pos
        
        # Check if object is grasped (based on gripper state and object position)
        is_grasped = self._is_object_grasped(gripper_state, gripper_pos, object_pos)
        
        # Generate events based on state changes
        if is_grasped and not self.object_grasped:
            events.append("object_grasped")
            self.object_grasped = True
        elif not is_grasped and self.object_grasped:
            events.append("object_released")
            self.object_grasped = False
            
        # Check if object is at target
        is_at_target = self._is_object_at_target(object_pos, target_pos)
        if is_at_target and not self.reached_target:
            events.append("reached_target")
            self.reached_target = True
        
        # Update previous object position
        self.prev_object_pos = object_pos
            
        return events
        
    def _get_gripper_state(self, obs: Dict) -> Optional[np.ndarray]:
        """
        Extract gripper state from observation.
        
        Args:
            obs: Current observation
            
        Returns:
            Gripper state information (typically gripper width) or None if not found
        """
        # In robosuite, gripper state is typically found in:
        # 1. Dict observation: 'robot0_gripper_qpos'
        # 2. Flattened observation: specific indices related to gripper
        
        if isinstance(obs, dict):
            if 'robot0_gripper_qpos' in obs:
                return obs['robot0_gripper_qpos']
        
        # If we have access to the actual environment, try to get it directly
        if self.pick_place_env is not None:
            try:
                robot = self.pick_place_env.robots[0]
                return robot.gripper.get_joint_positions()
            except (AttributeError, IndexError):
                pass
        
        # Default fallback for flattened observation
        # For most robosuite configs, gripper state is often at the end
        # This is an approximation and might need adjustment
        if isinstance(obs, np.ndarray) and len(obs) > 2:
            return obs[-2:]
            
        # If we couldn't extract gripper state, return None
        return None
        
    def _get_gripper_position(self, obs: Dict) -> Optional[np.ndarray]:
        """
        Extract gripper position from observation.
        
        Args:
            obs: Current observation
            
        Returns:
            3D position of gripper or None if not found
        """
        if isinstance(obs, dict):
            if 'robot0_eef_pos' in obs:
                return obs['robot0_eef_pos']
            
            # Alternative keys that might contain gripper position
            for key in ['gripper_pos', 'eef_pos', 'robot0_proprio-state']:
                if key in obs and len(obs[key]) >= 3:
                    return obs[key][:3]  # First 3 elements are typically position
        
        # If we have access to the environment, get it directly
        if self.pick_place_env is not None:
            try:
                robot = self.pick_place_env.robots[0]
                return robot.gripper.get_position()
            except (AttributeError, IndexError):
                pass
        
        # Default fallback for flattened observation
        # In robosuite, the gripper position is typically in the first few elements
        if isinstance(obs, np.ndarray) and len(obs) > 3:
            return obs[:3]
            
        return None
        
    def _get_object_position(self, obs: Dict) -> Optional[np.ndarray]:
        """
        Extract object position from observation based on robosuite's PickPlace structure.
        
        Args:
            obs: Current observation
            
        Returns:
            3D position of the object to be picked or None if not found
        """
        # Try to find the position of the first active object in the observation
        
        # In robosuite PickPlace, object positions are available in the observation dictionary
        if isinstance(obs, dict):
            # Try to find positions for the standard PickPlace objects (bread, milk, cereal, can)
            for obj_name in self.object_names:
                # Common observation key patterns in robosuite
                possible_keys = [
                    f'{obj_name}_pos',
                    f'object-state:{obj_name}',
                    f'{obj_name}-state',
                    f'{obj_name}_state'
                ]
                
                for key in possible_keys:
                    if key in obs:
                        # Usually first 3 elements are position
                        if len(obs[key]) >= 3:
                            return obs[key][:3]
            
            # Try generic object keys if specific ones not found
            for key in ['object-state', 'object_pos']:
                if key in obs and len(obs[key]) >= 3:
                    return obs[key][:3]
        
        # If we have access to the environment, try direct access
        if self.pick_place_env is not None:
            try:
                # Get position of the first object - handle both dict and list types for objects
                if len(self.object_names) > 0:
                    obj_name = self.object_names[0]
                    if hasattr(self.pick_place_env, 'objects'):
                        if isinstance(self.pick_place_env.objects, dict) and obj_name in self.pick_place_env.objects:
                            return self.pick_place_env.objects[obj_name].get_position()
                        elif isinstance(self.pick_place_env.objects, list) and len(self.pick_place_env.objects) > 0:
                            # Try to get the first object's position
                            return self.pick_place_env.objects[0].get_position()
            except (AttributeError, IndexError, KeyError, TypeError):
                pass
        
        # Default fallback for flattened observation
        # In robosuite's default configuration, object position often follows
        # the robot's proprioceptive state
        if isinstance(obs, np.ndarray) and len(obs) > 10:
            return obs[7:10]  # Approximate position for common setups
            
        return None
        
    def _get_target_position(self, obs: Dict) -> Optional[np.ndarray]:
        """
        Extract target position from observation based on robosuite's PickPlace structure.
        
        Args:
            obs: Current observation
            
        Returns:
            3D position of the target location or None if not found
        """
        # In robosuite PickPlace, target positions can be in observation dictionary
        if isinstance(obs, dict):
            # Try common keys for target position
            possible_keys = [
                'target-state',
                'target_pos',
                'bin_pos',
                'goal_pos'
            ]
            
            for key in possible_keys:
                if key in obs and len(obs[key]) >= 3:
                    return obs[key][:3]
        
        # If we have access to the environment, try direct access
        if self.pick_place_env is not None:
            # Try to get target bin position
            if self.target_bins is not None:
                try:
                    # Get position of the first target bin
                    target_bin_name = list(self.target_bins.keys())[0]
                    return self.pick_place_env.target_bins[target_bin_name].get_position()
                except (AttributeError, IndexError, KeyError):
                    pass
            
            # Alternative approach: get from environment task
            try:
                # In some versions, the target position is in a placement initializer
                if hasattr(self.pick_place_env, 'task_placement_initializer'):
                    placements = self.pick_place_env.task_placement_initializer.placements
                    if 'target' in placements:
                        return placements['target'][0]
            except (AttributeError, IndexError, KeyError):
                pass
        
        # Default fallback for flattened observation
        # In robosuite, target position often follows object position
        if isinstance(obs, np.ndarray) and len(obs) > 13:
            return obs[10:13]  # Approximate position for common setups
            
        return None
        
    def _is_object_grasped(self, gripper_state: Optional[np.ndarray], 
                          gripper_pos: Optional[np.ndarray], 
                          obj_pos: Optional[np.ndarray]) -> bool:
        """
        Check if the object is currently grasped by the gripper.
        
        Args:
            gripper_state: State of the gripper (typically joint positions)
            gripper_pos: 3D position of the gripper
            obj_pos: 3D position of the object
            
        Returns:
            Whether the object is grasped
        """
        if gripper_state is None or gripper_pos is None or obj_pos is None:
            return False
            
        # Calculate distance between gripper and object
        gripper_obj_distance = np.linalg.norm(gripper_pos - obj_pos)
        
        # Check if gripper is in a "holding" state
        # For most grippers, a partially closed state indicates holding
        if len(gripper_state) >= 1:
            # For typical grippers, a value between fully open and closed indicates grasping
            gripper_closed_partially = 0.01 < np.mean(gripper_state) < 0.05
            
            # Object is considered grasped if gripper is partially closed and object is close
            return gripper_closed_partially and gripper_obj_distance < self.grasp_threshold
        
        # Alternative detection if gripper state is not available
        # Check if object moves with gripper (object position relative to gripper stays constant)
        if self.prev_object_pos is not None:
            obj_movement = np.linalg.norm(obj_pos - self.prev_object_pos)
            return obj_movement > 0.005 and gripper_obj_distance < self.grasp_threshold
        
        return False
        
    def _is_object_at_target(self, obj_pos: np.ndarray, target_pos: np.ndarray) -> bool:
        """
        Check if the object is at the target position.
        
        Args:
            obj_pos: 3D position of the object
            target_pos: 3D position of the target
            
        Returns:
            Whether the object is at the target
        """
        if obj_pos is None or target_pos is None:
            return False
            
        # Calculate distance between object and target
        distance = np.linalg.norm(obj_pos - target_pos)
        
        # For PickPlace, we mostly care about x-y distance (placement on surface)
        # and less about exact z-height
        distance_xy = np.linalg.norm(obj_pos[:2] - target_pos[:2])
        
        # Object is at target if it's within threshold distance
        # We're a bit more lenient with the height (z-axis)
        return distance_xy < self.target_threshold
        
    def reset(self):
        """Reset the event detector state."""
        self.object_grasped = False
        self.reached_target = False
        self.prev_object_pos = None


class RewardMachineWrapper(gym.Wrapper):
    """
    A wrapper for robosuite environments that uses a reward machine to determine rewards.
    Compatible with the latest Gymnasium API.
    """
    
    def __init__(self, env, reward_machine: Optional[RewardMachine] = None):
        """
        Initialize the wrapper.
        
        Args:
            env: Robosuite environment to wrap
            reward_machine: Reward machine to use for determining rewards
        """
        super().__init__(env)
        self.env = env
        self.reward_machine = reward_machine or self._create_default_reward_machine()
        self.event_detector = PickPlaceEventDetector(env)
        
    def _create_default_reward_machine(self) -> RewardMachine:
        """
        Create a default reward machine for the PickPlace task.
        
        Returns:
            Default reward machine
        """
        # Define a simple reward machine with 3 states:
        # State 0: initial state (object not grasped, not at target)
        # State 1: object grasped
        # State 2: object at target (terminal state)
        
        rm = RewardMachine(states=[0, 1, 2], initial_state=0, terminal_states=[2])
        
        # Add transitions
        # (state, event) -> (next_state, reward)
        rm.add_transition(0, "object_grasped", 1, 0.5)  # Reward for grasping the object
        rm.add_transition(1, "object_released", 0, -0.1)  # Penalty for releasing too early
        rm.add_transition(1, "reached_target", 2, 1.0)  # Reward for reaching the target with object
        
        return rm
        
    def step(self, action):
        """
        Step the environment with an action and update the reward machine.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            next_obs: Next observation
            reward: Reward (determined by the reward machine)
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
            info: Additional information
        """
        # Step the environment
        next_obs, env_reward, terminated, truncated, info = self.env.step(action)
        
        # Detect events
        events = self.event_detector.detect_events(next_obs)
        
        # Update reward machine and get reward
        reward = 0
        for event in events:
            reward += self.reward_machine.step(event)
            
        # Check if reward machine reached terminal state
        if self.reward_machine.is_terminal():
            terminated = True
            
        # Add information about the reward machine state to info dict
        info['rm_state'] = self.reward_machine.current_state
        info['rm_events'] = events
        info['rm_reward'] = reward
        info['env_reward'] = env_reward  # Keep the original environment reward for reference
        
        return next_obs, reward, terminated, truncated, info
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """
        Reset the environment and the reward machine.
        
        Args:
            seed: Random seed
            options: Additional options for reset
            
        Returns:
            Initial observation and info dictionary
        """
        # Reset with the new Gymnasium API
        obs, info = self.env.reset(seed=seed, options=options)
        self.reward_machine.reset()
        self.event_detector.reset()
        
        # Add reward machine info to the info dict
        info['rm_state'] = self.reward_machine.current_state
        
        return obs, info


def create_pickplace_rm_env(robot_type="Panda", reward_machine=None, **kwargs):
    """
    Create a PickPlace environment with a reward machine.
    
    Args:
        robot_type: Type of robot to use
        reward_machine: Custom reward machine to use (optional)
        kwargs: Additional arguments to pass to the environment
        
    Returns:
        Wrapped environment with reward machine
    """
    # Create the base environment
    env = suite.make(
        env_name="PickPlace",
        robots=robot_type,
        has_renderer=True,  # Set to True for visualization
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        **kwargs
    )
    
    # For debugging: Print the type of the objects attribute in the environment
    try:
        pick_place_env = env  # Start with the top-level env
        # Attempt to find the base environment through wrapper layers
        for _ in range(5):
            if hasattr(pick_place_env, 'env'):
                pick_place_env = pick_place_env.env
            elif hasattr(pick_place_env, '_env'):
                pick_place_env = pick_place_env._env
            
            # Check if this is the PickPlace environment
            if isinstance(pick_place_env, PickPlace):
                break
                
        if hasattr(pick_place_env, 'objects'):
            print(f"Found objects attribute of type: {type(pick_place_env.objects)}")
            if isinstance(pick_place_env.objects, list):
                print(f"Objects list length: {len(pick_place_env.objects)}")
                if len(pick_place_env.objects) > 0:
                    print(f"First object type: {type(pick_place_env.objects[0])}")
            elif isinstance(pick_place_env.objects, dict):
                print(f"Objects dict keys: {list(pick_place_env.objects.keys())}")
    except Exception as e:
        print(f"Debug info: {e}")
        
    # Create a list of objects to include in the environment
    # The default PickPlace task includes bread, milk, cereal, and can
    if "object_type" not in kwargs:
        # Use default objects if not specified
        print("Using default PickPlace objects: bread, milk, cereal, can")
    
    # Convert to Gymnasium environment
    try:
        # Try with 'gymnasium' version first
        env = GymWrapper(env, gym_version='gymnasium')
    except TypeError:
        # Fall back to default if 'gym_version' param not supported
        env = GymWrapper(env)
        # Convert to Gymnasium manually if needed
        if not isinstance(env.observation_space, gym.spaces.Space):
            print("Warning: Unable to use Gymnasium wrapper directly. Using basic GymWrapper.")
    
    # Wrap with reward machine
    env = RewardMachineWrapper(env, reward_machine)
    
    return env


# Example usage:
if __name__ == "__main__":
    # Create a custom reward machine (optional)
    custom_rm = RewardMachine(states=[0, 1, 2], initial_state=0, terminal_states=[2])
    custom_rm.add_transition(0, "object_grasped", 1, 0.25)
    custom_rm.add_transition(1, "object_released", 0, -0.2)
    custom_rm.add_transition(1, "reached_target", 2, 2.0)
    
    # Create environment with default reward machine
    env = create_pickplace_rm_env()
    
    # Or with custom reward machine
    # env = create_pickplace_rm_env(reward_machine=custom_rm)
    
    # Run a simple test
    obs, info = env.reset(seed=42)
    terminated = truncated = False
    total_reward = 0
    steps = 0
    max_steps = 100
    
    print(f"Initial reward machine state: {info.get('rm_state', 0)}")
    
    while not (terminated or truncated) and steps < max_steps:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Print events if any occurred
        if 'rm_events' in info and info['rm_events']:
            print(f"Step {steps}: Events {info['rm_events']}, RM State: {info['rm_state']}")
        
    print(f"Episode finished with total reward: {total_reward} in {steps} steps")
    if terminated and 'rm_state' in info:
        if info['rm_state'] == 2:  # Success state
            print("Successfully completed the task!")
        else:
            print(f"Episode terminated in reward machine state: {info['rm_state']}")
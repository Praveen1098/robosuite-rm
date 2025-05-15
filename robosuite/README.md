# RL with Reward Machines Using robosuite

## Setup
> Create Python virtual environment

```
$ virtualenv -p python3 robosuite-env && source ~/robosuite-env/bin/activate
```

> Clone Git repository
```
$ git clone https://github.com/Praveen1098/robosuite-rm.git
$ cd robosuite-rm
```

> Install dependencies
```
$ pip3 install -r requirements.txt
$ pip3 install -r requirements-extra.txt
```
## RL Training with Reward Machines

Before running the training/rollout script, 
1. Set the correct path for `logdir` variable in `rl_training/logs/tmp/SAC/pick_and_place_milk_1_item.json`
2. Set the correct path for `rl_training/logs/tmp/SAC/pick_and_place_milk_1_item.json` file in the main of `robosuite_sb3_train.py`
3. Set the correct path for `rl_training/logs/tmp/SAC/pick_and_place_milk_1_item.json` as well as the model path `/logs/tmp/SAC/pick_and_place_milk_1_item/SEED17/` in the main of `robosuite_sb3_rollout.py`
4. Also, set the correct path to `rl_training/src/pick_and_place_sb3/utils/episode_slide_resized.jpg` in the `generate_rollout_video` function of `robosuite_sb3_rollout.py` for propoer rollout visualization and rendering
> Run training script 
```
$ cd ~/robosuite-rm/robosuite/rl_training/src/pick_and_place_sb3/utils
$ python robosuite_sb3_train.py
```
> To view tensorboard

Open a new terminal
```
$ tensorboard --logdir='/path/to/rl_training/logs/tmp/SAC/pick_and_place_milk_1_item/SEED17/<current_log_file_name>'
```
> To visualize rollout
```
$ cd ~/robosuite-rm/robosuite/rl_training/src/pick_and_place_sb3/utils
$ python robosuite_sb3_rollout.py
```

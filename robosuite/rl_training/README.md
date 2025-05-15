# RL Project Template for Robosuite and SB3 Training

> :construction: **Note**: This project is under active development.

Reinforcement learning project template for using Stable Baselines3 algorithms to train robots in robosuite environments.

## Overview

This folder structure template can be used when starting new RL projects. This template is customized to work for Stable Baselines3 training and robosuite environments. It also incorporates the necessary files to install this project as a python package. The Python package files were created with the assumption that you are using Hatchling as your build backend. 

⚠ **Note**: This project has been tested and works on Ubuntu 22.04 and python 3.10.12. I give no guarantees on other platforms.

## Installation

Follow these steps to clone this repository into your local directory that holds your git repositories. This example assumes the name of that directory is git-workspace.

```
cd git-workspace
git clone git@github.com:irishMee/rl-template-robosuite-sb3.git <new-repo-name>
```

Please replace "new-repo-name" with the name of your planned project repository.

## Setting Up Python Virtual Environment
I set up a virtual environment by using virtualenv package and following the steps below:
```
pip install virtualenv
cd ~/python_envs <!--- Folder created to store python environments. -->
python -m venv <your_python_env_name>
source <your_python_env_name>/bin/activate
```

Install dependencies (when you are in root):

```
pip install -r requirements.txt

```

## Instructions for Installation
 This python package is developed for Python 3.10+. This package can be installed locally by following:

1. Pip install "path" where "path" is the top-level directory containing pyproject.toml file. Note that the editable option can be included to track any package modifications. If you plan to help develop this project, then make sure to install using the editable option as shown by:
```
pip install -e <path>[dev]
```

Remember to replace "path" with the directory path to the pyproject.toml file. This command will install all of the necessary dependencies to start robosuite and Stable Baselines3 training.

### Test robosuite installation:
```
python -m robosuite.demos.demo_random_action
```

### Training using Stable Baselines3
This package comes with a robosuite_sb3_train.py file. This file is found under the src/rl_template_robosuite_sb3/utils/ directory. The train file can be used to launch training runs. You will need to provide a json file with environment and algorithmic information. When saving the model information from training, it is best to keep that information off git version tracking. You can do this automatically by setting your log path under /tmp directory. All files under /tmp directories are automatically ignored as setup in the .gitignore file with this package. 

### Visualize or Record Trained Policies
This package comes with a robosuite_sb3_rollout.py file. This file is found under the src/rl_template_robosuite_sb3/utils/ directory. The rollout file can be used to record or visualize on screen trained policies in any robosuite environment.

## Contributing
If you would like to contribute, please reach out to the listed maintainer below.

## License
Licensed under MIT license.

## Maintainers
Charles Meehan -- [email](mailto:cmeehan2@umd.edu)

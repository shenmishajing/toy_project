# A Toy project based on [project template](https://github.com/shenmishajing/project_template)

You can refer to this project to implement new projects.

This template project is based on [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and [lightning-template](https://github.com/shenmishajing/lightning_template). Please read the docs of them before using this template.

## Installation

### Python

We recommend you use the latest version of Python, which works well generally and may provide a better performance. The minimum supported version of Python is `3.8`. You can use the following command to create a conda environment with the specific version of Python.

```bash
conda create -n toy_project python=<python-version>
```

### Pytorch

Install [Pytorch](https://pytorch.org/get-started/locally/) from their official site manually. You have to choose the version of Pytorch based on the cuda version on your machine. Similarly, we recommend you use the latest version of Pytorch, which works well generally and may provide a better performance. You can skip this step if it's fine to use the latest version of Pytorch and the `pip` will install it in the next section. The minimum supported version of Pytorch is `1.11`.

### This project and its dependencies

Generally, you can just use the latest dependencies without a specific version, so you can use the command as follows to install this project and all required packages.

```bash
pip install -e ".[all]"
```

## Experiments

### Hyperparameter search

We employ the wandb to search hyperparameters for the toy model. You can use the following command to init an wandb sweep first, where you can set the hyperparameters you want to search.

```bash
wandb sweep --project toy_project configs/sweeps/toy_model_toy_dataset_1x.yaml
```

Then you can use the following command to start the hyperparameter search.

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> wandb agent <sweep_id>
```
where the `<gpu_ids>` is the gpus you want to use and the `<sweep_id>` is the id of the sweep you have created in the previous step. The `<sweep_id>` is a string like `your-entity/your-project/your-sweep-id`. Refer to the [wandb doc](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) for more details about the hyperparameter search.

### Train the toy model

You can use the following command to train the toy model.

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> cli fit --config configs/runs/toy_model/toy_model_toy_dataset_1x.yaml
```

### Evaluate the toy model

You can use the following command to evaluate the toy model on validation or test datasets.

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> cli fit --config configs/runs/toy_model/toy_model_toy_dataset_1x.yaml --ckpt_path work_dirs/toy_model_toy_dataset_1x/<run_id>/checkpoints/<checkpoint_name>.ckpt
```

### Predict with the toy model

#### Without checkpoint

You can use the following command to draw the confusion matrix with the toy model.

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> cli predict --config configs/runs/toy_model/toy_model_toy_dataset_1x.yaml
```

You can find the results in the `prediction` folder under your working directory.

#### With checkpoint

You can use the following command to draw the confusion matrix with the toy model and a specific checkpoint.

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> cli predict --config configs/runs/toy_model/toy_model_toy_dataset_1x.yaml --ckpt_path work_dirs/toy_model_toy_dataset_1x/<run_id>/checkpoints/<checkpoint_name>.ckpt
```

You can find the results in the `prediction` folder under the parent directory of the checkpoint, specifically, the `prediction` folder under the `work_dirs/toy_model_toy_dataset_1x/<run_id>` directory.

## A Turitol for developing a new project based on [project template](https://github.com/shenmishajing/project_template)

After you have played with the toy project, you may want to develop a new project based on the [project template](https://github.com/shenmishajing/project_template). You can follow the following steps to do this.

### Pick a name for your project

Before you start, you should pick a name for your project. We will use `<project-name>` as the name of the project in the following sections. You should replace `<project-name>` with the name you pick.

### Clone and republish to GitHub

If you want to develop based on the [project template](https://github.com/shenmishajing/project_template), you have to clone and republish it to your own GitHub repository. You have to create a new private repository on GitHub with name `<project-name>` and then you can use the following commands to do this.

```bash
git clone https://github.com/shenmishajing/project_template.git <project-name>
cd <project-name>
git remote rename origin project_template
git remote add origin https://github.com/<your-github-name>/<project-name>.git
git push -u --tags origin main
```

### Installation and dependencies

You can install your project in the same way as the description in the [Installation](#installation) section.

If you have to add more dependencies, you can add them to the `project.dependencies` parameter in the `pyproject.toml` file and then use the following command to install them.

```bash
pip install -e ".[all]"
```

### Create a wandb academic team

We use [wandb](https://wandb.ai/) as the default logger. You have to create an account on their [site](https://wandb.ai/) and login following their [doc](https://docs.wandb.ai/quickstart). Then you can create a new academic team and invite your team members to join it.

Note that you can only create one academic team for free, and you can not change the name of the team after you create it. So you should pick a good name for the team. You may not want to use the `<project-name>` as the name of the team, since you may want to use the team for multiple projects. We will use `<team-name>` as the name of the team in the following sections. You should replace `<team-name>` with the name you pick.

### Change the name of the project

In the file `configs/default_runtime.yaml`, set the value of `trainer.logger.init_args.project` to `<project-name>`, and the value of `trainer.logger.init_args.entity` to `<team-name>`.

### Commit and push the first commits to Github

Use the following commands to commit and push the first commits to your GitHub repository.

```bash
git add .
git commit -m 'feat(project-name): set the project name'
cz bump 1.0.0
git push
```

You should see the first commits on your GitHub repository now.

### Set up the pre-commit hooks

We recommend you install the pre-commit hooks to improve the quality of your code and commit message. The pre-commit hooks can check your code before you commit it, which will ensure the styles of your code and commit message follow the rules you set. The pre-commit hooks can also fix some problems automatically, which will save you time. However, if you are unfamiliar with them, you may need to spend some time dealing with problems brought by them.

We use [pre-commit](https://pre-commit.com/) to manage the pre-commit hooks. You can use the following command to install the pre-commit hooks.

```bash
pre-commit install
```

Generally, most of the pre-commit hooks will not bother you, but the [commitizen](https://github.com/commitizen-tools/commitizen) hook requires your commit message to follow the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style, which may lead to more work. We recommend you use the [Commit Message Editor](https://marketplace.visualstudio.com/items?itemName=adam-bender.commit-message-editor) extension of [vscode](https://code.visualstudio.com/) to generate the commit message following the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style. If you are not using [vscode](https://code.visualstudio.com/), you can `cz commit` to commit your changes. It will guide you to write the commit message in the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style.

Although requiring your commit message to follow the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style may lead to more work, we will get git history with more meaningful commit messages, which is helpful when you want to find or roll your code to a previous version, and also it will facilitate the version management tools. The version of the project will be generated from the commit messages following the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style automatically. For more details, you can refer to the [Version, Tag and Release](https://lightning-template.readthedocs.io/en/latest/get_started/contribution.html#version-tag-and-release) section of the [Contribution doc](https://lightning-template.readthedocs.io/en/latest/get_started/contribution.html#) of [lightning-template](https://github.com/shenmishajing/lightning_template).

## Development

Refer to the [Usage doc](https://lightning-template.readthedocs.io/en/latest/get_started/usage.html#) of [lightning-template](https://github.com/shenmishajing/lightning_template) for more details about how to implement your models and datasets etc.

## Experiment

You can use similar commands as the toy project to train, evaluate and predict with your model.

### Train your model

You can use the following command to train your model.

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> cli fit --config configs/runs/path/to/config.yaml
```

### Evaluate your model

You can use the following command to evaluate your model on validation or test datasets.

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> cli fit --config configs/runs/path/to/config.yaml --ckpt_path work_dirs/<run_name>/<run_id>/checkpoints/<checkpoint_name>.ckpt
```

### Predict with the toy model

#### Without checkpoint

You can use the following command to draw the confusion matrix with the toy model.

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> cli predict --config configs/runs/path/to/config.yaml
```

You can find the results in the `prediction` folder under your working directory.

#### With checkpoint

You can use the following command to draw the confusion matrix with the toy model and a specific checkpoint.

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> cli predict --config configs/runs/path/to/config.yaml --ckpt_path work_dirs/<run_name>/<run_id>/checkpoints/<checkpoint_name>.ckpt
```

You can find the results in the `prediction` folder under the parent directory of the checkpoint, specifically, the `prediction` folder under the `work_dirs/<run_name>/<run_id>` directory.

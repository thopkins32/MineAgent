# MineAgent

Welcome! This is a package intended for research into virtual intelligence in Minecraft.

## What is "virtual intelligence"?

By virtual intelligence, we mean "artificial intelligence that exists within a virtual world."

## Why Minecraft?

Minecraft offers a diverse, creative, and challenging virtual environment. While it is far from a perfect simulator for the real world  (it's a video game...),
we believe that it is a close enough approximation that methods that work here may have the potential to work in the real world.

## How can I contribute?

If you would like to contribute to the project, you can start by  forking the repository and adding things there. Then submit a PR and we will review your work to see if it fits with what we are trying to accomplish. If you have ideas you think may benefit the project, feel free to submit an [Issue](https://github.com/thomashopkins32/Minecraft-Virtual-Intelligence/issues).

## State of the Project

We are still early in the planning and development phase of the project. The first goal is to develop visual perception and curiosity while restricting the agent to only move through the environment.

## Installation

First, clone the repository:

```bash
git clone https://github.com/thomashopkins32/Minecraft-Virtual-Intelligence.git
cd Minecraft-Virtual-Intelligence
```

__RECOMMENDED__: We recommend using [Pixi](https://github.com/prefix-dev/pixi) to install the project by simply running:

```bash
pixi install
```

If you want to a conda environment you can install like so:

```bash
conda create -n mineagent python=3.11
conda activate mineagent
pip install .

# Install custom MineDojo fork
git clone https://github.com/thomashopkins32/MineDojo.git
cd MineDojo
pip install .
cd ..
```

 The official MineDojo distribution as of writing this is no longer maintained and package versions aren't frozen which can lead to compatability errors. Because of this, we recommend installing the [forked version of MineDojo](https://github.com/thomashopkins32/MineDojo).

## Running the project

To run the project, you can use the `mineagent` command.

Either using Pixi:

```bash
pixi run mineagent
```

Or if you installed with conda or venv:

```bash
mineagent
```

This will start the project and you can use the `mineagent` command to run the project starting from the `engine.run` function.

To view a list of all the commands you can use, run `mineagent --help`.

## Technologies

- [PyTorch](https://pytorch.org/)
- [MineDojo](https://minedojo.org/) (custom fork [here](https://github.com/thomashopkins32/MineDojo))

# PPO implementation in Pytorch

## Introduction

This repository provides a Minimal PyTorch implementation of Proximal Policy Optimization (PPO) with clipped objective for OpenAI gym environments. It is primarily intended for beginners in [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning) for understanding the PPO algorithm. It can still be used for complex environments but may require some hyperparameter-tuning or changes in the code.

To keep the training procedure simple : 
  - It has a **constant standard deviation** for the output action distribution (**multivariate normal with diagonal covariance matrix**) for the continuous environments, i.e. it is a hyperparameter and NOT a trainable parameter. However, it is **linearly decayed**. (action_std significantly affects performance)
  - It uses simple **monte-carlo estimate** for calculating advantages and NOT Generalized Advantage Estimate (check out the OpenAI spinning up implementation for that).
  - It is a **single threaded implementation**, i.e. only one worker collects experience. [One of the older forks](https://github.com/rhklite/Parallel-PPO-PyTorch) of this repository has been modified to have Parallel workers

A concise explaination of PPO algorithm can be found [here](https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl)


## Usage

- To train a new network : run `train.py`
- To test a preTrained network : run `test.py`
- To plot graphs using log files : run `plot_graph.py`
- To save images for gif and make gif using a preTrained network : run `make_gif.py`
- All parameters and hyperparamters to control training / testing / graphs / gifs are in their respective `.py` file
- `PPO_colab.ipynb` combines all the files in a jupyter-notebook
- All the **hyperparameters used for training (preTrained) policies are listed** in the [`README.md` in PPO_preTrained directory](https://github.com/nikhilbarhate99/PPO-PyTorch/tree/master/PPO_preTrained)


## References

- [PPO paper](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning up](https://spinningup.openai.com/en/latest/)



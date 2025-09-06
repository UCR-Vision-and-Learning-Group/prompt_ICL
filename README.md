# Provable Benefits of Task-Specific Prompts for In-context Learning

[![arXiv](https://img.shields.io/badge/arXiv-2503.02102-b31b1b.svg)](https://arxiv.org/abs/2503.02102)

This repository contains the code to reproduce the numerical results in the paper, "Provable Benefits of Task-Specific Prompts for In-context Learning."

## Overview

The in-context learning (ICL) capabilities of modern language models have motivated a deeper mathematical understanding of sequence models. This work theoretically analyzes the use of task-specific prompts and prediction heads within a one-layer linear attention model. The setting considers a global task distribution that can be partitioned into a union of conditional task distributions.

Our results on the loss landscape show that task-specific prompts facilitate a **covariance-mean decoupling**, where prompt-tuning explains the conditional mean of the distribution, while the variance is learned through in-context examples. This perspective explains how jointly training prompts and attention weights can provably outperform fine-tuning after pretraining.

## Summary of Theoretical Contributions

This paper provides a comprehensive analysis of a 1-layer linear attention model for multi-task in-context learning, focusing on how different training strategies affect performance. The central concept introduced is **Covariance-Mean Decoupling**.

### Covariance-Mean Decoupling

The core idea is that the challenge of multi-task learning can be broken down into two parts:
1.  Learning the **mean** (![mu_k](https://latex.codecogs.com/svg.latex?\mu_k)) of each conditional task distribution.
2.  Learning the **covariance** (![Sigma_beta_k](https://latex.codecogs.com/svg.latex?\Sigma_{\beta_k})) of each conditional task distribution.

An optimal in-context learner should decouple these two estimation problems. The task-specific parameters (prompts and heads) should learn the task mean, while the attention weights should learn the task variance from the in-context examples.

### Analysis of Training Strategies

We analyze the degree of this decoupling under different training settings:

* **Plain Training**: Without task-specific prompts, the attention weights are forced to learn both the mean and the covariance from the in-context examples. This leads to a "fully coupled" and suboptimal loss, which serves as a theoretical upper bound. The optimal loss, ![\mathcal{L}_{\text{PT}}^{*}](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{PT}}^{*}), is a function of the biased mixed-task covariance, ![\tilde{\Sigma}_{\beta}](https://latex.codecogs.com/svg.latex?\tilde{\Sigma}_{\beta}):
    ![Biased Covariance Formula](https://latex.codecogs.com/svg.latex?\tilde{\Sigma}_{\beta}%20=%20\Sigma_{x}\sum_{k=1}^{K}\pi_{k}(\Sigma_{\beta_{k}}+\mu_{k}\mu_{k}^{\top}))

* **Fine-tuning with Prompts**: Introducing and tuning task-specific prompts allows the model to learn the task means (![mu_k](https://latex.codecogs.com/svg.latex?\mu_k)), partially decoupling the problem. This provably reduces the loss compared to plain training (![\mathcal{L}_{\text{FT}}^{*} \le \mathcal{L}_{\text{PT}}^{*}](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{FT}}^{*}%20\le%20\mathcal{L}_{\text{PT}}^{*})). However, the attention weights, having been pretrained on the biased distribution, are not optimal for the now-debiased task.

* **Joint Training with Prompts**: By jointly optimizing both prompts and attention weights, the model achieves greater decoupling. The prompts focus on the task means, while the attention weights can better focus on learning the unbiased covariance, ![\bar{\Sigma}_{\beta}](https://latex.codecogs.com/svg.latex?\bar{\Sigma}_{\beta}):
    ![Unbiased Covariance Formula](https://latex.codecogs.com/svg.latex?\bar{\Sigma}_{\beta}%20=%20\Sigma_{x}\sum_{k=1}^{K}\pi_{k}\Sigma_{\beta_{k}})
    This results in a lower loss than fine-tuning (![\mathcal{L}_{\text{JT}}^{*} \le \mathcal{L}_{\text{FT}}^{*}](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{JT}}^{*}%20\le%20\mathcal{L}_{\text{FT}}^{*})).

* **Joint Training with Prompts and Heads (Fully Decoupled)**: The addition of task-specific prediction heads, alongside prompts, allows the model to achieve complete covariance-mean decoupling. This configuration achieves the optimal, fully-decoupled loss (![\mathcal{L}_{\text{PGD}}^{*}](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{PGD}}^{*})), which serves as a theoretical lower bound for the multi-task ICL problem with a single-layer linear attention model.

The key inequality derived from the analysis is:
![Loss Inequality](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{JT}}^{*}%20\le%20\mathcal{L}_{\text{FT}}^{*}%20\le%20\mathcal{L}_{\text{PT}}^{*})

## Running the Experiments

### Dependencies

The code is written in Python 3 and relies on the following libraries:

* `tqdm`
* `numpy`
* `torch`
* `matplotlib`
* `pickle`

You can install these dependencies using `pip`:
```bash
pip install tqdm numpy torch matplotlib
```

### Experiment Script

The main experimental logic is contained in `prompt_icl.py`. This script defines the data generator, model architectures (`LinearAttn`, `OneStepGD`, `MultiLayerLinearAttn`), evaluation functions, and the different training setups.

To run the experiments, execute the `prompt_icl.py` script:
```bash
python prompt_icl.py
```

### Experiment Configuration

The experiments are defined within the `prompt_icl.py` script in the `experiments` list. You can modify the parameters in this list to explore different settings. Each experiment configuration includes:

* `name`: A unique name for the experiment.
* `description`: A brief description of the experiment.
* `means`: The task means.
* `cov_matrices`: The task covariance matrices.
* `noise_levels`: The noise levels.
* `mixture_weights`: The mixture weights.

### Results

Experimental results are saved as `.pkl` files in the `results/` directory. Each file contains a dictionary with the results for joint training and fine-tuning, along with the experiment metadata.

## Plotting

The `plot.ipynb` Jupyter Notebook contains code to visualize the experimental results. To use it, start a Jupyter Notebook server:

```bash
jupyter notebook
```

Then, open `plot.ipynb` and run the cells to generate the plots. The notebook will load the `.pkl` files from the `results/` directory and plot the test loss versus context length for the different training settings, comparing them against the theoretical curves derived in the paper.

## Citation

This paper has been accepted to the 28th International Conference on Artificial Intelligence and Statistics (AISTATS) 2025. If you find this work useful, please consider citing the paper:

```bibtex
@inproceedings{chang2025provable,
  title={Provable Benefits of Task-Specific Prompts for In-context Learning},
  author={Chang, Xiangyu and Li, Yingcong and Kara, Muti and Oymak, Samet and Roy-Chowdhury, Amit K},
  booktitle={Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2025},
  series={PMLR},
  volume={258}
}


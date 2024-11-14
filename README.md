# PDiT: Perception and Decision Transformer (Interleaved)

This is implementation of PDiT based agents as mentioned in the [paper](). Here for enhanced decision-making, we utilize Perception Transformers alongside Decision-Making Transformers. By interleaving Perception and Decision Transformers, PDiT optimizes decision-making in complex environments. 

### Overview
PDiT focuses on environments where actions are taken based on image inputs and text prompts. Specifically, we explore scenarios where image-based environments require actions determined by text prompts, which traditionally lack direct correlation between text commands and image data. PDiT aims to improve this interaction by enabling more efficient perception and decision processes.

### Training
We have done both offline and online reinforcement learning. For offline learning, we have used the dataset given below for RvS(Reinforcement via supervized learning). And for Unsupervised learning we have used PPO.

### Dataset
We are using the **MiniGrid BabyAI** dataset for training and evaluation. This dataset provides a simplified, grid-based environment that supports diverse agent tasks, which is ideal for testing perception and decision-making abilities.

Dataset URL: [MiniGrid BabyAI](https://minigrid.farama.org/environments/babyai/)

## Installation Steps

1. **Set Up the Environment**:
   - Install Python (ensure Python 3.7+).
   - Create a virtual environment for PDiT:
     ```bash
     python3 -m venv pd_environment
     ```
   - Activate the environment:
     - **For Linux/Mac**:
       ```bash
       source pd_environment/bin/activate
       ```
     - **For Windows**:
       ```bash
       pd_environment\Scripts\activate
       ```

2. **Install Required Packages**:
   - Use the `requirements.txt` file to install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

---

## Video Demos

The demo links can be found [here]().

## Weights 

The weights for PPO based training method are [here]().
The weights for RvS based training method are [here]().

---
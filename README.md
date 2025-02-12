# Hands-On Reinforcement Learning – TD 1

This repository contains my individual work for the Hands-On Reinforcement Learning project. The project is organized in several parts that demonstrate the implementation and evaluation of reinforcement learning (RL) algorithms on the CartPole environment and a robotic arm environment using both custom PyTorch implementations and high-level libraries like Stable-Baselines3.

---

## 1. REINFORCE on CartPole

### Implementation
- **File:** `reinforce_cartpole.ipynb`  
  I implemented the REINFORCE (Vanilla Policy Gradient) algorithm in PyTorch to solve the CartPole-v1 environment.
  
### Training Results
- Over 500 episodes, the total rewards progressively increased, showing the learning progress of the agent.  
- **Training Plot:**  
  ![Training Plot](/images/train_rewards.png)  
  *(Figure: Total rewards per episode during training)*

### Model Saving
- The trained model weights are saved in the file: `reinforce_cartpole.pth`.

### Evaluation
- **File:** `evaluate_reinforce_cartpole.ipynb`  
  The saved model was evaluated over 100 episodes. An episode is considered successful if it reaches a total reward of 500.
- **Evaluation Results:**  
  100% of the episodes reached a total reward of 500.
- **Evaluation Plot:**  
  ![Evaluation Plot](/images/eval_rewards.png)  
  *(Figure: Evaluation performance over 100 episodes)*

---

## 2. A2C with Stable-Baselines3 on CartPole

### Implementation
- **File:** `a2c_sb3_cartpole.ipynb`  
  I used the Advantage Actor-Critic (A2C) algorithm provided by the Stable-Baselines3 package to solve the CartPole environment.
- The training run shows that the total rewards quickly reach 500 within the first few episodes.
- **Training Plot:**  
  ![SB3 CartPole Training Plot](path/to/training_plot_sb3_cartpole.png)  
  *(Figure: Total rewards per episode during A2C training on CartPole)*

### Evaluation
- The model was also evaluated, achieving 100% of episodes with a total reward of 500.
- **Evaluation Plot:**  
  ![SB3 CartPole Evaluation Plot](path/to/evaluation_plot_sb3_cartpole.png)  
  *(Figure: Evaluation performance over 100 episodes)*

### Model Upload
- The trained A2C model is available on Hugging Face Hub:  
  [A2C CartPole Model](https://huggingface.co/oussamab2n/a2c-cartpole)

---

## 3. Tracking with Weights & Biases (W&B) on CartPole

### Training with W&B
- I integrated Weights & Biases in `a2c_sb3_cartpole.ipynb` to monitor the training progress.
- **W&B Run:**  
  [W&B Run for A2C CartPole](https://wandb.ai/benyahiamohammedoussama-ecole-central-lyon/wb_sb3)
- **Training Plot from W&B:**  
  ![W&B Training Plot](path/to/wandb_training_plot.png)  
  *(Figure: Training episodes total rewards as tracked by W&B)*

### Model Upload
- After training with W&B, the model was also uploaded on Hugging Face Hub:  
  [A2C CartPole (W&B) Model](https://huggingface.co/oussamab2n/a2c-cartpole-wb)

### Evaluation
- The evaluation run (100 episodes) shows that 69% of episodes reached a total reward of 500, while the remaining episodes were very close to 500.
- **Evaluation Plot:**  
  ![W&B Evaluation Plot](path/to/wandb_evaluation_plot.png)  
  *(Figure: Evaluation performance over 100 episodes)*

---

## 4. Full Workflow with Panda-Gym

### Implementation
- **File:** `a2c_sb3_panda_reach.ipynb`  
  I used Stable-Baselines3 to train an A2C model on the PandaReachJointsDense-v3 environment (a robotic arm task) using 500k timesteps.
- The code integrates tracking with Weights & Biases.

### Training Results
- **W&B Run for Panda-Gym:**  
  [Panda-Gym W&B Run](https://wandb.ai/benyahiamohammedoussama-ecole-central-lyon/panda-gym)
- The training performance shows robust learning behavior in the complex 3D control task.

### Model Upload and Evaluation
- The trained model has been uploaded on Hugging Face Hub:  
  [A2C Panda-Reach Model](https://huggingface.co/oussamab2n/a2c-panda-reach)
- **Evaluation:**  
  In an evaluation run over 100 episodes (counting episodes that reach a total reward of -0.25), 97% of episodes meet the criterion.
- **Evaluation Plot:**  
  ![Panda-Gym Evaluation Plot](path/to/evaluation_plot_panda_gym.png)  
  *(Figure: Evaluation performance on PandaReachJointsDense-v3 environment)*

---

## Conclusion

This project provided a hands-on experience with:
- Implementing a basic RL algorithm (REINFORCE) in PyTorch.
- Evaluating the RL model with a custom evaluation script.
- Using Stable-Baselines3 for more advanced RL tasks and integrating with external tools such as Weights & Biases and Hugging Face Hub.
- Addressing both classical control (CartPole) and a complex robotic control task (PandaReach).


---
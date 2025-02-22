# Hands-On Reinforcement Learning – TD 1  

This repository contains my individual work for the **Hands-On Reinforcement Learning** project. The project explores reinforcement learning (RL) techniques applied to the **CartPole** and **Panda-Gym robotic arm** environments. The goal is to implement and evaluate RL models using both **custom PyTorch implementations** and **high-level libraries like Stable-Baselines3**.  

---

## 1. REINFORCE on CartPole  

### Implementation  
- **File:** `reinforce_cartpole.ipynb`  
  The **REINFORCE (Vanilla Policy Gradient)** algorithm was implemented using PyTorch. The model learns an optimal policy for solving the **CartPole-v1** environment by updating the policy network using gradients computed from episode returns.

### Training Results  
- The training process lasted for **500 episodes**, and we observed a steady increase in total rewards, confirming that the model successfully learned to balance the pole.  
- **Training Plot:**  
  ![Training Plot](/images/train_rewards.png)  
  *(Figure: The total rewards increase per episode, showing a successful learning process.)*

### Model Saving  
- The trained model is saved as: `reinforce_cartpole.pth`.

### Evaluation  
- **File:** `evaluate_reinforce_cartpole.ipynb`  
  The model was evaluated over **100 episodes**, and the success criterion was reaching a total reward of **500**.  
- **Evaluation Results:**  
  - **100%** of the episodes reached a total reward of 500, demonstrating the model’s reliability.  
- **Evaluation Plot:**  
  ![Evaluation Plot](/images/eval_rewards.png)  
  *(Figure: The model consistently reaches a total reward of 500 over 100 evaluation episodes.)*

---

## 2. A2C with Stable-Baselines3 on CartPole  

### Implementation  
- **File:** `a2c_sb3_cartpole.ipynb`  
  I used **Advantage Actor-Critic (A2C)** from **Stable-Baselines3**, which is an advanced RL algorithm combining value-based and policy-based methods.  

### Training Results  
- The total rewards **quickly reach 500** within the first few episodes, indicating that **A2C is significantly more efficient** than the REINFORCE approach.  
- **Training Plot:**  
  ![SB3 CartPole Training Plot](/images/sb3_train.png)  
  *(Figure: A2C rapidly achieves optimal performance within a few episodes.)*

### Evaluation  
- The trained model was evaluated, and **100%** of the episodes successfully reached a total reward of **500**.  
- **Evaluation Plot:**  
  ![SB3 CartPole Evaluation Plot](/images/sb3_eval.png)  
  *(Figure: The A2C-trained model consistently achieves perfect performance over 100 episodes.)*

### Model Upload  
- The trained A2C model is available on Hugging Face Hub:  
  [A2C CartPole Model](https://huggingface.co/oussamab2n/a2c-cartpole)

---

## 3. Tracking with Weights & Biases (W&B) on CartPole  

### Training with W&B  
- **File:** `a2c_sb3_cartpole.ipynb`  
  The A2C training process was tracked using **Weights & Biases (W&B)** to monitor performance metrics.  
- **W&B Run:**  
  [W&B Run for A2C CartPole](https://wandb.ai/benyahiamohammedoussama-ecole-central-lyon/wb_sb3)  

### Training Analysis  
- **Observations:**  
  - The training curve indicates that the **A2C model converges very quickly**.  
  - The **performance remains stable**, showing that the policy does not degrade after convergence.  
- **Training Plot:**  
  ![W&B Training Plot](/images/sb3_wb_train.png)  
  *(Figure: Training performance tracked using W&B.)*

### Model Upload  
- The trained A2C model (tracked with W&B) is available on Hugging Face Hub:  
  [A2C CartPole (W&B) Model](https://huggingface.co/oussamab2n/a2c-cartpole-wb)

### Evaluation  
- **Evaluation Results:**  
  - **100%** of the episodes successfully reached a total reward of **500**.  
  - This further confirms that **A2C is highly stable and performs consistently well.**  
- **Evaluation Plot:**  
  ![W&B Evaluation Plot](/images/sb3_wb_eval.png)  
  *(Figure: Evaluation results tracked using W&B.)*

---

## 4. Full Workflow with Panda-Gym  

### Implementation  
- **File:** `a2c_sb3_panda_reach.ipynb`  
  I used **Stable-Baselines3** to train an **A2C model** on the **PandaReachJointsDense-v3** environment, which involves controlling a robotic arm to reach a target in **3D space**.  
- **Training Duration:** **500,000 timesteps**  
- The code integrates **Weights & Biases** for tracking.  

### Training Results  
- **W&B Run for Panda-Gym:**  
  [Panda-Gym W&B Run](https://wandb.ai/benyahiamohammedoussama-ecole-central-lyon/panda-gym)  
- **Observations:**  
  - The training curve **shows consistent improvement** over time.  
  - The model **learns to reach the target efficiently**.  
- **Training Plot:**  
  ![Training Total Rewards Plot](/images/panda_sb3_train.png)  
  *(Figure: The robotic arm’s learning progress over 500,000 timesteps.)*

### Model Upload and Evaluation  
- The trained model has been uploaded on Hugging Face Hub:  
  [A2C Panda-Reach Model](https://huggingface.co/oussamab2n/a2c-panda-reach)

### Evaluation  


- **Evaluation Process:**

- The model was evaluated over 100 episodes.
- An episode is considered successful if it reaches a total reward of -0.25.

- **Updated Evaluation Results:**

- **Total episodes with truncation:** 99/100
- **Average reward at truncation:** -7.68
- **Percentage of episodes meeting the reward threshold:** 97%, indicating strong performance.


- **Evaluation Plot:**  
  ![Evaluation Plot](/images/panda_sb3_eval.png)  
  *(Figure: The robotic arm’s performance on the PandaReachJointsDense-v3 environment.)*

---

## Conclusion  

This project provided a comprehensive hands-on experience with **reinforcement learning**, covering both **custom implementation** and **high-level library usage**. The key takeaways include:  

✅ **Custom RL Implementation (REINFORCE)**
- Demonstrated a **gradual learning process** over 500 episodes.  
- Achieved **100% success rate** in evaluation.  

✅ **Stable-Baselines3 (A2C)**
- Achieved optimal performance **very quickly** compared to REINFORCE.  
- The model remained **stable across multiple evaluation runs**.  

✅ **Tracking with Weights & Biases**
- Provided **real-time tracking** and performance analysis.  
- Confirmed the **stability and consistency** of the trained models.  

✅ **Robotic Control with Panda-Gym**
- Successfully trained an **A2C agent** to control a robotic arm in **3D space**.  
- **97% success rate** in evaluation.  

This project highlights the efficiency of **A2C over REINFORCE**, the benefits of **W&B tracking**, and the feasibility of **reinforcement learning in robotic control applications**. 🚀  

---
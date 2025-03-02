# Hands-On Reinforcement Learning – TD 1

This repository contains my individual work for the **Hands-On Reinforcement Learning** project. The project explores reinforcement learning (RL) techniques applied to the **CartPole** and **Panda-Gym robotic arm** environments. The goal is to implement and evaluate RL models using both **custom PyTorch implementations** and **high-level libraries like Stable-Baselines3**.

---

## 1. REINFORCE on CartPole

### Implementation  
- **File:** `reinforce_cartpole.ipynb`  
  The **REINFORCE (Vanilla Policy Gradient)** algorithm was implemented using PyTorch. The model learns an optimal policy for solving the **CartPole-v1** environment by updating the policy network using gradients computed from episode returns.

### Training Results  
- The model was trained for **500 episodes**, showing a steady increase in total rewards. The goal (total reward = 500) was reached consistently after **400 episodes**, confirming successful learning.
- **Training Plot:**  
  ![Training Plot](/images/train_rewards.png)  
  *(Figure: Total rewards increase per episode, indicating successful learning.)*

### Model Saving  
- The trained model is saved as: `reinforce_cartpole.pth`.

### Evaluation  
- **File:** `evaluate_reinforce_cartpole.ipynb`  
  The model was evaluated over **100 episodes**, with the success criterion being a total reward of **500**.
- **Evaluation Results:**  
  - **100%** of the episodes reached a total reward of **500**, demonstrating the model’s reliability.
- **Evaluation Plot:**  
  ![Evaluation Plot](/images/eval_rewards.png)  
  *(Figure: The model consistently reaches a total reward of 500 over 100 evaluation episodes.)*

- **Example Video:**  
  ![REINFORCE CartPole Evaluation Video](reinforce_cartpole.mp4)  

---

## 2. A2C with Stable-Baselines3 on CartPole

### Implementation  
- **File:** `a2c_sb3_cartpole.ipynb`  
  Implemented **Advantage Actor-Critic (A2C)** using **Stable-Baselines3**, which combines value-based and policy-based RL methods.

### Training Results  
- The model was trained for **500,000 timesteps**, reaching a total reward of **500** consistently after **400 episodes**. It continued training for **1,400 episodes**, confirming stable convergence similar to the REINFORCE approach.
- **Training Plot:**  
  ![SB3 CartPole Training Plot](/images/sb3_train.png)  
  *(Figure: A2C training performance over time.)*

### Evaluation  
- The trained model was evaluated, achieving **100% success**, with all episodes reaching a total reward of **500**.
- **Evaluation Plot:**  
  ![SB3 CartPole Evaluation Plot](/images/sb3_eval.png)  
  *(Figure: A2C model consistently achieves perfect performance over 100 episodes.)*

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
  - The training curve indicates that the **A2C model stabilizes after 1,300 episodes**.  
  - The model exhibits strong and consistent performance.
- **Training Plot:**  
  ![W&B Training Plot](/images/sb3_wb_train.png)  

### Model Upload  
- The trained A2C model (tracked with W&B) is available on Hugging Face Hub:  
  [A2C CartPole (W&B) Model](https://huggingface.co/oussamab2n/a2c-cartpole-wb)

### Evaluation  
- **Evaluation Results:**  
  - **100%** of episodes reached a total reward of **500**, confirming the model’s reliability.
- **Evaluation Plot:**  
  ![W&B Evaluation Plot](/images/sb3_wb_eval.png)  
  *(Figure: Evaluation results tracked using W&B.)*
- **Example Video:**  
  ![W&B Evaluation Video](a2c_sb3_cartpole.mp4)  
  The A2C model stabilizes the balancing process more efficiently due to its superior performance compared to the REINFORCE approach.

---

## 4. Full Workflow with Panda-Gym

### Implementation  
- **File:** `a2c_sb3_panda_reach.ipynb`  
  Used **Stable-Baselines3** to train an **A2C model** on the **PandaReachJointsDense-v3** environment, controlling a robotic arm to reach a target in **3D space**.
- **Training Duration:** **500,000 timesteps**  
- Integrated **Weights & Biases** for tracking.

### Training Results  
- **W&B Run for Panda-Gym:**  
  [Panda-Gym W&B Run](https://wandb.ai/benyahiamohammedoussama-ecole-central-lyon/panda-gym)  
- **Observations:**  
  - The training curve shows consistent improvement over time.  
  - The model successfully learns to reach the target efficiently.  
  - It stabilizes after **2,500 episodes**, with minor fluctuations in rewards.
- **Training Plot:**  
  ![Training Total Rewards Plot](/images/panda_sb3_train.png)  
  *(Figure: The robotic arm’s learning progress over 500,000 timesteps.)*

### Model Upload and Evaluation  
- The trained model is available on Hugging Face Hub:  
  [A2C Panda-Reach Model](https://huggingface.co/oussamab2n/a2c-panda-reach)

### Evaluation  
- **Evaluation Results:**  
  - The total reward across all episodes ranged between **0 and -1**, indicating stable control.  
  - **100% of episodes** met the success criteria.
- **Evaluation Plot:**  
  ![Evaluation Plot](/images/panda_sb3_eval.png)  
  *(Figure: The robotic arm’s performance in the PandaReachJointsDense-v3 environment.)*
- **Example Video:**  
  ![Panda-Gym Evaluation Video](a2c_sb3_panda_reach.mp4)  

---

## Conclusion  

This project successfully applied reinforcement learning techniques to control both a **CartPole system** and a **Panda-Gym robotic arm** using **REINFORCE** and **A2C** algorithms. The experiments demonstrated that:  

- **REINFORCE** efficiently learned an optimal policy for CartPole but required more episodes to stabilize.  
- **A2C (Stable-Baselines3)** improved training stability and efficiency, reaching optimal performance faster.  
- **Weights & Biases (W&B)** was valuable for tracking and analyzing training performance in real-time.  
- The **Panda-Gym experiment** showed that A2C effectively trained the robotic arm to reach targets in 3D space.  

These results confirm the effectiveness of policy-gradient-based RL methods for solving **control and robotics problems**, highlighting the advantages of **actor-critic approaches** in stabilizing learning. Future work could explore more **advanced RL algorithms** (e.g., PPO, SAC) and extend experiments to **more complex robotic tasks**.

Further improvements could include testing **PPO or SAC algorithms** for comparison and expanding experiments to **more complex robotic tasks**.

---


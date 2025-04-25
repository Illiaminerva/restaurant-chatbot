"""
Reinforcement learning training script for the restaurant chatbot.
Implements reward-based fine-tuning to improve response quality.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import os
import gc
import mlflow
import time
from transformers import get_linear_schedule_with_warmup
from models.chatbot import RestaurantChatbot
from evaluation.evaluate_responses import compute_metrics, evaluate_responses, TEST_QUERIES

class RewardModel:
    """
    Computes rewards for generated responses based on multiple criteria:
    - Relevance to query
    - Information completeness
    - Response quality
    """
    
    def __init__(self):
        """Initialize reward computation keywords."""
        self.keywords = {
            'price': ['cheap', 'expensive', 'affordable', 'budget', 'cost', 'price', '$'],
            'cuisine': ['italian', 'chinese', 'indian', 'mexican', 'japanese', 'thai'],
            'recommendation': ['recommend', 'try', 'suggest', 'visit', 'check out'],
            'location': ['nearby', 'close', 'around', 'location', 'area', 'downtown'],
            'quality': ['good', 'great', 'excellent', 'amazing', 'best', 'delicious']
        }

    def compute_reward(self, query: str, response: str) -> float:
        """
        Compute reward for a generated response.
        Considers multiple quality metrics and keyword matching.
        """
        reward = 0.0
        query_lower = query.lower()
        response_lower = response.lower()

        for category, words in self.keywords.items():
            if any(word in query_lower for word in words):
                if any(word in response_lower for word in words):
                    reward += 1.0
                else:
                    reward -= 0.5

        if '$' in response:
            reward += 0.5
        if any(word in response_lower for word in ['located', 'address', 'street']):
            reward += 0.5
        if any(word in response_lower for word in ['menu', 'dish', 'serve']):
            reward += 0.5

        words = response.split()
        if len(words) < 10:
            reward -= 1.0
        elif len(words) > 100:
            reward -= 0.5

        metrics = compute_metrics(response, query)
        reward += metrics['relevance_score'] * 2.0
        reward += metrics['completeness_score']
        reward += metrics['length_score']
        reward += metrics['overall_score']

        return reward

def train_rl(
    model_path='models/baseline/best_model.pt',
    output_dir='models/rl',
    num_episodes=1000,
    batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    max_length=128,
    experiment_name="restaurant_chatbot_rl",
    run_name=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the model using reinforcement learning.
    Fine-tunes baseline model using reward-based optimization.
    """
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.makedirs(output_dir, exist_ok=True)
    mlflow.set_experiment(experiment_name)
    if run_name is None:
        run_name = f"rl_run_{int(time.time())}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_type": "gpt2_rl",
            "num_episodes": num_episodes,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "max_length": max_length
        })

        model = RestaurantChatbot(device=device)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.model.gradient_checkpointing_enable()

        reward_model = RewardModel()
        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=0.01)

        best_avg_reward = float('-inf')
        episode_rewards = []
        running_rewards = []

        for episode in range(num_episodes):
            model.train()
            total_reward = 0
            optimizer.zero_grad()
            policy_loss = []

            for batch_idx in range(0, batch_size * gradient_accumulation_steps, batch_size):
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()

                batch_queries = TEST_QUERIES * (batch_size // len(TEST_QUERIES) + 1)
                batch_queries = batch_queries[batch_idx:batch_idx + batch_size]
                responses, rewards, log_probs = [], [], []

                for query in batch_queries:
                    try:
                        response, response_ids, log_prob = model.generate_with_log_prob(
                            query, max_length=max_length, temperature=0.7, top_k=50, top_p=0.9
                        )
                        reward = reward_model.compute_reward(query, response)
                        running_rewards.append(reward)
                        if len(running_rewards) > 100:
                            running_rewards.pop(0)

                        responses.append(response)
                        rewards.append(reward)
                        log_probs.append(log_prob)
                        total_reward += reward
                    except RuntimeError as e:
                        print(f"Error during generation: {e}")
                        continue

                if not responses:
                    continue

                rewards_tensor = torch.tensor(rewards, device=device)
                running_mean = torch.tensor(running_rewards, device=device).mean()
                running_std = torch.tensor(running_rewards, device=device).std()
                normalized_rewards = (rewards_tensor - running_mean) / (running_std + 1e-8)
                normalized_rewards = torch.clamp(normalized_rewards, min=-10, max=10)

                for log_prob, reward in zip(log_probs, normalized_rewards):
                    policy_loss.append(-log_prob * reward)

                del rewards, log_probs, responses, normalized_rewards
                if device == 'cuda':
                    torch.cuda.empty_cache()

            # ⬇️ Do backward and optimizer step ONCE per episode, not per batch
            if policy_loss:
                final_loss = torch.stack(policy_loss).mean()
                final_loss = final_loss / gradient_accumulation_steps
                final_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                policy_loss = []  # optional: clear just in case

            avg_reward = total_reward / (batch_size * gradient_accumulation_steps)
            episode_rewards.append(avg_reward)

            mlflow.log_metrics({
                "episode": episode,
                "average_reward": avg_reward,
                "policy_loss": final_loss.item() if 'final_loss' in locals() else 0.0
            })

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"Average Reward: {avg_reward:.4f}")
                if 'final_loss' in locals():
                    print(f"Policy Loss: {final_loss.item():.4f}")


            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                model_path = os.path.join(output_dir, 'best_rl_model.pt')
                torch.save(model.state_dict(), model_path)
                model.tokenizer.save_pretrained(output_dir)
                print(f"New best model saved with reward: {best_avg_reward:.4f}")
                mlflow.log_artifacts(output_dir)

            if (episode + 1) % 50 == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_episode_{episode+1}.pt')
                torch.save({
                    'episode': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_avg_reward': best_avg_reward,
                    'running_rewards': running_rewards,
                }, checkpoint_path)

                model.eval()
                with torch.no_grad():
                    test_metrics, test_responses = evaluate_responses(model)
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
                print("Test Metrics:", test_metrics)

        mlflow.log_metrics({
            "final_average_reward": episode_rewards[-1],
            "best_average_reward": best_avg_reward
        })

        return {
            'episode_rewards': episode_rewards,
            'best_reward': best_avg_reward,
            'final_metrics': test_metrics
        }

import pygame
import torch
from CONSTANTS import *
from Environment import Environment
import numpy as np
from PPO_Agent import PPO_Agent
from Graphics import Graphics
import os
import wandb

class Trainer:
    """
    Trainer class for running the Actor-Critic training loop.

    Attributes:
        graphics (Graphics): Handles graphics rendering for the environment.
        env (Environment): The game environment.
        player (ActorCriticAgent): The actor-critic agent.
        optim (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler): Scheduler for learning rate adjustment.
        transition_buffer (TransitionBuffer): Stores transitions for n-step returns.
    """
    def __init__(self, num):
        """
        Initialize the Trainer.

        Args:
            num (int): Identifier for this training run.
            n_step (int): Number of steps for n-step returns.
        """
        self.graphics = Graphics()
        self.env = Environment(surface=self.graphics.main_surf)
        self.num = num

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.player = PPO()
        self.init_params()
        self.transition_buffer = TransitionBuffer(maxlen=self.n_steps, gamma=self.gamma, device=self.device)

        self.checkpoint_path = f"Data/Actor_Critic{self.num}.pth"
        self.resume_wandb = False
        self.load_checkpoint()

        self.wb = WandB(
            "Space_Invaders",
            self.resume_wandb,
            self.num,
            self.checkpoint_path,
            self.learning_rate,
            self.epochs,
            self.start_epoch,
            self.gamma,
            self.player.policy_value,
            self.device,
        )

    def init_params(self):
        """
        Initialize hyperparameters and optimizer settings.

        Args:
            n_step (int): Number of steps for n-step returns.
        """
        self.best_score = 0
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.n_steps = 5
        self.epochs = 30000
        self.start_epoch = 0
        self.optim = torch.optim.Adam(self.player.policy_value.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, 100, gamma=0.90)
        self.scores = []
        self.losses = []
        self.avg_score = []
        self.avg = 0
        self.step = 0
        self.betta = 0.05    # entropy regularization weight.
        self.max_grad_norm = 0.5

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            self.resume_wandb = True
            checkpoint = torch.load(self.checkpoint_path)
            self.start_epoch = checkpoint['epoch'] + 1
            self.player.policy_value.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.player.policy_value.train()

    def update_model(self, done, next_value):
        """
        Update the model parameters using transitions from the buffer.

        Args:
            done (bool): Whether the episode has ended.
            next_value (float): The value of the next state.
        """
        action_probs, _, values = self.transition_buffer.get_all_transitions()
        n_step_returns = self.transition_buffer.calculate_n_step_returns(value=next_value, done=done)

        # Calculate delta values in a vectorized manner
        deltas = n_step_returns - values

        # Compute entropy to encourage exploration
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(-1).mean()

        # Compute actor and critic losses using mean
        actor_loss = -torch.mean(torch.log(action_probs) * deltas.detach()) - self.betta * entropy
        critic_loss = torch.mean(deltas ** 2)

        loss = actor_loss + critic_loss
        self.optim.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosive updates
        torch.nn.utils.clip_grad_norm_(self.player.policy_value.parameters(), self.max_grad_norm)
        
        self.optim.step()

        self.transition_buffer.clear()  # Clear the buffer after optimization
        self.loss = loss  # Assign loss for logging

    def train(self, epochs = 50000, n_steps = 5):
        """
        Run the training loop for the agent.
        """
        self.epochs = epochs
        self.n_steps = n_steps
        for epoch in range(self.start_epoch, self.epochs):
            self.env.restart()
            done = False
            state = self.env.state()
            self.step = 0
            while not done:
                self.graphics.clear()
                self.graphics.event_pump()
                # self.graphics.events()
                action, action_prob, value = self.player.get_action_and_value(state)
                reward, done = self.env.move(action=action)
                # next_state = self.env.state()
                self.transition_buffer.append((action_prob, reward, value))
                self.step += 1

                if len(self.transition_buffer) >= self.n_steps or done:
                    self.update_model(done, value)

                state = self.env.state()
                self.graphics.header_writing(env=self.env, epoch=epoch)
                self.graphics.update()

            self.scheduler.step()
            self.log_and_plot(epoch)

            if epoch % 1000 == 0 and epoch > 0:
                self.save_checkpoint(epoch)
        pygame.quit()

    def save_checkpoint(self, epoch):
        """
        Save model checkpoint.

        Args:
            epoch (int): Current training epoch.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.player.policy_value.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, self.checkpoint_path)

    def log_and_plot(self, epoch):
        """
        Log metrics and display training information.

        Args:
            epoch (int): Current training epoch.
        """
        print(
            f'num: {self.num} epoch: {epoch} loss: {self.loss.item():.5f} LR: {self.scheduler.get_last_lr()} step: {self.step} '
            f'score: {self.env.score} level: {self.env.level} best_score: {self.best_score} n_steps: {self.n_steps} average score : {self.avg}  '
        )
        self.best_score = max(self.best_score, self.env.score)
        # Log and compute average every 10 epochs
        if epoch % 1 == 0:
            self.scores.append(self.env.score)
            self.losses.append(self.loss.item())
            self.avg = sum(self.scores) / len(self.scores)
            self.avg_score.append(self.avg)
            self.wb.log(score=self.env.score, loss=self.loss.item(), avg=self.avg)
           

class WandB:
    """
    WandB class for logging metrics to Weights & Biases.
    """
    def __init__(self, project_name, resume, num, checkpoint_path,
                 learning_rate, epochs, start_epoch, gamma, model, device):
        """
        Initialize the WandB logger.

        Args:
            project_name (str): Name of the project.
            resume (bool): Whether to resume logging.
            num (int): Run identifier.
            checkpoint_path (str): Path to save checkpoints.
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Total number of epochs.
            start_epoch (int): Starting epoch.
            gamma (float): Discount factor.
            model (str): Model description.
            device (str): Device used (CPU or GPU).
        """
    
        wandb.init(
            project=project_name,
            resume=resume,
            id=f'{project_name} {num}',
            config={
                "name": f"{project_name} {num}",
                "checkpoint": checkpoint_path,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "start_epoch": start_epoch,
                "gamma": gamma,
                "Model": str(model),
                "device": str(device),
            },
        )
        
    def log(self, score, loss, avg):
        """
        Log training metrics to WandB.

        Args:
            score (float): Current score.
            loss (float): Current loss.
            avg (float): Average score.
        """
        wandb.log({"score": score, "loss": loss, "avg_score": avg})


if __name__ == "__main__":
    # Start the training process
    trainer = Trainer(num=709)
    trainer.train(n_steps=7)

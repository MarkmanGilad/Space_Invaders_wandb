import pygame
import torch
from CONSTANTS import *
from Environment import Environment
from ActorCritic_Agent import ActorCriticAgent
from Graphics import Graphics
import os
import wandb
from collections import deque

class TransitionBuffer:
    """
    TransitionBuffer stores transitions and calculates n-step returns for training.
    
    Attributes:
        buffer (deque): Stores transitions up to a maximum length.
        gamma (float): Discount factor for future rewards.
    """
    def __init__(self, maxlen, gamma):
        """
        Initialize the TransitionBuffer.

        Args:
            maxlen (int): Maximum allowed length of the buffer.
            gamma (float): Discount factor for future rewards.
        """
        self.buffer = deque(maxlen=maxlen)
        self.gamma = gamma

    def append(self, transition):
        """
        Append a transition to the buffer.

        Args:
            transition (tuple): A tuple containing state, action probability, reward, and value.
        """
        self.buffer.append(transition)

    def calculate_n_step_returns(self, value=0, done=False):
        """
        Calculate n-step returns from the current buffer.

        Args:
            value (float): The value of the next state.
            done (bool): Whether the episode is done.

        Returns:
            list: A list of n-step returns for each transition.
        """
        with torch.no_grad():
            G = value * (1 - done)
            n_step_returns = []
            for transition in reversed(self.buffer):
                _, _, reward, _ = transition
                G = reward + self.gamma * G
                n_step_returns.insert(0, G)
        return n_step_returns

    def get_all_transitions(self):
        """
        Retrieve all transitions stored in the buffer.

        Returns:
            tuple: A tuple containing states, action probabilities, rewards, and values.
        """
        states, action_probs, rewards, values = zip(*self.buffer)
        return states, action_probs, rewards, values

    def clear(self):
        """
        Clear the buffer.
        """
        self.buffer.clear()

    def __len__(self):
        """
        Return the length of the buffer.

        Returns:
            int: The number of transitions in the buffer.
        """
        return len(self.buffer)


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
    def __init__(self, num, n_step=5):
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
        self.player = ActorCriticAgent()
        self.init_params(n_step=n_step)
        self.transition_buffer = TransitionBuffer(maxlen=self.n_steps, gamma=self.gamma)

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

    def init_params(self, n_step):
        """
        Initialize hyperparameters and optimizer settings.

        Args:
            n_step (int): Number of steps for n-step returns.
        """
        self.best_score = 0
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.n_steps = n_step
        self.epochs = 30000
        self.start_epoch = 0
        self.optim = torch.optim.Adam(self.player.policy_value.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, 10000, gamma=0.95)
        self.scores = []
        self.losses = []
        self.avg_score = []
        self.avg = 0
        self.step = 0

    def load_checkpoint(self):
        """
        Load model checkpoint if it exists.
        """
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
        states, action_probs, _, values = self.transition_buffer.get_all_transitions()
        n_step_returns = self.transition_buffer.calculate_n_step_returns(value=next_value, done=done)

        actor_loss, critic_loss = 0, 0
        for G, value, action_prob in zip(n_step_returns, values, action_probs):
            delta = G - value
            actor_loss += -torch.log(action_prob) * delta.detach()
            critic_loss += delta ** 2

        loss = (actor_loss + critic_loss) / len(n_step_returns)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.transition_buffer.clear()  # Clear the buffer after optimization
        self.loss = loss  # Assign loss for logging

    def train(self):
        """
        Run the training loop for the agent.
        """
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.start_epoch, self.epochs):
            self.env.restart()
            done = False
            state = self.env.state()
            self.step = 0

            while not done:
                self.graphics.clear()
                self.graphics.events()
                action, action_prob, value = self.player.get_action_and_value(state)
                reward, done = self.env.move(action=action)
                next_state = self.env.state()
                self.transition_buffer.append((state, action_prob, reward, value))
                self.step += 1

                if len(self.transition_buffer) >= self.n_steps or done:
                    self.update_model(done, value)

                state = next_state
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
            f'epoch: {epoch} loss: {self.loss.item():.5f} LR: {self.scheduler.get_last_lr()} step: {self.step} '
            f'score: {self.env.score} level: {self.env.level} best_score: {self.best_score}'
        )

        # Log and compute average every 10 epochs
        if epoch % 10 == 0:
            self.scores.append(self.env.score)
            self.losses.append(self.loss.item())
            self.avg = (self.avg * (epoch % 10) + self.env.score) / (epoch % 10 + 1)

        if (epoch + 1) % 10 == 0:
            self.avg_score.append(self.avg)
            self.wb.log(score=self.env.score, loss=self.loss.item(), avg=self.avg)
            print(f'average score last 10 games: {self.avg} ')
            self.avg = 0


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
        if not resume:
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
        else:
            wandb.config.update(allow_val_change=True)

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
    trainer = Trainer(num=603, n_step=30)
    trainer.train()

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
    def __init__(self, maxlen, gamma):
        """
        Initialize the transition buffer.

        Args:
            maxlen (int): Maximum allowed length of the buffer.
            gamma (float): Discount factor for future rewards.
        """
        self.buffer = deque(maxlen=maxlen)
        self.gamma = gamma

    def append(self, transition):
        """Add a transition to the buffer."""
        self.buffer.append(transition)

    def calculate_n_step_return(self, value=0, done=False):
        """
        Calculate the n-step return (G) for all transitions in the buffer.
        Args:
            value (float): the critic value at the state after the last step.
            done (boolean): True if end of game.
        Returns:
            list: A list of n-step returns (G) for each transition in the buffer.
        """
        with torch.no_grad():
            G = value * (1 - done)
            n_step_returns = []
            for transition in reversed(self.buffer):
                _, _, reward, _ = transition
                G = reward + self.gamma * G
                n_step_returns.insert(0, G)
        return n_step_returns

    def pop(self):
        """Remove and return the first transition from the buffer."""
        return self.buffer.popleft()

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

    def first(self):
        """Return the first transition in the buffer."""
        return self.buffer[0]

    def __len__(self):
        """Return the number of transitions in the buffer."""
        return len(self.buffer)


class Trainer:
    def __init__(self, num, n_step=5):
        """Initialize the Trainer class with environment, agent, and training parameters."""
        self.graphics = Graphics()
        self.env = Environment(surface=self.graphics.main_surf)
        self.num = num

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.player = ActorCriticAgent()  # The agent responsible for action and value predictions

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
        """Initialize training parameters."""
        self.best_score = 0
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.n_steps = n_step  # Number of steps for n-step return
        self.epochs = 30000
        self.start_epoch = 0
        self.loss = 0
        self.avg = 0
        self.scores, self.losses, self.avg_score = [], [], []
        self.optim = torch.optim.Adam(self.player.policy_value.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, 10000, gamma=0.95)
        self.step = 0

    def load_checkpoint(self):
        """Load the checkpoint if it exists."""
        if os.path.exists(self.checkpoint_path):
            self.resume_wandb = True
            checkpoint = torch.load(self.checkpoint_path)
            self.start_epoch = checkpoint['epoch'] + 1
            self.player.policy_value.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.losses = checkpoint['losses']
            self.scores = checkpoint['scores']
            self.avg_score = checkpoint['avg_score']
        self.player.policy_value.train()

    def log_and_plot(self, epoch):
        """Log metrics and handle plotting at regular intervals."""
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

    def train(self):
        """Run the training loop for the agent."""
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.start_epoch, self.epochs):
            self.env.restart()
            done = False
            state = self.env.state()
            self.step = 0
            ############ Episode = Game ##############
            while not done:
                print(self.step, end='\r')  # Print current step for progress tracking
                self.step += 1
                self.graphics.clear()
                self.graphics.events()

                ############# Sample Environment #########################

                # Get action and value predictions from the agent
                action, action_prob, value = self.player.get_action_and_value(state)

                # Execute the action in the environment
                reward, done = self.env.move(action=action)
                next_state = self.env.state()

                # Store the current transition in the buffer
                self.transition_buffer.append((state, action_prob, reward, value))
             

                ########## Update after n steps ###########
                if len(self.transition_buffer) >= self.n_steps:
                    
                    # Retrieve the first transition from the buffer
                    state_, action_prob_, reward_, value_ = self.transition_buffer.first()
                    
                    # Calculate the return for the first action in buffer
                    G = self.transition_buffer.calculate_n_step_return(value=value, done=done)[0]
                    
                    delta = G - value_
        
                    # Calculate actor and critic losses 
                    actor_loss = -torch.log(action_prob_) * delta
                    critic_loss = delta ** 2  # Mean squared error for critic loss
                    loss = critic_loss + actor_loss
                    self.loss = loss            # Assign self.loss for logging

                    # Perform backpropagation and optimization
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    # Remove the processed transition from the buffer if not done
                    if not done:
                        self.transition_buffer.pop()
                    else:
                        self.transition_buffer.clear()
                
                state = next_state  # Move to the next state
                self.graphics.header_writing(env=self.env, epoch=epoch)
                self.graphics.update()
            
            self.scheduler.step()  # Update learning rate scheduler
            
            self.log_and_plot(epoch)

            # Save checkpoint every 1000 epochs
            if epoch % 1000 == 0 and epoch > 0:
                self.save_checkpoint(epoch)
            

        pygame.quit()


class WandB:

    def __init__(self, project_name, resume, num, checkpoint_path,
        learning_rate, epochs, start_epoch, gamma, model, device):
        # Initialize the WandB project for logging
        if not resume:
            wandb.init(
                project=project_name,
                resume=resume,
                id=f'{project_name} {num}',
                # Track hyperparameters and run metadata
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
        # Log metrics to WandB
        wandb.log(
            {
                "score": score,
                "loss": loss,
                "avg_score": avg,
            }
        )


if __name__ == "__main__":
    trainer = Trainer(num=600)
    trainer.train()

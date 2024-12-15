import pygame
import torch
from CONSTANTS import *
from Environment import Environment
from ActorCritic_Agent import ActorCriticAgent
from Graphics import Graphics
import os
import wandb

class TransitionBuffer:
    def __init__(self, maxlen, gamma):
        """
        Initialize the transition buffer.

        Args:
            maxlen (int): Maximum allowed length of the buffer.
            gamma (float): Discount factor for future rewards.
        """
        self.buffer = []
        self.maxlen = maxlen
        self.gamma = gamma

    def append(self, transition):
        """Add a transition to the buffer. Remove the oldest if exceeding maxlen."""
        self.buffer.append(transition)
        if len(self.buffer) > self.maxlen:
            self.buffer.pop(0)

    def calculate_n_step_return(self, done, player):
        """
        Update the n-step return (G) for all transitions in the buffer.

        Args:
            done (bool): Whether the episode has terminated.
            player (ActorCriticAgent): The agent used to predict the value of the next state.
        """
        if not self.buffer:
            return

        if not done:  # Add the critic value for the last state if the episode is not done
            with torch.no_grad():
                _, next_value = player.policy_value(self.buffer[-1][0])  # next_state
        else:
            next_value = 0

        # Calculate G (n-step return) iteratively from the last transition
        G = next_value
        updated_buffer = []
        for i in reversed(range(len(self.buffer))):
            state, action_prob, value, reward, done_flag, _ = self.buffer[i]
            G = reward + self.gamma * G * (1 - done_flag)
            updated_buffer.append((state, action_prob, value, reward, done_flag, G))

        self.buffer = updated_buffer[::-1]

    def pop(self):
        """Remove and return the first transition from the buffer."""
        return self.buffer.pop(0)

    def clear(self):
        """Clear the buffer."""
        self.buffer = []

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
            f'epoch: {epoch} loss: {self.loss.item():.7f} LR: {self.scheduler.get_last_lr()} step: {self.step} '
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
                self.transition_buffer.append((state, action_prob, value, reward, done, 0))

                # Update n-step returns in the buffer
                self.transition_buffer.calculate_n_step_return(done, self.player)

                state = next_state  # Move to the next state

                ########## Update after n steps ###########
                if len(self.transition_buffer) == self.n_steps or done:
                    # Retrieve the first transition from the buffer
                    _, first_action_prob, first_value, _, _, G = self.transition_buffer.first()

                    # Calculate actor and critic losses with detach tensors
                    delta = (G - first_value).detach()
                    actor_loss = -torch.log(first_action_prob) * delta
                    critic_loss = delta ** 2  # Mean squared error for critic loss
                    loss = actor_loss + critic_loss
                    # self.loss = loss.detach()

                    # Perform backpropagation and optimization
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    # Remove the processed transition from the buffer if not done
                    if not done:
                        self.transition_buffer.pop()
                    else:
                        self.transition_buffer.clear()

                self.graphics.header_writing(env=self.env, epoch=epoch)
                self.graphics.update()
            
            self.scheduler.step()  # Update learning rate scheduler
            
            self.log_and_plot(epoch)

            # Save checkpoint every 1000 epochs
            if epoch % 1000 == 0 and epoch > 0:
                self.save_checkpoint(epoch)
            

        pygame.quit()


class WandB:
    def __init__(
        self,
        project_name,
        resume,
        num,
        checkpoint_path,
        learning_rate,
        epochs,
        start_epoch,
        gamma,
        model,
        device,
    ):
        # Initialize the WandB project for logging
        if not resume:
            wandb.init(
                project=project_name,
                resume=resume,
                id=f'Space_invaders {num}',
                # Track hyperparameters and run metadata
                config={
                    "name": f"Space_invaders {num}",
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

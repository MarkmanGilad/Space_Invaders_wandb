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
        for i in reversed(range(len(self.buffer))):
            state, action_prob, value, reward, done_flag, _ = self.buffer[i]
            G = reward + self.gamma * G * (1 - done_flag)
            self.buffer[i] = (state, action_prob, value, reward, done_flag, G)

    def pop(self):
        """Remove and return the first transition from the buffer."""
        return self.buffer.pop(0)

    def clear(self):
        """Clear the buffer."""
        self.buffer = []

    def first(self):
        """Return the first transition in the buffer."""
        return self.buffer[0]

def main():
    graphics = Graphics()
    env = Environment(surface=graphics.main_surf)
    num = 600

    #region ###### params and models ############
    best_score = 0
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    player = ActorCriticAgent()  # The agent responsible for action and value predictions
    learning_rate = 0.001
    gamma = 0.99
    n_steps = 5  # Number of steps for n-step return
    epochs = 30000
    start_epoch = 0
    loss = torch.tensor(0)
    avg = 0
    scores, losses, avg_score = [], [], []
    optim = torch.optim.Adam(player.policy_value.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 10000, gamma=0.95)
    step = 0

    # Transition buffer
    transition_buffer = TransitionBuffer(maxlen=n_steps, gamma=gamma)
    #endregion

    #region ######## checkpoint Load ############
    checkpoint_path = f"Data/Actor_Critic{num}.pth"
    resume_wandb = False
    if os.path.exists(checkpoint_path):
        resume_wandb = True
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        player.policy_value.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        losses = checkpoint['loss']
        scores = checkpoint['scores']
        avg_score = checkpoint['avg_score']
    player.policy_value.train()
    #endregion

    wb = WandB(
        "Space_Invaders",
        resume_wandb,
        num,
        checkpoint_path,
        learning_rate,
        epochs,
        start_epoch,
        gamma,
        player.policy_value,
        device,
    )

    #region ########### training loop #####################

    for epoch in range(start_epoch, epochs):
        env.restart()
        done = False
        state = env.state()

        #region ############ Episode = Game ##############
        while not done:
            print(step, end='\r')  # Print current step for progress tracking
            step += 1
            graphics.clear()
            graphics.events()

            #region ############# Sample Environment #########################

            # Get action and value predictions from the agent
            action, action_prob, value = player.get_action_and_value(state)

            # Execute the action in the environment
            reward, done = env.move(action=action)
            next_state = env.state()

            # Store the current transition in the buffer
            transition_buffer.append((state, action_prob, value, reward, done, 0))

            # Update n-step returns in the buffer
            transition_buffer.calculate_n_step_return(done, player)

            state = next_state  # Move to the next state

            #endregion

            #region ########## Update after n steps ###########
            if len(transition_buffer.buffer) == n_steps or done:
                # Retrieve the first transition from the buffer
                first_state, first_action_prob, first_value, _, _, G = transition_buffer.first()

                # Calculate actor and critic losses
                delta = G - first_value
                actor_loss = -torch.log(first_action_prob) * delta.detach()
                critic_loss = delta ** 2  # Mean squared error for critic loss
                loss = actor_loss + critic_loss

                # Perform backpropagation and optimization
                optim.zero_grad()
                loss.backward()
                optim.step()

                # Remove the processed transition from the buffer if not done
                if not done:
                    transition_buffer.pop()
                else:
                    transition_buffer.clear()

            #endregion

            graphics.header_writing(env=env, epoch=epoch)
            graphics.update()

        # endregion
        scheduler.step()  # Update learning rate scheduler

        #region ####################### Plotting and Logging ####################
        print(
            f'epoch: {epoch} loss: {loss.item():.7f} LR: {scheduler.get_last_lr()} step: {step} '
            f'score: {env.score} level: {env.level} best_score: {best_score}'
        )
        step = 0
        if epoch % 10 == 0:
            scores.append(env.score)
            losses.append(loss.item())

        avg = (avg * (epoch % 10) + env.score) / (epoch % 10 + 1)

        if (epoch + 1) % 10 == 0:
            avg_score.append(avg)
            wb.log(score=env.score, loss=loss.item(), avg=avg)
            print(f'average score last 10 games: {avg} ')
            avg = 0

        if epoch % 1000 == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': player.policy_value.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': losses,
                'scores': scores,
                'avg_score': avg_score,
            }
            torch.save(checkpoint, checkpoint_path)
        #endregion

    pygame.quit()

    #endregion


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
    main()

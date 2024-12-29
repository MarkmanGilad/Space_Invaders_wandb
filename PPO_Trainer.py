import pygame
import torch
from CONSTANTS import *
from Environment import Environment
import numpy as np
from PPO_Agent import PPO_Agent
from Graphics import Graphics
from collections import deque
import os
import wandb


class Trainer:
    """
    Trainer class for running the Actor-Critic training loop.

    Attributes:
        graphics (Graphics): Handles graphics rendering for the environment.
        env (Environment): The game environment.
        agent (PPO_Agent): The PPO agent.
        optim (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler): Scheduler for learning rate adjustment.

    """
    def __init__(self, chkpt):
        """
        Initialize the Trainer.

        Args:
            num (int): Identifier for this training run.
            n_step (int): Number of steps for n-step returns.
        """
        self.graphics = Graphics()
        self.env = Environment(surface=self.graphics.main_surf)
        self.chkpt = chkpt
        self.logger = Logger(chkpt)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent = PPO_Agent(chkpt=self.chkpt, logger = self.logger)
        self.init_params()
        self.checkpoint_path = f"Data/PPO_checkpt{self.chkpt}.pth"
        self.resume_wandb = False
        self.load_checkpoint()

        self.wb = WandB(
            "Space_Invaders_PPO",
            self.resume_wandb,
            self.chkpt,
            self.checkpoint_path,
            self.epochs,
            self.n_steps,
            self.device,
            self.agent.actor,
            self.agent.critic,
            self.agent.gamma,
            self.agent.policy_clip,
            self.agent.value_clip,
            self.agent.n_epochs,
            self.agent.gae_lambda,
            self.agent.entropy_coefficient,
            self.agent.max_grad_norm,
            self.agent.batch_size,
            self.agent.lr_actor,
            self.agent.lr_critic,
            self.agent.optim_step,
            self.agent.optim_gamma
        )

    def init_params(self):
        """
        Initialize hyperparameters and optimizer settings.

        Args:
            n_step (int): Number of steps for n-step returns.
        """
        self.n_steps = 2048
        self.epochs = 30000
        self.start_epoch = 1
        self.step = 0
        self.save_epoch = 1000
        self.best_score = 0
        self.avg = 0
        self.scores = []
        self.losses = []
        self.avg_score = []
    
    def train(self, epochs = 50000):
        """
        Run the training loop for the agent.
        """
        agent = self.agent
        self.epochs = epochs

        for epoch in range(self.start_epoch, self.epochs):
            self.env.restart()
            done = False
            state = self.env.state()
            self.step = 0
            while not done:
                self.graphics.clear()
                self.graphics.event_pump()
                self.graphics.events()
                action, prob, val = agent.choose_action(state)
                reward, done = self.env.move(action=action)
                agent.remember(state, action, prob, val, reward, done)
                self.step += 1
                # if self.step % 10 == 0:
                    # print(f'self.step: {self.step} action: {action} prob: {prob} val: {val}')

                if done or self.step % self.n_steps == 0:
                    agent.learn(epoch)
                    # self.log_and_plot(epoch)   

                state = self.env.state()
                self.graphics.header_writing(env=self.env, epoch=epoch)
                self.graphics.update()
            
            self.save_checkpoint(epoch)
            self.log_and_plot(epoch)

        pygame.quit()

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            self.resume_wandb = True
            checkpoint = torch.load(self.checkpoint_path)
            self.start_epoch = checkpoint['epoch'] + 1
            self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.agent.actor.optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])
            self.agent.critic.optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
            self.agent.actor.scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
            self.agent.critic.scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])

    def save_checkpoint(self, epoch):
        """
        Save model checkpoint.

        Args:
            epoch (int): Current training epoch.
        """
        if epoch % 10==0:
            self.logger.save()

        if epoch % self.save_epoch != 0:
            return
        torch.save({
            'epoch': epoch,
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'actor_optim_state_dict': self.agent.actor.optimizer.state_dict(),
            'critic_optim_state_dict': self.agent.critic.optimizer.state_dict(),
            'actor_scheduler_state_dict': self.agent.actor.scheduler.state_dict(),
            'critic_scheduler_state_dict': self.agent.critic.scheduler.state_dict(),
        }, self.checkpoint_path)

    def log_and_plot(self, epoch, log_epoch=1):
        
        """
        Log metrics and display training information.

        Args:
            epoch (int): Current training epoch.
        """
        print(
            f'chkpt: {self.chkpt} epoch: {epoch}',
            f'actor_loss: {self.agent.actor_loss:.5f} critic_loss: {self.agent.critic_loss:.5f}',
            f'total_loss: {self.agent.total_loss:.5f}',
            f'actor_lr: {self.agent.actor.scheduler.get_last_lr()} critic_lr: {self.agent.critic.scheduler.get_last_lr()}',
            f'score: {self.env.score} level: {self.env.level}'
            
        )
        self.logger.log('actor_loss', self.agent.actor_loss)
        self.logger.log('critic_loss', self.agent.critic_loss)
        self.logger.log('total_loss', self.agent.total_loss)
        self.logger.log('actor_lr', self.agent.actor.scheduler.get_last_lr())
        self.logger.log('critic_lr', self.agent.critic.scheduler.get_last_lr())
        self.logger.log('score', self.env.score)
        self.logger.log('level', self.env.level)
        
        self.best_score = max(self.best_score, self.env.score)
        # Log and compute average every 10 epochs
        if epoch % log_epoch == 0:
            self.scores.append(self.env.score)
            self.avg = sum(self.scores) / len(self.scores)
            self.avg_score.append(self.avg)
            self.wb.log(score=self.env.score, actor_loss=self.agent.actor_loss,critic_loss=self.agent.critic_loss, 
                        total_loss=self.agent.total_loss, avg=self.avg)
           

class WandB:
    """
    WandB class for logging metrics to Weights & Biases.
    """
    def __init__(self, project_name, resume, chkpt, checkpoint_path, epochs, n_steps, device, actor_model, critic_model,
                 gamma, policy_clip, value_clip, n_epochs, gae_lambda, entropy_coefficient, max_grad_norm, batch_size,
                 lr_actor, lr_critic, optim_step, optim_gamma):
        """
        Initialize the WandB logger.

        Args:
            project_name (str): Name of the project.
            resume (bool): Whether to resume logging.
            chkpt (int): Run identifier.
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
            id=f'{project_name} {chkpt}',
            config={
                "name": f"{project_name} {chkpt}",
                "checkpoint": checkpoint_path,
                "epochs": epochs,
                "n_steps": n_steps, 
                "device": str(device),
                "actor_model":str(actor_model), 
                "critic_model":str(critic_model),
                 "gamma":gamma, 
                 "policy_clip":policy_clip, 
                 "value_clip":value_clip, 
                 "n_epochs":n_epochs, 
                 "gae_lambda":gae_lambda, 
                 "entropy_coefficient":entropy_coefficient, 
                 "max_grad_norm":max_grad_norm, 
                 "batch_size":batch_size,
                 "lr_actor":lr_actor, 
                 "lr_critic":lr_critic, 
                 "optim_step":optim_step, 
                 "optim_gamma":optim_gamma,
                 
                
            },
        )
        
    def log(self, score, actor_loss, critic_loss, total_loss, avg):
        """
        Log training metrics to WandB.

        Args:
            score (float): Current score.
            loss (float): Current loss.
            avg (float): Average score.
        """
        wandb.log({"score": score, "actor_loss": actor_loss, "critic_loss":critic_loss, "total_loss":total_loss, "avg_score": avg})

class Logger:
    '''
            'actor_loss': [],
            'critic_loss': [],
            'total_loss': [],
            'actor_lr': [],
            'critic_lr': [],
            'score': [],
            'level': [],
            'max_actor_grad': [],
            'max_critic_grad': [],
            'advantage_mean': [],
            'advantage_std': [],
        '''
        
    def __init__(self, chkpt, maxlen = 100):
        self.chkpt = chkpt
        self.log_dict = {}
        self.maxlen = maxlen
    
    def log (self, key, value):
        if key not in self.log_dict:
            self.log_dict[key] = deque(maxlen=self.maxlen)    
        self.log_dict[key].append(value)

    def save (self):
        torch.save(self.log_dict, f'Data/logger{self.chkpt}.pth',)

    def load (self):
        self.log_dict = torch.load(f'Data/logger{self.chkpt}.pth', weights_only=False)
    
    def print_key(self, key =None, range=10):
        print(key)
        print(list(self.log_dict[key])[-range:])
    
    def print_all(self):
        for key, item in self.log_dict.items():
            print (key, "\t", item)
    
    def print_keys(self):
        for key in self.log_dict:
            print(key)

if __name__ == "__main__":
    # Start the training process
    trainer = Trainer(chkpt=25)
    trainer.train()

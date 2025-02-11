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
        self.wandb = self.wandb_init(self.remark)
        self.agent.wandb = self.wandb

    def init_params(self):
        """
        Initialize hyperparameters and optimizer settings.

        Args:
            n_step (int): Number of steps for n-step returns.
        """
        self.n_steps = 512
        self.epochs = 100000
        self.start_epoch = 1
        self.step = 0
        self.save_epoch = 1000
        self.best_score = 0
        self.avg = 0
        self.remark = '''5 enemies 150 bullets for game'''
        self.scores = []
        self.losses = []
        self.avg_score = []
        self.reward = 0             # for logging
    
    def wandb_init(self, remark):
        project_name = "Space_Invaders_PPO"
        config={
                "name": f"{project_name} {self.chkpt}",
                "checkpoint": self.checkpoint_path,
                "epochs": self.epochs,
                "n_steps": self.n_steps, 
                "device": str(self.device),
                "actor_model":str(self.agent.actor), 
                "critic_model":str(self.agent.critic),
                 "gamma":self.agent.gamma, 
                 "policy_clip":self.agent.policy_clip, 
                 "value_clip":self.agent.value_clip, 
                 "n_epochs":self.agent.n_epochs, 
                 "gae_lambda":self.agent.gae_lambda, 
                 "max_grad_norm":self.agent.max_grad_norm, 
                 "batch_size":self.agent.batch_size,
                 "lr_actor":self.agent.lr_actor, 
                 "lr_critic":self.agent.lr_critic, 
                 "weight_decay": self.agent.weight_decay,
                 "optim_step":self.agent.optim_step, 
                 "optim_gamma":self.agent.optim_gamma,
                 "reward_hit":self.env.hit,
                 "reward_end_of_game": self.env.end_of_game,
                 "reward_end_of_stage": self.env.end_of_stage,
                 "reward_amunition": self.env.amunition_reward,
                 'reward_enemy_above': self.env.misile_above_reward,
                 'delta_width': self.env.delta,
                 'remark': remark,  
                 'critic_actor_ratio': self.agent.critic_actor_ratio,
                 'frame_skip': self.agent.frame_skip,  
                 'entropy_decay': self.agent.entropy_decay,
                 'entropy_decay_steps': self.agent.entropy_decay_steps, 
                 "entropy_coefficient":self.agent.entropy_coefficient, 
                 'entropy_coe_min':self.agent.entropy_coe_min,
                 'enemies': MAX_ENEMY_SHIPS,
                 'max_ammunition': MAX_AMMUNITION
            }
        return WandB(project_name, self.chkpt, config, self.resume_wandb)

    def train(self, epochs = 50000):
        """
        Run the training loop for the agent.
        """
        agent = self.agent
        self.epochs = epochs
        self.step = 0
        for epoch in range(self.start_epoch, self.epochs):
            self.env.restart()
            done = False
            self.reward = 0
            state = self.env.state()        
            if self.env.level == 1:                 # clearing score after logging only when new_game
                self.env.score = 0
            while not done:
                self.graphics.clear()
                self.graphics.event_pump()
                self.graphics.events()
                action, log_prob, val = agent.choose_action(state)
                reward, done = self.env.move(action=action)
                self.reward += reward        # for logging
                agent.remember(state, action, log_prob, val, reward, done)
                self.step += 1
                if self.step % self.n_steps == 0:
                    if not done:                        # calculate next state value
                        action, log_prob, val = agent.choose_action(state)
                        agent.learn(val)
                    else:
                        agent.learn(0.0)             # nect state value is 0.0

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
        if not hasattr(self.agent, 'actor_loss'):
            return
        
        print(
            f'chkpt: {self.chkpt} epoch: {epoch}',
            f'actor_loss: {self.agent.actor_loss:.5f} critic_loss: {self.agent.critic_loss:.5f}',
            f'total_loss: {self.agent.total_loss:.5f}',
            f'actor_lr: {self.agent.actor.scheduler.get_last_lr()[0]:.5f} critic_lr: {self.agent.critic.scheduler.get_last_lr()[0]:.5f}',
            f'score: {self.env.score} level: {self.env.level}',
            f'entropy_coefficient: {self.agent.entropy_coefficient:.4f}',
            f'sum_reward: {self.reward:.3f}'
            
        )
        self.logger.log('actor_loss', self.agent.actor_loss)
        self.logger.log('critic_loss', self.agent.critic_loss)
        self.logger.log('total_loss', self.agent.total_loss)
        self.logger.log('entropy', self.agent.entropy)
        self.logger.log('actor_lr', self.agent.actor.scheduler.get_last_lr())
        self.logger.log('critic_lr', self.agent.critic.scheduler.get_last_lr())
        self.logger.log('score', self.env.score)
        self.logger.log('level', self.env.level)
        
        self.best_score = max(self.best_score, self.env.score)
        # Log and compute average every log_epoch
        if epoch % log_epoch == 0:
            self.scores.append(self.env.score)
            self.avg = sum(self.scores) / len(self.scores)
            self.avg_score.append(self.avg)
            
            self.wandb(score = self.env.score, actor_loss = self.agent.actor_loss, critic_loss = self.agent.critic_loss,
                       total_loss = self.agent.total_loss, avg = self.avg, entropy = self.agent.entropy, 
                       advantage_mean = self.agent.advantage_mean, advantage_std = self.agent.advantage_std,
                       advantage_norm = self.agent.advantage_norm, reward = self.reward )
            self.wandb.log()

class WandB:
    def __init__(self, project_name, chkpt, config, resume):
        self.wandb_dict = {}
        wandb.init(
            project=project_name,
            resume=resume,
            id=f'{project_name} {chkpt}',
            config=config,
        )

    def update_dict (self, **kwds):
        self.wandb_dict.update(kwds)

    def log (self):
        wandb.log(self.wandb_dict)
        self.wandb_dict = {}

    def __call__(self, *args, **kwds):
        self.update_dict(**kwds)
    
class Logger:
        
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
    chkpt = torch.load('Data/train_number')
    chkpt+=1
    torch.save(chkpt, 'Data/train_number')
    trainer = Trainer(chkpt)
    trainer.train()

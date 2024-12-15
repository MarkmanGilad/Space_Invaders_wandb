import pygame
import torch
from CONSTANTS import *
from Environment import Environment
from ActorCritic_Agent import ActorCriticAgent
from Graphics import Graphics
import os
import wandb

def main ():
    
    graphics = Graphics()
    env = Environment(surface=graphics.main_surf)
    num = 600
    
    #region ###### params and models ############
    best_score = 0
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    player = ActorCriticAgent()
    learning_rate = 0.001
    gamma = 0.99
    epochs = 30000
    start_epoch = 0
    loss = torch.tensor(0)
    avg = 0
    scores, losses, avg_score = [], [], []
    optim = torch.optim.Adam(player.policy_value.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim,10000, gamma=0.95)
    step = 0
    #endregion

    #region ######## checkpoint Load ############
    checkpoint_path = f"Data/Actor_Critic{num}.pth"
    resume_wandb = False
    if os.path.exists(checkpoint_path):
        resume_wandb = True
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']+1
        player.policy_value.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        losses = checkpoint['loss']
        scores = checkpoint['scores']
        avg_score = checkpoint['avg_score']
    player.policy_value.train()
    #endregion

    wb = WandB ("Space_Invaders", resume_wandb, num, checkpoint_path, learning_rate, epochs, start_epoch, gamma, player.policy_value, device)
    
    #region ########### training loop #####################

    for epoch in range(start_epoch, epochs):
        env.restart()
        done = False
        state = env.state()
        
        #region ############ Episode = Game ##############
        while not done:
            print (step, end='\r')
            step += 1
            graphics.clear()
            graphics.events()
                        
            #region ############# Sample Environement #########################
            
            # Agent's move + Forward
            action, action_prob, value = player.get_action_and_value(state)
            
            # Step in the environment with the agent's action
            reward, done = env.move(action=action)
            next_state = env.state()

            # Get next value, using V(s_{t+1}) * (1 - done) to handle terminal states
            with torch.no_grad():
                _, next_value = player.policy_value(next_state)

            delta = reward + gamma * next_value * (1 - done) - value

            if done:
                best_score = max(best_score, env.score)
            state = next_state
            
            #endregion
            
            #region ########## compute loss ###########
            # Actor loss - forward
            actor_loss = -torch.log(action_prob) * delta.detach()
            
            # Critic loss - forward
            critic_loss = delta ** 2     # MSELoss
            
            # Total loss for this step - forward
            loss = actor_loss + critic_loss

            # Backward
            optim.zero_grad()
            loss.backward()
            optim.step()
        
            #endregion

            graphics.header_writing(env=env, epoch=epoch)
            graphics.update()
            # clock.tick(FPS)

        # endregion
        scheduler.step()
        
        #region ####################### ploting and logging ####################
        print (f'epoch: {epoch} loss: {loss.item():.7f} LR: {scheduler.get_last_lr()} step: {step} ' \
            f'score: {env.score} level: {env.level} best_score: {best_score}')
        step = 0
        if epoch % 10 == 0:
            scores.append(env.score)
            losses.append(loss.item())

        avg = (avg * (epoch % 10) + env.score) / (epoch % 10 + 1)
        
        if (epoch + 1) % 10 == 0:
            avg_score.append(avg)
            wb.log(score=env.score, loss=loss.item(),avg=avg)
            print (f'average score last 10 games: {avg} ')
            avg = 0

        if epoch % 1000 == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': player.policy_value.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': losses,
                'scores':scores,
                'avg_score': avg_score
            }
            torch.save(checkpoint, checkpoint_path)
        #endregion

    pygame.quit()

    #endregion


class WandB ():
    def __init__(self, project_name, resume, num, checkpoint_path, learning_rate, epochs, start_epoch, gamma, model, device):
        # set the wandb project where this run will be logged
        if not resume:
            wandb.init(
                project=project_name,
                resume=resume, 
                id=f'Space_invaders {num}',
                # track hyperparameters and run metadata
                config={
                "name": f"Space_invaders {num}",
                "checkpoint": checkpoint_path,
                "learning_rate": learning_rate,
                # "Schedule": f'{str(scheduler.milestones)} gamma={str(scheduler.gamma)}',
                "epochs": epochs,
                "start_epoch": start_epoch,
                "gamma": gamma,
                "Model":str(model),
                "device": str(device)
                })
        else:
            wandb.config.update(allow_val_change=True)
    
    def log(self, score, loss, avg):
        wandb.log ({
                "score": score,
                "loss": loss,
                "avg_score": avg
            })

    
        
if __name__ == "__main__":
    main ()
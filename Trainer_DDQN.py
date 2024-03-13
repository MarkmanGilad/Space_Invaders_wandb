import pygame
import torch
from CONSTANTS import *
from Environment import Environment
from DQN_Agent import DQN_Agent
from ReplayBuffer import ReplayBuffer
import os
import wandb

def main ():

    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Space')
    # clock = pygame.time.Clock()

    header_surf = pygame.Surface((WIDTH, 100))
    main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
    header_surf.fill(BLUE)
    main_surf.fill(LIGHTGRAY)

    env = Environment(surface=main_surf)

    screen.blit(header_surf, (0,0))
    screen.blit(main_surf, (0,100))
    write (header_surf, "Score: " + str(env.score) + " Ammunition: " + str(env.spaceship.ammunition))

    best_score = 0
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    ####### params and models ############
    player = DQN_Agent(devive=device)
    player_hat = DQN_Agent(devive=device)
    player_hat.DQN = player.DQN.copy()
    batch_size = 128
    buffer = ReplayBuffer(path=None)
    learning_rate = 0.0001
    ephocs = 200000
    start_epoch = 0
    C, tau = 3, 0.001
    loss = torch.tensor(0)
    avg = 0
    scores, losses, avg_score = [], [], []
    optim = torch.optim.Adam(player.DQN.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim,100000, gamma=0.50)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[5000*1000, 10000*1000, 15000*1000, 20000*1000, 25000*1000, 30000*1000], gamma=0.5)
    step = 0

    ######### checkpoint Load ############
    num = 301
    checkpoint_path = f"Data/checkpoint{num}.pth"
    buffer_path = f"Data/buffer{num}.pth"
    resume_wandb = False
    if os.path.exists(checkpoint_path):
        resume_wandb = True
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']+1
        player.DQN.load_state_dict(checkpoint['model_state_dict'])
        player_hat.DQN.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        buffer = torch.load(buffer_path)
        losses = checkpoint['loss']
        scores = checkpoint['scores']
        avg_score = checkpoint['avg_score']
    player.DQN.train()
    player_hat.DQN.eval()
    
    ################# Wandb.init #####################
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="Space_Invaders",
        resume=resume_wandb, 
        id=f'Space_invaders {num}',
        # track hyperparameters and run metadata
        config={
        "name": f"Space_invaders DDQN {num}",
        "checkpoint": checkpoint_path,
        "learning_rate": learning_rate,
        "Schedule": f'{str(scheduler.milestones)} gamma={str(scheduler.gamma)}',
        "epochs": ephocs,
        "start_epoch": start_epoch,
        "decay": epsiln_decay,
        "gamma": 0.99,
        "batch_size": batch_size, 
        "C": C,
        "tau":tau,
        "Model":str(player.DQN),
        "device": str(device)
        }
    )
    # wandb.config.update({"Model":str(player.DQN)}, allow_val_change=True)
    
    #################################

    for epoch in range(start_epoch, ephocs):
        env.restart()
        end_of_game = False
        state = env.state()
        while not end_of_game:
            print (step, end='\r')
            step += 1
            main_surf.fill(LIGHTGRAY)
            header_surf.fill(BLUE)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return
            
            ############## Sample Environement #########################
            action = player.get_Action(state=state, epoch=epoch)
            reward, done = env.move(action=action)
            next_state = env.state()
            buffer.push(state, torch.tensor(action, dtype=torch.int64), torch.tensor(reward, dtype=torch.float32), 
                        next_state, torch.tensor(done, dtype=torch.float32))
            if done:
                best_score = max(best_score, env.score)
                break

            state = next_state

            write(header_surf,"Level: " + str(env.level), (200, 20))
            write(header_surf, "epoch: " + str (epoch), (400, 20))
            write(header_surf, "Score: " + str(env.score), (200, 60))
            write(header_surf, "Ammunition: " + str(env.spaceship.ammunition),(400, 60))
            screen.blit(header_surf, (0,0))
            screen.blit(main_surf, (0,100))
            pygame.display.update()
            # clock.tick(FPS)
            
            if len(buffer) < MIN_BUFFER:
                continue
    
            ############## Train ################
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            Q_values = player.Q(states, actions)
            next_actions, _ = player.get_Actions_Values(next_states)
            Q_hat_Values = player_hat.Q(next_states, next_actions)


            loss = player.DQN.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()

        if epoch % C == 0:
            # player_hat.DQN.load_state_dict(player.DQN.state_dict())
            player_hat.fix_update(dqn=player.DQN)
            # player_hat.soft_update(dqn=player.DQN, tau=tau)

        #########################################
        print (f'epoch: {epoch} loss: {loss:.7f} LR: {scheduler.get_last_lr()} step: {step} ' \
               f'score: {env.score} level: {env.level} best_score: {best_score}')
        step = 0
        if epoch % 10 == 0:
            scores.append(env.score)
            losses.append(loss.item())

        avg = (avg * (epoch % 10) + env.score) / (epoch % 10 + 1)
        if (epoch + 1) % 10 == 0:
            avg_score.append(avg)
            wandb.log ({
                "score": env.score,
                "loss": loss.item(),
                "avg_score": avg
            })
            print (f'average score last 10 games: {avg} ')
            avg = 0

        if epoch % 1000 == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': player.DQN.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': losses,
                'scores':scores,
                'avg_score': avg_score
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(buffer, buffer_path)


def write (surface, text, pos = (50, 20)):
    font = pygame.font.SysFont("arial", 36)
    text_surface = font.render(text, True, WHITE, BLUE)
    surface.blit(text_surface, pos)


        
if __name__ == "__main__":
    main ()
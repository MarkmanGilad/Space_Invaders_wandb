import pygame
from CONSTANTS import *
from Human_Agent import Human_Agent
from Environment import Environment
from DQN_Agent import DQN_Agent
from ActorCritic_Agent import ActorCriticAgent
from Graphics import Graphics

def main ():

    graphics = Graphics()

    env = Environment(surface=graphics.main_surf)
    graphics.blit()

    player = Human_Agent()
    # player = DQN_Agent(parametes_path=None, train=False)
    # player = ActorCriticAgent(player=1)

    # Main Loop
    run = True
    while (run):
        graphics.clear()
        
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                run = False
            
        action = player.get_Action(events=events, state=env.state())
        reward, done = env.move(action=action)
        if done:
            graphics.write ("End Of Game - Score: " + str (env.score))
            graphics.write ("Another Game ?  Y / N", pos=(300, 60))
            
            graphics.blit()
            pygame.display.update()
            if another_game():
                env.restart()
            else:
                break
        graphics
        graphics.header_writing(env=env, epoch=None)
        graphics.update()
        graphics.tick()
   

def write (surface, text, pos = (50, 20)):
    font = pygame.font.SysFont("arial", 36)
    text_surface = font.render(text, True, WHITE, BLUE)
    surface.blit(text_surface, pos)

def another_game ():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_y]:
            return True
        if keys[pygame.K_n]:
            return False
        
if __name__ == "__main__":
    main ()
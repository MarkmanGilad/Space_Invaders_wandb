import pygame
from CONSTANTS import *
from Human_Agent import Human_Agent
from Environment import Environment
from DQN_Agent import DQN_Agent


def main ():

    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Space')
    clock = pygame.time.Clock()

    header_surf = pygame.Surface((WIDTH, 100))
    main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
    header_surf.fill(BLUE)
    main_surf.fill(LIGHTGRAY)

    env = Environment(surface=main_surf)

    screen.blit(header_surf, (0,0))
    screen.blit(main_surf, (0,100))

    player = Human_Agent()
    # player = DQN_Agent(parametes_path=None, train=False)
    
    write (header_surf, "Score: " + str(env.score) + " Ammunition: " + str(env.spaceship.ammunition))

    # Main Loop
    run = True
    while (run):
        main_surf.fill(LIGHTGRAY)
        header_surf.fill(BLUE)

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                run = False
            
        action = player.get_Action(events=events, state=env.state())
        reward, done = env.move(action=action)
        if done:
            write (header_surf, "End Of Game - Score: " + str (env.score))
            write (header_surf, "Another Game ?  Y / N", pos=(300, 60))
            screen.blit(header_surf, (0,0))
            pygame.display.update()
            if another_game():
                env.restart()
            else:
                break
        
        write(header_surf, "Score: " + str(env.score) ,(200, 60))
        write(header_surf, "Ammunition: " + str(env.spaceship.ammunition), (400, 60))
        write(header_surf,"Level: " + str(env.level), (200, 20))
        screen.blit(header_surf, (0,0))
        screen.blit(main_surf, (0,100))
        pygame.display.update()
        clock.tick(FPS)
   

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
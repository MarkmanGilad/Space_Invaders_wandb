import pygame
from CONSTANTS import *
from Environment import Environment

class Graphics:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Space')
        self.clock = pygame.time.Clock()

        self.header_surf = pygame.Surface((WIDTH, 100))
        self.main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
        self.clear()
        self.blit()
        
    def clear(self):
        self.header_surf.fill(BLUE)
        self.main_surf.fill(LIGHTGRAY)

    def blit(self):
        self.screen.blit(self.header_surf, (0,0))
        self.screen.blit(self.main_surf, (0,100))
        
    def events(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def update (self):
        self.blit()
        pygame.display.update()

    def write (self, text, pos = (50, 20)):
        font = pygame.font.SysFont("arial", 36)
        text_surface = font.render(text, True, WHITE, BLUE)
        self.header_surf.blit(text_surface, pos)

    def header_writing(self, env, epoch ):
        self.write("Level: " + str(env.level), (200, 20))
        self.write( "epoch: " + str (epoch), (400, 20))
        self.write( f"Score: {env.score:.2f}", (200, 60))
        self.write( f'Ammunition: {env.spaceship.ammunition}',(400, 60))

    def tick(self):
        self.clock.tick(FPS)
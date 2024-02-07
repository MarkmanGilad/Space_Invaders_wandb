import pygame
from CONSTANTS import *

class Bullet (pygame.sprite.Sprite):
    def __init__(self, pos, speed_x=0, speed_y= -10) -> None:
        super().__init__()
        self.image = pygame.Surface((5,5))
        self.image.fill(RED)
        self.rect = self.image.get_rect(midbottom = pos)
        self.mask = pygame.mask.from_surface(self.image)
        self.speed_x = speed_x
        self.speed_y = speed_y

    def update(self) -> None:
        self.move()

    def move (self):
        self.rect.y += self.speed_y
        if self.rect.y < 0:
            self.kill() 
        if self.rect.y > MAIN_SURF_HEIGHT:
            self.kill() 
       

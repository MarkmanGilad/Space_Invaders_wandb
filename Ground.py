import pygame
from CONSTANTS import *

class Ground (pygame.sprite.Sprite):
    def __init__(self) -> None:
        super().__init__()
        self.image = pygame.Surface((WIDTH, 40))
        self.image.fill(LIGHTGRAY)
        self.rect = self.image.get_rect(bottom = MAIN_SURF_HEIGHT)
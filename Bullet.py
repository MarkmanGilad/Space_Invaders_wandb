import pygame
from CONSTANTS import *

class Bullet (pygame.sprite.Sprite):

    state_index = []
    bullets_num = 0

    @classmethod
    def clear_state_index(cls):
        cls.state_index = [None] * cls.bullets_num

    def __init__(self, pos, speed_x=0, speed_y= -10, color = RED) -> None:
        super().__init__()
        self.image = pygame.Surface((5,5))
        self.image.fill(color)
        self.rect = self.image.get_rect(midbottom = pos)
        self.mask = pygame.mask.from_surface(self.image)
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.set_state_index()

    def update(self) -> None:
        self.move()

    def move (self):
        self.rect.y += self.speed_y
        if self.rect.y < 0:
            self.kill() 
        if self.rect.y > MAIN_SURF_HEIGHT:
            self.kill() 
       
    def set_state_index (self):
        free = self.__class__.state_index.index(None)
        self.__class__.state_index[free] = self
        self.state_pos = free

    def kill(self):
        super().kill()
        self.__class__.state_index[self.state_pos] = None

    


class Enemy_bullet(Bullet):

    state_index = [None] * MAX_ENEMY_BULLETS
    bullets_num = MAX_ENEMY_BULLETS

    def __init__(self, pos, speed_x=0, speed_y=-10, color=RED):
        super().__init__(pos, speed_x, speed_y, color)


class Ship_bullet (Bullet):

    state_index = [None] * SPACE_SHIP_BURST
    bullets_num = SPACE_SHIP_BURST

    def __init__(self, pos, speed_x=0, speed_y=-10, color=RED):
        super().__init__(pos, speed_x, speed_y, color)
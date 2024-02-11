import pygame
from CONSTANTS import *
import random
from Bullet import Bullet

class Enemy (pygame.sprite.Sprite):
    shoots_factor = ENEMY_SHOOTS_FACTOR
    speed_y = 40
    def __init__(self, img, pos, Enemy_bullets_Group) -> None:
        super().__init__()
        self.image = img
        self.rect = self.image.get_rect(topleft = pos)
        self.mask = pygame.mask.from_surface(self.image)
        self.speed_x = ENEMY_START_SPEED
        self.Enemy_bullets_Group = Enemy_bullets_Group

    def update(self) -> None:
        self.move()
        self.shoot()


    def move (self):
        if self.rect.right > WIDTH:
            self.rect.y += Enemy.speed_y
            self.speed_x = -self.speed_x
        if self.rect.left < 0:
            self.rect.y += Enemy.speed_y
            self.speed_x = -self.speed_x
        self.rect.x += self.speed_x

    def shoot (self):
        if random.random() < Enemy.shoots_factor/1000 and len(self.Enemy_bullets_Group) < MAX_ENEMY_BULLETS:
            self.Enemy_bullets_Group.add(Bullet(self.rect.midbottom,speed_y=ENEMY_BULLET_SPEED))

    

    

    
        

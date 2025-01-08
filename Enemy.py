import pygame
from CONSTANTS import *
import random
from Bullet import Bullet

class Enemy (pygame.sprite.Sprite):
    shoots_factor = ENEMY_SHOOTS_FACTOR
    speed_y = 40
    explotion_img = pygame.image.load("img/explosion.png")
    explotion_img = pygame.transform.scale(explotion_img, (40, 40))

    def __init__(self, img, pos, Enemy_bullets_Group, speed = ENEMY_START_SPEED) -> None:
        super().__init__()
        self.image = img
        self.rect = self.image.get_rect(topleft = pos)
        self.mask = pygame.mask.from_surface(self.image)
        self.speed_x = speed
        self.Enemy_bullets_Group = Enemy_bullets_Group
        self.live = 1 

    def update(self) -> None:
        if self.live == -1:
            self.live = 0
            return
        if self.live == 0:
            self.kill()
            return
         
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

    
    def explode(self):
        self.image = Enemy.explotion_img
        self.live = -1
    

    
        

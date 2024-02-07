import pygame
import numpy as np
import torch
from CONSTANTS import *
from SpaceShip import SpaceShip
from Enemy import Enemy
from Ground import Ground
import random

class Environment:
    def __init__(self, surface) -> None:
        self.bullets_Group = pygame.sprite.Group()
        self.spaceship = SpaceShip(SPACESHIP_URL, (WIDTH //2, HEIGHT - 100), self.bullets_Group)
        self.spaceship_Group = pygame.sprite.GroupSingle(self.spaceship)
        self.enemy_bullets_Group = pygame.sprite.Group()
        self.enemy_img = pygame.image.load(ENEMY_URL)
        self.enemy_img = pygame.transform.scale(self.enemy_img, (40, 40))
        self.enemy_Group = self.make_enemy_group()
        self.score = 0
        self.surface = surface
        self.level = 1
        self.ground = Ground()
        self.ground_Group = pygame.sprite.GroupSingle(self.ground)

    def make_enemy_group (self, row=ENEMY_ROWS, col=ENEMY_COLS, space_row = 80, space_col = 120):
        enemy_Group = pygame.sprite.Group()
        row , col = 3 , 6
        for r in range (row):
            for c in range (col):
                enemy_Group.add(Enemy(self.enemy_img, (c * space_col, r * space_row, ), self.enemy_bullets_Group))
        return enemy_Group
    
    def update (self):
        self.spaceship_Group.update()
        self.enemy_Group.update()
        self.bullets_Group.update()
        self.enemy_bullets_Group.update()
    
    def draw (self):
        surface = self.surface
        self.ground_Group.draw(surface)
        self.spaceship_Group.draw(surface)
        self.enemy_Group.draw(surface)
        self.bullets_Group.draw(surface)
        self.enemy_bullets_Group.draw(surface)

    def restart (self, add_speed = 0, add_shoot_factor = 0, new_game = True):
        if new_game:
            # width =  random.randint(50, WIDTH-50)
            width = WIDTH // 2 - 30
            self.spaceship.rect.midbottom = ( width, HEIGHT - 100)
            Enemy.speed_add = 0
            Enemy.shoots_factor = ENEMY_SHOOTS_FACTOR
            self.score = 0
            self.level = 1
        else:
            Enemy.speed_add += add_speed
            Enemy.shoots_factor += add_shoot_factor
            self.level += 1
        self.enemy_Group = self.make_enemy_group()
        self.spaceship.ammunition = MAX_AMMUNITION
        self.bullets_Group.empty()
        self.enemy_bullets_Group.empty()
        
    def move (self, action):
        reward = 0
        if action == 1:
            self.spaceship.move_left()
        elif action == 2:
            self.spaceship.move_right()
        elif action == 3:
            self.spaceship.shoot ()
        self.update()
        self.draw()
        reward += self.hits()
        if self.is_end_of_stage():
            reward += 0
            self.restart(add_speed=0.5, add_shoot_factor=0.1, new_game=False)
        self.score += reward
        done = self.is_end_of_Game()
        if done:
            reward -= 3
        return reward, done
    
    def is_end_of_stage (self):
        return len(self.enemy_Group) == 0
   
    def is_end_of_Game (self):
        enemy_landed = pygame.sprite.spritecollide(self.ground, self.enemy_Group, dokill=True)
        spaceship_hit = pygame.sprite.spritecollide(self.spaceship, self.enemy_bullets_Group, dokill=True, collided= pygame.sprite.collide_mask) 
        return len(enemy_landed) > 0 or len(spaceship_hit) > 0
        
    def hits (self):
        collisions = pygame.sprite.groupcollide(self.enemy_Group, self.bullets_Group, True, True, pygame.sprite.collide_mask)
        return len(collisions)
    
    def state (self):
        enemy_ships = ENEMY_COLS * ENEMY_ROWS * 3              # x,y,speed 3 * 6 * 3 = 54  
        enemy_speed_y = 1                                       # 1
        enemy_bullets = MAX_ENEMY_BULLETS * 2         # 10 * 2 = 20
        enemy_bullet_speed_y = 1                                # 1
        SpaceShip_pos_shape = 2                                 # 2
        SpaceShip_speed_x = 1                                   # 1
        SpaceShip_Bullet_pos_shape = SPACE_SHIP_BURST * 2       # 3 * 2 = 6
        SpaceShip_bullets_speed_y = 1                           # 1
        SpaceShip_ammunition = 1                                # 1
        level = 1                                               # 1
        # score = 1                                               # 1
        total = enemy_ships + enemy_speed_y + enemy_bullets + enemy_bullet_speed_y + SpaceShip_pos_shape + SpaceShip_speed_x + \
        SpaceShip_Bullet_pos_shape + SpaceShip_bullets_speed_y + SpaceShip_ammunition + level 
        # total = 88
        
        state_list = []
        # 0 - 53
        index = 0                                           # 0 - 53
        for sprite in self.enemy_Group:
            state_list.append(sprite.rect.centerx)
            state_list.append(sprite.rect.centery)
            state_list.append(sprite.speed_x)
            index += 3
        for i in range(enemy_ships-index):
            state_list.append(0)
        state_list.append(Enemy.speed_y)                    # 54
        index = 0
        for sprite in self.enemy_bullets_Group:             # 55 - 74
            state_list.append(sprite.rect.centerx)
            state_list.append(sprite.rect.centery)
            index += 2
        for i in range(enemy_bullets-index):
            state_list.append(0)
        state_list.append(ENEMY_BULLET_SPEED)               # 75
        state_list.append(self.spaceship.rect.centerx)      # 76
        state_list.append(self.spaceship.rect.centery)      # 77
        state_list.append(SPACESHIP_SPEED)                  # 78
        index = 0
        for sprite in self.bullets_Group:                   # 79 - 84
            state_list.append(sprite.rect.centerx)
            state_list.append(sprite.rect.centery)
            index += 2
        for i in range(SpaceShip_Bullet_pos_shape-index):
            state_list.append(0)
        state_list.append(SPACESHIP_BULLET_SPEED)           # 85
        state_list.append(self.spaceship.ammunition)        # 86
        state_list.append(self.level)                       # 87
        # state_list.append(self.score)                       # 88
        return torch.tensor(state_list, dtype=torch.float32)

     
    







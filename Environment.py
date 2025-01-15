
import pygame
import numpy as np
import torch
from CONSTANTS import *
from SpaceShip import SpaceShip
from Enemy import Enemy, Explosion
from Ground import Ground
from Bullet import Bullet, Enemy_bullet, Ship_bullet

import random


class Environment:
        
    def __init__(self, surface) -> None:
        self.bullets_Group = pygame.sprite.Group()
        self.spaceship = SpaceShip((WIDTH //2, HEIGHT - 100), self.bullets_Group)
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
        self.init_rewards()
        self.add_shoot_factor = 0.1
        self.next_stage = False
        self.Explosion_Group = pygame.sprite.Group()
        self.hit = 0
        self.end_of_game = False
        self.end_of_stage = False

    def init_rewards (self):
        self.game_reward = -100
        self.stage_reward = 100
        self.hit_reward = 1
        self.amunition_reward = -0.00
        self.misile_above_reward = -0.01
        self.delta = 7.5  # width of spaceship / 2

    def make_enemy_group (self, row=ENEMY_ROWS, col=ENEMY_COLS, space_row = 80, space_col = 120, speed = ENEMY_START_SPEED):
        enemy_Group = pygame.sprite.Group()
        
        row , col = 3 , 6
        # for r in range (row):
        #     for c in range (col):
        #         enemy_Group.add(Enemy(self.enemy_img, (c * space_col, r * space_row, ), self.enemy_bullets_Group,speed=speed))
        all_enemies = [(x, y) for x in range(4) for y in range(7)]
        sample_enemies = random.sample(all_enemies, 1)
        for r, c in sample_enemies:
            enemy_Group.add(Enemy(self.enemy_img, (c * space_col, r * space_row, ), self.enemy_bullets_Group,speed=speed))
        return enemy_Group
    
    def update (self):
        self.spaceship_Group.update()
        self.enemy_Group.update()
        self.bullets_Group.update()
        self.enemy_bullets_Group.update()
        self.Explosion_Group.update()
    
    def draw (self):
        surface = self.surface
        self.ground_Group.draw(surface)
        self.spaceship_Group.draw(surface)
        self.enemy_Group.draw(surface)
        self.bullets_Group.draw(surface)
        self.enemy_bullets_Group.draw(surface)
        self.Explosion_Group.draw(surface)

    def restart (self):
        width = WIDTH // 2 - 30
        Enemy.clear_state_index()
        Enemy_bullet.clear_state_index()
        Ship_bullet.clear_state_index()

        if self.end_of_stage:
            self.level += 1
            Enemy.shoots_factor += self.add_shoot_factor
            self.enemy_Group = self.make_enemy_group(speed= int(ENEMY_START_SPEED + self.level/2))
            self.spaceship.rect.midbottom = (width, HEIGHT - 100)
            
        if self.end_of_game:
            self.spaceship = SpaceShip((WIDTH //2, HEIGHT - 100), self.bullets_Group)
            self.spaceship_Group = pygame.sprite.GroupSingle(self.spaceship)
            self.spaceship.rect.midbottom = (width, HEIGHT - 100)
            Enemy.shoots_factor = ENEMY_SHOOTS_FACTOR
            self.score = 0
            self.level = 1
            self.enemy_Group = self.make_enemy_group()

        self.end_of_stage = False
        self.end_of_game = False      
        self.hit = 0  
        self.spaceship.ammunition = MAX_AMMUNITION
        self.bullets_Group.empty()
        self.enemy_bullets_Group.empty()    
        
    def move (self, action):
        reward = self.hit * self.hit_reward
        if self.end_of_stage:
            reward += self.stage_reward
            self.restart()
            return reward, False

        if self.end_of_game:
            reward += self.game_reward
            self.restart()
            return reward, True
        
        if action == 1:
            self.spaceship.move_left()
        elif action == 2:
            self.spaceship.move_right()
        elif action == 3:
            self.spaceship.shoot ()
            if self.spaceship.ammunition > 0:
                reward += self.amunition_reward              # don't waste ammunition
        
        self.update()
        self.draw()        
        if self.is_enemy_missile_above():
            reward += self.misile_above_reward
        
        self.hit = self.hits()
        self.score += self.hit
        self.is_end_of_stage()
        self.is_end_of_Game()
                
        return reward, False
    
    def is_enemy_missile_above(self):
        SpaceShip_x = self.spaceship.rect.centerx
        delta = self.delta
        is_bullet_above = any(abs(sprite.rect.centerx - SpaceShip_x) <= delta for sprite in self.enemy_bullets_Group)
        return is_bullet_above

    def is_end_of_stage (self):
        enemies = len(self.enemy_Group)
        self.end_of_stage = (enemies == 0)
   
    def is_end_of_Game (self):
        
        if self.spaceship.ammunition == 0 and len(self.enemy_Group) > 0 and len(self.bullets_Group)==0:      
            done = True
            return
        
        else:
            enemy_landed = pygame.sprite.spritecollide(self.ground, self.enemy_Group, dokill=True)
            spaceship_hit = pygame.sprite.spritecollide(self.spaceship, self.enemy_bullets_Group, dokill=True, collided= pygame.sprite.collide_mask) 
            done = len(enemy_landed) > 0 or len(spaceship_hit) > 0
        
        self.end_of_game = done

        
        
    def hits (self):
        collisions = pygame.sprite.groupcollide(self.enemy_Group, self.bullets_Group, True, True, pygame.sprite.collide_mask)
        for enemy, bullets in collisions.items():
            enemy.explode()
            # self.enemy_Group.remove(enemy)          # Remove the enemy from self.enemy_Group
            # self.bullets_Group.remove(bullets[0])   # Remove the first (and only) bullet from self.bullets_Group 
        for enemy, bullets in collisions.items():
            explosion = Explosion(enemy.rect.topleft)
            self.Explosion_Group.add(explosion)
        return len(collisions)
    
    def normX(self, x):
        return x / 500      #WIDTH=800
    
    def normY(self, y):
        return y / 500      #MAIN_SURF_HEIGHT = 500
    
    def normSpeed(self, s):
        return s / 10

    def state (self):
        normX = self.normX
        normY = self.normY
        normS = self.normSpeed
        
        ship_x = self.spaceship.rect.centerx
        ship_y = self.spaceship.rect.centery
        ship_w = 15 / 2
        ship_h = 34 / 2
        enemy_w = 34 / 2
        enemy_h = 30 / 2
        bullet_w = 5 / 2
        bullet_h = 5 / 2

        state_list = []
        state_list.append(len(self.enemy_Group)/ENEMY_SHIPS)

        for sprite in Enemy.state_index:
            if sprite:
                state_list.append(1)
                state_list.append(normX(sprite.rect.centerx-ship_x-enemy_w))
                state_list.append(normX(sprite.rect.centerx-ship_x+enemy_w))
                state_list.append(normY(sprite.rect.centery-ship_y-enemy_h))
                state_list.append( normY(sprite.rect.centery-ship_y+enemy_h))
                state_list.append( normS(sprite.speed_x))
            else:
                state_list.extend([0,0,0,0,0,0])
        
        state_list.append(normS(Enemy.speed_y))                     
        
        for sprite in Enemy_bullet.state_index:
            if sprite:
                state_list.append(1)
                state_list.append(normX(sprite.rect.centerx-ship_x-bullet_w))
                state_list.append(normX(sprite.rect.centerx-ship_x+bullet_w))
                state_list.append(normY(sprite.rect.centery-ship_y-bullet_h))
                state_list.append(normY(sprite.rect.centerx-ship_y+bullet_h))
            else:
                state_list.extend([0,0,0,0,0])
    
        state_list.append(normS(ENEMY_BULLET_SPEED))               
        state_list.append(normX(-ship_w))                               #      width of space ship to the right
        state_list.append(normX(ship_w))                               #      width of space ship to the right
        state_list.append(normY(-ship_h))                               #      width of space ship to the right
        state_list.append(normY(ship_h))                               #      width of space ship to the right
        state_list.append(normS(SPACESHIP_SPEED))                  
        
        for sprite in Ship_bullet.state_index:                         
            if sprite:
                state_list.append(1)
                state_list.append(normX(sprite.rect.centerx-ship_x-bullet_w))
                state_list.append(normX(sprite.rect.centerx-ship_x+bullet_w))
                state_list.append(normY(sprite.rect.centery-ship_y-bullet_h))
                state_list.append(normY(sprite.rect.centery-ship_y+bullet_h))
            else:
                state_list.extend([0,0,0,0,0])
    
        state_list.append(normS(SPACESHIP_BULLET_SPEED))            
        state_list.append(self.spaceship.ammunition/100)            
        state_list.append(self.level)                               
        state_list.append(len(self.spaceship_Group))                             
        
        return torch.tensor(state_list, dtype=torch.float32)
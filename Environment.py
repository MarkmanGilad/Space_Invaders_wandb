
import pygame
import numpy as np
import torch
from CONSTANTS import *
from SpaceShip import SpaceShip
from Enemy import Enemy, Explosion
from Ground import Ground

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
        self.die = False

    def init_rewards (self):
        self.end_of_game = -1
        self.end_of_stage = 2
        self.hit = 1
        self.amunition = -0.01
        self.enemy_above = -0.00
        self.delta = 7.5  # width of spaceship / 2

    def make_enemy_group (self, row=ENEMY_ROWS, col=ENEMY_COLS, space_row = 80, space_col = 120, speed = ENEMY_START_SPEED):
        enemy_Group = pygame.sprite.Group()
        row , col = 3 , 6
        for r in range (row):
            for c in range (col):
                enemy_Group.add(Enemy(self.enemy_img, (c * space_col, r * space_row, ), self.enemy_bullets_Group,speed=speed))
        # all_enemies = [(x, y) for x in range(4) for y in range(7)]
        # sample_enemies = random.sample(all_enemies, 1)
        # for r, c in sample_enemies:
        #     enemy_Group.add(Enemy(self.enemy_img, (c * space_col, r * space_row, ), self.enemy_bullets_Group,speed=speed))
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

        if self.next_stage:
            self.level += 1
            Enemy.shoots_factor += self.add_shoot_factor
            self.enemy_Group = self.make_enemy_group(speed= int(ENEMY_START_SPEED + self.level/2))
            self.spaceship.rect.midbottom = (width, HEIGHT - 100)
            self.next_stage = False
        else:
            self.spaceship = SpaceShip((WIDTH //2, HEIGHT - 100), self.bullets_Group)
            self.spaceship_Group = pygame.sprite.GroupSingle(self.spaceship)
            self.spaceship.rect.midbottom = (width, HEIGHT - 100)
            self.die = False
            Enemy.shoots_factor = ENEMY_SHOOTS_FACTOR
            self.score = 0
            self.level = 1
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
            if self.spaceship.ammunition > 0:
                reward += self.amunition              # don't waste ammunition
        self.update()
        self.draw()        
        hits = self.hits()
        reward +=  hits * self.hit
        self.score += hits
        if self.is_end_of_stage():
            reward += self.end_of_stage
            self.next_stage = True
            self.restart()
            return reward, False
        done, die = self.is_end_of_Game()
        if done:
            return reward, True
        if die:
            reward += self.end_of_game
            return reward, False
        if self.is_enemy_missile_above():
            reward += self.enemy_above
        return reward, False
    
    def is_enemy_missile_above(self):
        SpaceShip_x = self.spaceship.rect.centerx
        delta = self.delta
        is_bullet_above = any(abs(sprite.rect.centerx - SpaceShip_x) <= delta for sprite in self.enemy_bullets_Group)
        return is_bullet_above

    def is_end_of_stage (self):
        enemies = len(self.enemy_Group)
        return enemies == 0
   
    def is_end_of_Game (self):
        if self.die:
            done = len(self.spaceship_Group)==0 and len(self.Explosion_Group) == 0
            return done, self.die

        if self.spaceship.ammunition == 0 and len(self.enemy_Group) > 0 and len(self.bullets_Group)==0:      
            self.die = True
        else:
            enemy_landed = pygame.sprite.spritecollide(self.ground, self.enemy_Group, dokill=True)
            spaceship_hit = pygame.sprite.spritecollide(self.spaceship, self.enemy_bullets_Group, dokill=True, collided= pygame.sprite.collide_mask) 
            self.die = len(enemy_landed) > 0 or len(spaceship_hit) > 0
        if self.die:
            explosion = Explosion(self.spaceship.rect.topleft)
            self.Explosion_Group.add(explosion)
            self.spaceship.kill()

        return False, self.die
        
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
        number_of_enemies = 1                                   # 1
        enemy_ships = ENEMY_COLS * ENEMY_ROWS                   # 3 * 6 * 6 = 108  exists, x-width, x+width,  y-height, y+height, speed 
        enemy_speed_y = 1                                       # 1
        enemy_bullets = MAX_ENEMY_BULLETS                       # 10 * 5 = 50     exists, x-width, x+width,  y-height, y+height
        enemy_bullet_speed_y = 1                                # 1
        SpaceShip_pos_shape = 4                                 # 4               -width, + width, -Height, + height
        SpaceShip_speed_x = 1                                   # 1
        SpaceShip_Bullet_pos_shape = SPACE_SHIP_BURST           # 3 * 5 = 15       exists, x-width, x+width,  y-height, y+height
        SpaceShip_bullets_speed_y = 1                           # 1
        SpaceShip_ammunition = 1                                # 1
        level = 1                                               # 1
        # score = 1                                             
        total = enemy_ships + enemy_speed_y + enemy_bullets + enemy_bullet_speed_y + SpaceShip_pos_shape + SpaceShip_speed_x + \
        SpaceShip_Bullet_pos_shape + SpaceShip_bullets_speed_y + SpaceShip_ammunition + level 
        # total = 183

        ship_x = self.spaceship.rect.centerx
        ship_y = self.spaceship.rect.centery
        ship_w = 15 / 2
        ship_h = 34 / 2
        enemy_w = 34 / 2
        enemy_h = 30 / 2
        bullet_w = 5 / 2
        bullet_h = 5 / 2

        state_list = []
        state_list.append(len(self.enemy_Group)/enemy_ships)
        for sprite in self.enemy_Group:
            state_list.append(sprite.live)
            state_list.append(normX(sprite.rect.centerx-ship_x-enemy_w))
            state_list.append(normX(sprite.rect.centerx-ship_x+enemy_w))
            state_list.append(normY(sprite.rect.centery-ship_y-enemy_h))
            state_list.append(normY(sprite.rect.centery-ship_y+enemy_h))
            state_list.append(normS(sprite.speed_x))
        
        for i in range(enemy_ships-len(self.enemy_Group)):
            state_list.extend([0,0,0,0,0,0])
        state_list.append(normS(Enemy.speed_y))                     
        
        for sprite in self.enemy_bullets_Group:                     
            state_list.append(1)
            state_list.append(normX(sprite.rect.centerx-ship_x-bullet_w))
            state_list.append(normX(sprite.rect.centerx-ship_x+bullet_w))
            state_list.append(normY(sprite.rect.centery-ship_y-bullet_h))
            state_list.append(normY(sprite.rect.centerx-ship_x+bullet_h))
        
        for i in range(enemy_bullets-len(self.enemy_bullets_Group)):
            state_list.extend([0,0,0,0,0])
        state_list.append(normS(ENEMY_BULLET_SPEED))               
        state_list.append(normX(-ship_w))                               #      width of space ship to the right
        state_list.append(normX(ship_w))                               #      width of space ship to the right
        state_list.append(normY(-ship_h))                               #      width of space ship to the right
        state_list.append(normY(ship_h))                               #      width of space ship to the right
        state_list.append(normS(SPACESHIP_SPEED))                  
        
        for sprite in self.bullets_Group:                          
            state_list.append(1)
            state_list.append(normX(sprite.rect.centerx-ship_x-bullet_w))
            state_list.append(normX(sprite.rect.centerx-ship_x+bullet_w))
            state_list.append(normY(sprite.rect.centery-ship_y-bullet_h))
            state_list.append(normY(sprite.rect.centery-ship_y+bullet_h))

        for i in range(SpaceShip_Bullet_pos_shape-len(self.bullets_Group)):
            state_list.extend([0,0,0,0,0])
        state_list.append(normS(SPACESHIP_BULLET_SPEED))            
        state_list.append(self.spaceship.ammunition/100)            
        state_list.append(self.level)                               
        state_list.append(len(self.spaceship_Group))                             
        return torch.tensor(state_list, dtype=torch.float32)
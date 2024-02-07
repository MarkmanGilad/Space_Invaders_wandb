import pygame
from CONSTANTS import *
from Bullet import Bullet

class SpaceShip (pygame.sprite.Sprite):
    def __init__(self, img_Url, pos, bullets_Group) -> None:
        super().__init__()
        self.image = pygame.image.load(img_Url)
        self.image = pygame.transform.scale(self.image, (60, 60))
        self.rect = self.image.get_rect(midbottom = pos)
        self.mask = pygame.mask.from_surface(self.image)
        self.bullets_Group = bullets_Group
        self.speed_x = SPACESHIP_SPEED
        self.ammunition = MAX_AMMUNITION
        self.burst = SPACE_SHIP_BURST

    def update(self) -> None:
        pass

    def move_right (self):
        self.rect.x += self.speed_x
        if self.rect.right > WIDTH + 20:
            self.rect.right = WIDTH + 20 

    def move_left (self):
        self.rect.x -= self.speed_x
        if self.rect.left < 0 - 20:
            self.rect.left = 0 - 20 

    def shoot (self):
        if self.ammunition > 0 and len(self.bullets_Group) < self.burst:
            self.bullets_Group.add(Bullet(self.rect.midtop,speed_y= -SPACESHIP_BULLET_SPEED))
            self.ammunition -= 1

    def action (self, action):
        if action == 1:
            self.move_left()
        elif action == 2:
            self.move_right()
        elif action == 3:
            self.shoot()
        

        

import pygame

class Human_Agent:
    def __init__ (self):
        self.action = 0

    def get_Action (self, events=None, state= None):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.action = 1
                if event.key == pygame.K_RIGHT:
                    self.action = 2
                if event.key == pygame.K_SPACE:
                    self.action = 3
            if event.type == pygame.KEYUP:
                self.action = 0
        if self.action == 3: # ירי בבודדת
            self.action = 0
            return 3
        return self.action
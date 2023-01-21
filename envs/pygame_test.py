import pygame

background_colour = (139,0,0)
screen = pygame.display.set_mode((600, 300))
pygame.display.set_caption('GuidingCode.com')
screen.fill(background_colour)
pygame.display.flip()
running = True
while running:
   for event in pygame.event.get():
       if event.type == pygame.QUIT:
           running = False
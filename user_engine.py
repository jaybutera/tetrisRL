import pygame
import time
from engine import TetrisEngine

def play_game():
    last_tick = pygame.time.get_ticks()
    refresh_rate = 300
    print(engine)

    done = False
    # Global action
    action = 6

    while not done:
        action = 6
        events = pygame.event.get()
        for event in events:
            print(event)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT or event.key == pygame.K_l:
                    action = 0 # Shift left
                if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    action = 1 # Shift right
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    action = 2 # Hard drop
                if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    action = 3 # Soft drop
                if event.key == pygame.K_q:
                    action = 4 # Rotate left
                if event.key == pygame.K_e:
                    action = 5 # Rotate right

        # Game update
        now = pygame.time.get_ticks()
        if now - last_tick > refresh_rate:
            print(action)
            last_tick = now
            # Game step
            state, reward, done = engine.step(action)
            # Update render
            #print(engine)

def play_again():
    print('Play Again? [y/n]')
    print('> ')
    choice = input()

    return True if choice.lower == 'y' else False

if __name__ == '__main__':
    pygame.init()
    pygame.display.iconify()

    # Init environment
    width, height = 10, 20 # standard tetris friends rules
    engine = TetrisEngine(width, height)

    # Play games on repeat
    while True:
        engine.clear()
        play_game()

        # Prompt to play again
        if not play_again():
            print('Thanks for contributing!')
            break

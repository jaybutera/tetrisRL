import curses
import numpy as np
import os
from engine import TetrisEngine

def play_game():
    # Store play information
    db = []
    # Initial rendering
    stdscr.addstr(str(env))

    done = False
    # Global action
    action = 6

    while not done:
        action = 6
        key = stdscr.getch()

        if key == -1: # No key pressed
            action = 6
        elif key == ord('a'): # Shift left
            action = 0
        elif key == ord('d'): # Shift right
            action = 1
        elif key == ord('w'): # Hard drop
            action = 2
        elif key == ord('s'): # Soft drop
            action = 3
        elif key == ord('q'): # Rotate left
            action = 4
        elif key == ord('e'): # Rotate right
            action = 5

        # Game step
        state, reward, done = env.step(action)
        db.append((state,reward,done,action))

        # Render
        stdscr.clear()
        stdscr.addstr(str(env))
        stdscr.addstr('reward: ' + str(reward))

    return db

def play_again():
    #stdscr.addstr('Play Again? [y/n]')
    print('Play Again? [y/n]')
    print('> ', end='')
    #stdscr.addstr('> ')
    choice = input()
    #choice = stdscr.getch()

    return True if choice.lower() == 'y' else False

def save_game():
    print('Accumulated reward: {0} | {1} moves'.format(sum([i[1] for i in db]), len(db)))
    print('Would you like to store the game info as training data? [y/n]')
    #stdscr.addstr('Would you like to store the game info as training data? [y/n]\n')
    #stdscr.addstr('> ')
    print('> ', end='')
    choice = input()
    return True if choice.lower() == 'y' else False

def terminate():
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()

def init():
    # Don't display user input
    curses.noecho()
    # React to keys without pressing enter (700ms delay)
    curses.halfdelay(7)
    # Enumerate keys
    stdscr.keypad(True)

    #return stdscr

if __name__ == '__main__':
    # Curses standard screen
    stdscr = curses.initscr()

    # Init environment
    width, height = 10, 20 # standard tetris friends rules
    env = TetrisEngine(width, height)

    # Play games on repeat
    while True:
        init()
        stdscr.clear()
        env.clear()
        db = play_game()

        # Return to terminal
        terminate()
        # Should the game info be saved?
        if save_game():
            try:
                fr = open('training_data.npy', 'rb')
                x = np.load(fr)
                fr.close()
                fw = open('training_data.npy', 'wb')
                x = np.concatenate((x,db))
                #print('Saving {0} moves...'.format(len(db)))
                np.save(fw, x)
                print('{0} data points in the training set'.format(len(x)))
            except Exception as e:
                print('no training file exists. Creating one now...')
                fw = open('training_data.npy', 'wb')
                print('Saving {0} moves...'.format(len(db)))
                np.save(fw, db)
        # Prompt to play again
        if not play_again():
            print('Thanks for contributing!')
            break

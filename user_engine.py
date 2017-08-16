import curses
import numpy as np
from engine import TetrisEngine

def play_game():
    # Store play information
    db = []
    # Initial rendering
    #stdscr.addstr(str(env))
    print(env)

    done = False
    # Global action
    action = 6

    while not done:
        action = 6
        #key = stdscr.getch()
        key = input()

        if key == -1: # No key pressed
            action = 6
        elif key == ord('a'):
            action = 0
        elif key == ord('d'):
            action = 1
        elif key == ord('w'):
            action = 2
        elif key == ord('s'):
            action = 3
        elif key == ord('q'):
            action = 4
        elif key == ord('e'):
            action = 5

        # Game step
        state, reward, done = env.step(action)
        db.append((state,reward,done,action))

        # Render
        print(env)
        #stdscr.clear()
        #stdscr.addstr(str(env))

    return db

def play_again(stdscr):
    stdscr.addstr('Play Again? [y/n]')
    stdscr.addstr('> ', end='')
    #choice = input()
    choice = stdscr.getch()

    return True if choice.lower() == 'y' else False

def save_game(stdscr):
    #print('Would you like to store the game info as training data? [y/n]')
    stdscr.addstr('Would you like to store the game info as training data? [y/n]')
    stdscr.addstr('> ', end='')
    choice = input()
    return True if choice.lower() == 'y' else False

def terminate():
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()

def init():
    #stdscr = curses.initscr()
    # Don't display user input
    curses.noecho()
    # React to keys without pressing enter (700ms delay)
    #curses.cbreak()
    curses.halfdelay(7)
    # Enumerate keys
    stdscr.keypad(True)

    #return stdscr

if __name__ == '__main__':
    #stdscr = curses.initscr()
    #init(stdscr)

    # Init environment
    width, height = 10, 20 # standard tetris friends rules
    env = TetrisEngine(width, height)

    # Play games on repeat
    #while True:
    for i in range(2):
        #stdscr = curses.initscr()
        #init()
        env.clear()
        db = play_game()

        # Return to terminal
        #terminate()
        # Should the game info be saved?
        '''
        if save_game():
            with open('training_data.npy', 'ab') as f:
                print('Saving {0} moves...'.format(len(db)))
                np.save(f, db)
                print('Saved!')
        # Prompt to play again
        if not play_again():
            print('Thanks for contributing!')
            terminate(stdscr)
            break
        '''
        #stdscr = init()

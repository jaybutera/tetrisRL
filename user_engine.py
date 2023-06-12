import curses
import numpy as np
import os
from engine import TetrisEngine

def play_game():
    # Store play information
    db = []
    '''
    states = []
    rewards = []
    done_flags = []
    actions = []
    '''
    # Initial rendering
    stdscr.addstr(str(env))

    reward_sum = 0
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
        reward_sum += reward
        db.append(np.array((state,reward,done,action)))
        '''
        states.append(state)
        rewards.append(reward)
        done_flags.append(done)
        actions.append(action)
        '''

        # Render
        stdscr.clear()
        stdscr.addstr(str(env))
        stdscr.addstr('\ncumulative reward: ' + str(reward_sum))
        stdscr.addstr('\nreward: ' + str(reward))

    '''
    db = {
        'states' : np.array(states),
        'rewards' : np.array(rewards),
        'done_flags' : np.array(done_flags),
        'actions' : np.array(actions),
    }
    '''
    return db

def play_again():
    print('Play Again? [y/n]')
    print('> ', end='')
    choice = input()

    return True if choice.lower() == 'y' else False

def prompt_save_game():
    #print('Accumulated reward: {0} | {1} moves'.format(sum(db['rewards']), db['actions'].shape[0]))
    print('Accumulated reward: {0} | {1} moves'.format(sum([i[1] for i in db]), len(db)))
    print('Would you like to store the game info as training data? [y/n]')
    #stdscr.addstr('Would you like to store the game info as training data? [y/n]\n')
    #stdscr.addstr('> ')
    print('> ', end='')
    choice = input()
    return True if choice.lower() == 'y' else False

def save_game(path='training_data.npy'):
    if os.path.exists(path):
        x = np.load(path, allow_pickle=True)
        x = np.concatenate((x,db))
        np.save(path, x)
        print('{0} data points in the training set'.format(len(x)))
    else:
        print('no training file exists. Creating one now...')
        #fw = open('training_data.npy', 'wb')
        print('Saving {0} moves...'.format(len(db)))
        np.save(path, db)

def terminate():
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()
    os.system("stty sane")

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
        if prompt_save_game():
            save_game()
        # Prompt to play again
        if not play_again():
            print('Thanks for contributing!')
            break

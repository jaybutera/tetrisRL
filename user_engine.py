import curses
import time
from engine import TetrisEngine

def play_game(stdscr):
    refresh_rate = 200
    #print(engine)
    stdscr.addstr(str(engine))

    done = False
    # Global action
    action = 6

    while not done:
        action = 6
        key = stdscr.getch()

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
        state, reward, done = engine.step(action)

        # Render
        stdscr.clear()
        stdscr.addstr(str(engine))

def play_again():
    print('Play Again? [y/n]')
    print('> ')
    choice = input()

    return True if choice.lower == 'y' else False

def terminate(stdscr):
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()

def init():
    stdscr = curses.initscr()
    # Don't display user input
    curses.noecho()
    # React to keys without pressing enter (300ms delay)
    #curses.cbreak()
    curses.halfdelay(7)
    # Enumerate keys
    stdscr.keypad(True)

    return stdscr

if __name__ == '__main__':
    stdscr = init()

    # Init environment
    width, height = 10, 20 # standard tetris friends rules
    engine = TetrisEngine(width, height)

    # Play games on repeat
    while True:
        engine.clear()
        play_game(stdscr)

        # Prompt to play again
        terminate(stdscr)
        if not play_again():
            print('Thanks for contributing!')
            break
        stdscr = init()

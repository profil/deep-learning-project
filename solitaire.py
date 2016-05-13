#!/usr/bin/env python2
import pygame
from os import path
from random import shuffle
from pygame.locals import *

CARDWIDTH = 73
CARDHEIGHT = 97
OFFSET = 16
MARGIN = 4
WIDTH = (MARGIN + CARDWIDTH) * 7 - MARGIN
HEIGHT = (7 + 14) * OFFSET + 2 * (MARGIN + CARDHEIGHT) - 3
CURSOR_COLOR = pygame.Color(0, 0, 255)
CURSOR_SELECTED_COLOR = pygame.Color(255, 0, 255)

class Card:
    def __init__(self, suit, value, hidden=True):
        self.suit = suit
        self.value = value
        self.hidden = hidden

    def __repr__(self):
        return "%s %s Hidden: %s" % (self.suit, self.value, self.hidden)

class Deck:
    def __init__(self):
        cards = []
        for suit in range(4):
            for value in range(1, 14):
                cards.append(Card(suit, value))
        shuffle(cards)

        self.rows = [[] for i in range(7)]
        for i in range(7):
            for j in range(i, 7):
                self.rows[j].append(cards.pop(0))

        for i in self.rows:
            i[-1].hidden = False

        self.deck = cards
        self.showing = []
        self.showed = []

    def deal(self):
        cards = self.deck[:3]
        del self.deck[:3]

        self.showing = cards
        if cards:
            self.showed.extend(self.showing)
        else:
            self.deck = self.showed
            self.showed = []

    def cards_in_stack(self, stack):
        return len(self.rows[stack])

class Cursor:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cards = 1

class Selecter:
    def __init__(self):
        self.from_stack = None
        self.cards = 0
        self.selected = False

class Solitaire:
    def __init__(self, screen, cards, backside, bottom):
        self.screen = screen
        self.cards = cards
        self.backside = backside
        self.bottom = bottom
        self.cursor = Cursor(0, 1)
        self.selector = Selecter()
        self.reset()

    def cards_in_stack(self):
        return self.deck.cards_in_stack(self.cursor.x)

    def draw(self):
        self.screen.blit(self.backside, (0, 0))
        if self.deck.showing:
            x = 0
            for c in self.deck.showing:
                self.screen.blit(self.cards[c.suit][c.value - 1], ((MARGIN + CARDWIDTH) + x, 0))
                x += OFFSET
        else:
            self.screen.blit(self.bottom, (MARGIN + CARDWIDTH, 0))

        for i in range(3, 7):
            self.screen.blit(self.bottom, ((MARGIN + CARDWIDTH) * i, 0))

        for i, r in enumerate(self.deck.rows):
            y = 0
            for c in r:
                card = self.backside if c.hidden else self.cards[c.suit][c.value - 1]
                self.screen.blit(card, ((MARGIN + CARDWIDTH) * i, (2 * MARGIN + CARDHEIGHT) + y))
                y += OFFSET

        self.draw_cursor()

    def move_right(self):
        if self.cursor.x == 0 and self.cursor.y == 0 and not self.deck.showing:
            self.cursor.x = 3
        elif self.cursor.x == 1 and self.cursor.y == 0:
            self.cursor.x = 3
        else:
            self.cursor.x = min(self.cursor.x + 1, 6)
        self.cursor.cards = 1

    def move_left(self):
        if self.cursor.x == 3 and self.cursor.y == 0 and not self.deck.showing:
            self.cursor.x = 0
        elif self.cursor.x == 3 and self.cursor.y == 0:
            self.cursor.x = 1
        else:
            self.cursor.x = max(self.cursor.x - 1, 0)
            self.cursor.cards = 1

    def move_down(self):
        if self.cursor.y == 1:
            self.cursor.cards = max(self.cursor.cards - 1, 1)
        self.cursor.y = 1

    def move_up(self):
        if (not self.selector.selected and
            self.cursor.cards < self.cards_in_stack() and self.cursor.y == 1):
            self.cursor.cards += 1
        else:
            if not self.deck.showing and (self.cursor.x == 1 or self.cursor.x == 2):
                self.cursor.x = 0
            elif self.cursor.x == 2: # The gap between the main deck and the four goal piles
                self.cursor.x = 1
            self.cursor.y = 0
            self.cursor.cards = 1

    def select(self):
        #TODO: Restrictions and make it work for not only between stacks
        if self.selector.selected:
            selected_cards = self.deck.rows[self.selector.from_stack][-self.selector.cards:]
            self.deck.rows[self.cursor.x] = self.deck.rows[self.cursor.x] + selected_cards
            del self.deck.rows[self.selector.from_stack][-self.selector.cards:]
        else:
            self.selector.from_stack = self.cursor.x
            self.selector.cards = self.cursor.cards

        self.selector.selected = not self.selector.selected

    def reset(self):
        self.deck = Deck()

    def draw_cursor(self):
        y = self.cursor.y * (2 * MARGIN + CARDHEIGHT)
        if self.cursor.y == 1:
            y += OFFSET * max(self.cards_in_stack() - self.cursor.cards, 0)

        x = self.cursor.x * (MARGIN + CARDWIDTH)
        if self.cursor.y == 0 and self.cursor.x == 1:
            x += OFFSET * 2

        pygame.draw.rect(self.screen,
                         CURSOR_SELECTED_COLOR if self.selector.selected else CURSOR_COLOR,
                         pygame.Rect(x,
                                     y,
                                     CARDWIDTH,
                                     CARDHEIGHT + OFFSET * (self.cursor.cards - 1)),
                         2)

def init_game():
    cards = [[pygame.image.load(path.join('cards', '{0:02d}'.format(value) + suit + ".gif"))
            for value in range(1, 14)]
            for suit in ['c', 'd', 'h', 's']]
    backside = pygame.image.load(path.join('cards', 'back192.gif'))
    bottom = pygame.image.load(path.join('cards', 'bottom01-n.gif'))
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((0, 130, 0))

    solitaire = Solitaire(background, cards, backside, bottom)
    solitaire.draw()

    screen.blit(background, (0, 0))
    pygame.display.update()

    while 1:
        event = pygame.event.wait()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                solitaire.move_left()
            elif event.key == pygame.K_RIGHT:
                solitaire.move_right()
            elif event.key == pygame.K_UP:
                solitaire.move_up()
            elif event.key == pygame.K_DOWN:
                solitaire.move_down()
            elif event.key == pygame.K_SPACE:
                solitaire.select()
            background.fill((0, 130, 0))

            solitaire.draw()

            screen.blit(background, (0, 0))
            pygame.display.update()
        elif event.type == pygame.QUIT:
            pygame.quit()
            exit()

def main():
    init_game()

if __name__ == '__main__':
    main()

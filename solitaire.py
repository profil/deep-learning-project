#!/usr/bin/env python2
import pygame
from os import path
from random import shuffle

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

class Solitaire:
    def __init__(self, screen, cards, backside, bottom):
        self.screen = screen
        self.cards = cards
        self.backside = backside
        self.bottom = bottom
        self.cursor = (0, 0)
        self.selected = False
        self.reset()

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

        pygame.draw.rect(self.screen,
                         CURSOR_COLOR,
                         pygame.Rect((MARGIN + CARDWIDTH) * 3,
                                     OFFSET + (2 * MARGIN + CARDHEIGHT),
                                     CARDWIDTH,
                                     CARDHEIGHT + 2 * OFFSET),
                         2)

    def reset(self):
        self.deck = Deck()

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
    pygame.time.delay(1000)

    for i in range(2):
        background.fill((0, 130, 0))

        solitaire.deck.deal()
        solitaire.draw()

        screen.blit(background, (0, 0))
        pygame.display.update()
        pygame.time.delay(2000)

def main():
    init_game()

if __name__ == '__main__':
    main()

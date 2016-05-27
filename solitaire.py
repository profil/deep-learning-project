#!/usr/bin/env python2
import pygame
from os import path
from random import shuffle
from pygame.locals import *
import numpy as np

CARDWIDTH = 73
CARDHEIGHT = 97
OFFSET = 16
MARGIN = 4
WIDTH = (MARGIN + CARDWIDTH) * 7 - MARGIN
HEIGHT = (7 + 14) * OFFSET + 2 * (MARGIN + CARDHEIGHT) - 3
CURSOR_COLOR = pygame.Color(0, 0, 255)
CURSOR_SELECTED_COLOR = pygame.Color(255, 0, 255)
SIZE = 100

class Card:
    def __init__(self, suit, value, hidden=True):
        self.suit = suit
        self.value = value
        self.hidden = hidden

    def __repr__(self):
        return "%s %s Hidden: %s" % (self.suit, self.value, self.hidden)

class RowStack:
    def __init__(self):
        self.cards = []

    def add(self, cards):
        if self.accept(cards[0]):
            for c in cards: c.hidden = False
            self.cards.extend(cards)
            return True

    def accept(self, card):
        last = self.cards[-1] if len(self.cards) > 0 else None
        return (not last and card.value == 13 or
                last and
                not last.hidden and
                last.suit % 2 != card.suit % 2 and
                last.value == card.value+1)

class GoalStack:
    def __init__(self):
        self.cards = []
        self.suit = None

    def add(self, card):
        if self.accept(card):
            card.hidden = False
            self.cards.append(card)
            return True

    def accept(self, card):
        if not self.suit and card.value == 1:
            self.suit = card.suit
            return True
        else:
            last = self.cards[-1] if len(self.cards) > 0 else None
            return (not last and
                    card.value == 1 or
                    last and
                    last.suit == card.suit and
                    last.value == card.value-1)

class Deck:
    def __init__(self):
        cards = []
        for suit in range(4):
            for value in range(1, 14):
                cards.append(Card(suit, value))
        shuffle(cards)

        self.rows = [RowStack() for _ in range(7)]
        for i in range(7):
            for j in range(i, 7):
                self.rows[j].cards.append(cards.pop(0))

        for row in self.rows:
            row.cards[-1].hidden = False

        self.goals = [GoalStack() for _ in range(4)]

        self.deck = cards
        self.showing = []
        self.showed = []

    def deal(self):
        cards = self.deck[:3]
        del self.deck[:3]

        if self.showing:
            self.showed.extend(self.showing)

        self.showing = cards
        if not cards:
            self.deck = self.showed
            self.showed = []

    def stack(self, stack):
        return self.rows[stack]

class Cursor:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.nCards = 1

class Selected:
    def __init__(self, x, y, nCards, cards):
        self.x = x
        self.y = y
        self.nCards = nCards
        self.cards = cards

class Solitaire:
    def __init__(self, size=SIZE):
        self.cards = [[pygame.image.load(path.join('cards', '{0:02d}'.format(value) + suit + ".gif"))
                for value in range(1, 14)]
                for suit in ['c', 'd', 's', 'h']]
        self.backside = pygame.image.load(path.join('cards', 'back192.gif'))
        self.bottom = pygame.image.load(path.join('cards', 'bottom01-n.gif'))
        pygame.init()
        self.screen = pygame.display.set_mode((size, size))

        self.reset()

        background = pygame.Surface((WIDTH, HEIGHT))
        self.background = background.convert()
        self.step()


    def draw(self):
        self.background.blit(self.backside, (0, 0))
        if self.deck.showing:
            x = 0
            for c in self.deck.showing:
                self.background.blit(self.cards[c.suit][c.value - 1], ((MARGIN + CARDWIDTH) + x, 0))
                x += OFFSET
        else:
            self.background.blit(self.bottom, (MARGIN + CARDWIDTH, 0))

        for i, s in enumerate(self.deck.goals):
            if s.cards:
                card = s.cards[-1]
                img = self.cards[card.suit][card.value - 1]
            else:
                img = self.bottom
            self.background.blit(img, ((MARGIN + CARDWIDTH) * (i + 3), 0))

        for i, r in enumerate(self.deck.rows):
            y = 0
            for c in r.cards:
                card = self.backside if c.hidden else self.cards[c.suit][c.value - 1]
                self.background.blit(card, ((MARGIN + CARDWIDTH) * i, (2 * MARGIN + CARDHEIGHT) + y))
                y += OFFSET

        if(self.selected): self.draw_cursor(self.selected, CURSOR_SELECTED_COLOR)
        self.draw_cursor(self.cursor)

    def right(self):
        if self.cursor.x == 0 and self.cursor.y == 0 and not self.deck.showing:
            self.cursor.x = 3
        elif self.cursor.x == 1 and self.cursor.y == 0:
            self.cursor.x = 3
        else:
            self.cursor.x = min(self.cursor.x + 1, 6)
        self.cursor.nCards = 1

    def left(self):
        if self.cursor.x == 3 and self.cursor.y == 0 and not self.deck.showing:
            self.cursor.x = 0
        elif self.cursor.x == 3 and self.cursor.y == 0:
            self.cursor.x = 1
        else:
            self.cursor.x = max(self.cursor.x - 1, 0)
            self.cursor.nCards = 1

    def down(self):
        if self.cursor.y == 1:
            self.cursor.nCards = max(self.cursor.nCards - 1, 1)
        self.cursor.y = 1

    def up(self):
        if (not self.selected and
            self.cursor.nCards < len(self.deck.stack(self.cursor.x).cards) and
            self.cursor.y == 1 and
            not self.deck.stack(self.cursor.x).cards[-(self.cursor.nCards + 1)].hidden):
            self.cursor.nCards += 1
        else:
            if not self.deck.showing and (self.cursor.x == 1 or self.cursor.x == 2):
                self.cursor.x = 0
            elif self.cursor.x == 2: # The gap between the main deck and the four goal piles
                self.cursor.x = 1
            self.cursor.y = 0
            self.cursor.nCards = 1

    def select(self):
        if self.cursor.y == 0:
            if self.cursor.x == 0:
                self.deck.deal()
                self.selected = None
                return
            elif self.cursor.x == 1:
                self.selected = Selected(1, 0, 1, self.deck.showing)
                return

        if self.selected:
            selected_cards = self.selected.cards[-self.selected.nCards:]
            if self.cursor.y == 0 and len(selected_cards) == 1:
                if self.deck.goals[self.cursor.x - 3].add(selected_cards[0]):
                    del self.selected.cards[-self.selected.nCards:]
                    if self.selected.cards != self.deck.goals[self.selected.x - 3].cards:
                        self.score += 10
            else:
                if self.deck.rows[self.cursor.x].add(selected_cards):
                    del self.selected.cards[-self.selected.nCards:]
                    if self.selected.cards and self.selected.cards == self.deck.showing:
                        self.score += 5
                    elif self.selected.cards and self.selected.cards == self.deck.goals[self.selected.x - 3].cards:
                        self.score -= 15
                        if self.score < 0: self.score = 0

            self.selected = None
        else:
            cards = (self.deck.goals[self.cursor.x - 3].cards
                if self.cursor.y == 0
                else self.deck.rows[self.cursor.x].cards)
            if cards:
                if cards[-1].hidden:
                    cards[-1].hidden = False
                    self.score += 5
                else:
                    self.selected = Selected(self.cursor.x,
                                             self.cursor.y,
                                             self.cursor.nCards,
                                             cards)

    def reset(self):
        self.cursor = Cursor(0, 0)
        self.selected = None
        self.score = 0
        self.deck = Deck()
        self.last_move = ()
        self.last_score = 0
        self.fails = 0

    def draw_cursor(self, cursor, color=CURSOR_COLOR):
        y = cursor.y * (2 * MARGIN + CARDHEIGHT)
        if cursor.y == 1:
            y += OFFSET * max(len(self.deck.stack(cursor.x).cards) - cursor.nCards, 0)

        x = cursor.x * (MARGIN + CARDWIDTH)
        if cursor.y == 0 and cursor.x == 1:
            x += OFFSET * (len(self.deck.showing) - 1)

        pygame.draw.rect(self.background,
                         color,
                         pygame.Rect(x,
                                     y,
                                     CARDWIDTH,
                                     CARDHEIGHT + OFFSET * (cursor.nCards - 1)),
                         5)
    def step(self, action=None):
        score = self.score
        terminal = False

        if self.fails >= 20:
            terminal = True
        elif not action:
            pass
        elif action == 'up':
            self.up()
        elif action == 'down':
            self.down()
        elif action == 'left':
            self.left()
        elif action == 'right':
            self.right()
        elif action == 'select':
            self.select()

        self.background.fill((0, 130, 0))
        self.draw()
        asd = pygame.transform.smoothscale(self.background, (SIZE, SIZE))
        self.screen.blit(asd, (0, 0))
        pygame.display.flip()

        image = pygame.surfarray.array3d(self.screen)
        return np.transpose(image, (1, 0, 2)), self.score-score, terminal

    def play(self):
        while 1:
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.left()
                elif event.key == pygame.K_RIGHT:
                    self.right()
                elif event.key == pygame.K_UP:
                    self.up()
                elif event.key == pygame.K_DOWN:
                    self.down()
                elif event.key == pygame.K_SPACE:
                    self.select()
                elif event.key == pygame.K_RETURN:
                    s = self.bot()
                    if s == 'left':
                        self.left()
                    elif s == 'right':
                        self.right()
                    elif s == 'down':
                        self.down()
                    elif s == 'up':
                        self.up()
                    elif s == 'select':
                        self.select()
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()

            self.background.fill((0, 130, 0))
            self.draw()
            self.screen.blit(self.background, (0, 0))
            pygame.display.flip()

    def bot(self):
        if self.selected:
            if self.selected.nCards == 1: # Move selected card to goal
                card = self.selected.cards[-self.selected.nCards]
                for i, g in enumerate(self.deck.goals):
                    if g.accept(card):
                        return self.botmove(3 + i, 0 , 1)
                for i, r in enumerate(self.deck.rows):
                    if r.accept(card):
                        return self.botmove(i, 1, 1)
            else: # Move selected cards to other stack
                card = self.selected.cards[-self.selected.nCards]
                for i, r in enumerate(self.deck.rows):
                    if r.accept(card):
                        return self.botmove(i, 1, 1)
        else:
            for i, r in enumerate(self.deck.rows):
                if self.deck.rows[i].cards:
                    if self.deck.rows[i].cards[-1].hidden:
                        return self.botmove(i, 1, 1)
            for i, r in enumerate(self.deck.rows):
                for j, g in enumerate(self.deck.goals):
                    if self.deck.rows[i].cards:
                        card = self.deck.rows[i].cards[-1]
                        if g.accept(card):
                            return self.botmove(i, 1, 1)
            if self.deck.showing:
                card = self.deck.showing[-1]
                for i, r in enumerate(self.deck.rows):
                    if r.accept(card):
                        return self.botmove(1, 0, 1)
                for i, g in enumerate(self.deck.goals):
                    if g.accept(card):
                        return self.botmove(1, 0, 1)
            for i, r in enumerate(self.deck.rows):
                if self.deck.rows[i].cards:
                    for n, c in enumerate(self.deck.rows[i].cards):
                        if not c.hidden:
                            for j, rr in enumerate(self.deck.rows):
                                if j == i: continue
                                card = self.deck.rows[i].cards[n]
                                if rr.accept(card):
                                    nCards = len(self.deck.rows[i].cards) - n
                                    if self.last_move == (i, 1):
                                        continue
                                    return self.botmove(i, 1, nCards)
        return self.botmove(0, 0, 1)

    def botmove(self, x, y, nCards):
        if self.cursor.y < y:
            return 'down'
        elif self.cursor.y > y:
            return 'up'
        else:
            if self.cursor.x < x:
                return 'right'
            elif self.cursor.x > x:
                return 'left'
            else:
                if self.cursor.nCards < nCards:
                    return 'up'
                else:
                    self.last_move = (x, y)
                    if self.last_score == self.score:
                        self.fails += 1
                    else:
                        self.fails = 0
                    self.last_score = self.score
                    return 'select'
def main():
    sol = Solitaire(WIDTH)
    sol.play()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import randrange
from random import choice
from typing import List
from random import shuffle
from copy import deepcopy
import numpy as np
import math


class Genome:

    def __init__(self, coins: List[int], change: int):
        self.coins = coins                          # list of available coins
        self.change = change                        # change to be found
        self.coinBits = self.setCoins()             # number of bits to code a coin
        self.genome = self.generateIndividual()     # list of bits coding genome

    def setCoins(self) -> List[int]:
        coinBits = [0]
        coinSum = 0
        for coin in self.coins:
            numOfCoins = math.floor(self.change / coin)
            if numOfCoins != 0:
                coinSum += math.floor(math.log(numOfCoins, 2)) + 1
            coinBits.append(coinSum)
        return coinBits

    def generateIndividual(self) -> List[int]:
        return np.random.randint(2, size=self.coinBits[-1]).tolist()

    def fitnessFunction(self):
        i = 0
        sumOfCoins = 0
        coinsCount = 0
        for coin in self.coins:
            bitNum = self.genome[self.coinBits[i]:self.coinBits[i + 1]]
            i += 1

            numOfCoins = 0
            for bit in bitNum:
                numOfCoins += (numOfCoins << 1) | bit
            sumOfCoins += numOfCoins * coin
            coinsCount += numOfCoins

        return abs(sumOfCoins - self.change) + 1.25*coinsCount

    def crossOver(self, g: "Genome"):
        half = math.floor(len(self.genome) / 2)
        tmp = (self.genome[:half] + g.genome[half:])
        g.genome = tmp
        return g

    def mutate(self, mutationProbability: int):
        for i in range(0, len(self.genome)):
            if np.random.uniform(0, 1) < mutationProbability:
                self.genome[i] = (self.genome[i] + 1) % 2

    def getCoinSet(self):
        i = 0
        ret = []
        for coin in self.coins:
            bitNum = self.genome[self.coinBits[i]:self.coinBits[i + 1]]
            i += 1

            numOfCoins = 0
            for bit in bitNum:
                numOfCoins += (numOfCoins << 1) | bit
            ret.append([coin, numOfCoins])
        return ret


class Simulation:

    def __init__(self, coins: List[int], change: int, populationSize: int, mutationProbability: float,
                 maxIterations: int):
        self.populationSize = populationSize
        self.mutationProbability = mutationProbability
        self.maxIterations = maxIterations
        self.population = self.generatePopulation(coins, change)

    def generatePopulation(self, coins: List[int], change: int) -> List['Genome']:
        return [Genome(coins, change) for _ in range(self.populationSize)]

    def reproduce(self) -> List[List[int]]:
        populationCopy = deepcopy(self.population)
        offspring = []

        # Cross Over
        for i in range(0, len(populationCopy), 2):
            offspring.append(populationCopy[i].crossOver(populationCopy[i + 1]))
            offspring.append(populationCopy[i + 1].crossOver(populationCopy[i]))

        # Mutate
        for generation in offspring:
            generation.mutate(self.mutationProbability)

        # Pick Best
        newPop = self.population + offspring
        newPop = sorted(newPop, key=lambda item: item.fitnessFunction())
        self.population = newPop[:self.populationSize]
        shuffle(self.population)

    def evolve(self):
        for i in range(self.maxIterations):
            self.reproduce()

    def getBest(self):
        sortedPopulation = sorted(self.population, key=lambda item: item.fitnessFunction())
        return sortedPopulation[0].getCoinSet()


def main():
    coins = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    change = 5839
    print("Coins set", coins)
    print("Change", change)

    totalCoins = 0
    totalChange = 0
    reps = 10
    for _ in range(reps):
        populacja = Simulation(coins, change, populationSize=20, mutationProbability=0.05, maxIterations=500)
        populacja.evolve()
        best = populacja.getBest()
        numOfCoins = 0
        foundChange = 0
        for [a,b] in best:
            numOfCoins += b
            foundChange += a*b
        totalCoins += numOfCoins
        totalChange += foundChange

    print("AVG change", totalChange/reps)
    print("AVG coins", totalCoins/reps)



if __name__ == '__main__':
    main()

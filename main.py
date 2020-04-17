#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import randrange
from typing import List
import numpy as np
import math
from random import shuffle
from copy import deepcopy


class Genome:

    def __init__(self, coins: List[int], change: int):
        self.coins = coins
        self.change = change
        self.coinBits = self.setCoins()
        self.genome = self.generateIndividual()

    def setCoins(self) -> List[int]:
        coinBits = [0]
        coinSum = 0
        for coin in self.coins:
            numOfCoins = math.floor(self.change / coin)
            coinSum += math.floor(math.log(numOfCoins, 2)) + 1
            coinBits.append(coinSum)  # number of bits to code a coin
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

        return abs(sumOfCoins - self.change) + coinsCount*0.01

    def crossOver(self, g: "Genome"):
        half = math.floor(len(self.genome) / 2)
        tmp = (self.genome[:half] + g.genome[half:])
        g.genome = tmp
        return g

    def mutate(self, mutationProbability: int):
        for i in range(0, len(self.genome)):
            if np.random.uniform(0, 1) < mutationProbability:
                self.genome[i] = (self.genome[i] + 1) % 2

    def printCoinSet(self):
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

    def printAll(self):
        print("Genome: ----------")
        print(self.coins)
        print(self.genome)
        print(self.fitnessFunction())
        print(self.printCoinSet())
        print("------------------")

    def printResult(self):
        print(self.printCoinSet());
        print("Uzyto", int(self.fitnessFunction()*100 % 100), "monet.")


class Simulation:

    def __init__(self, coins: List[int], change: int, populationSize: int, mutationProbability: float,
                 maxIterations: int):
        self.populationSize = populationSize
        self.mutationProbability = mutationProbability
        self.maxIterations = maxIterations
        self.population = self.generatePopulation(coins, change)

    def generatePopulation(self, coins: List[int], change: int) -> List["Genome"]:
        population = []
        for i in range(0, self.populationSize):
            population.append(Genome(coins, change))
        return population

    def reproduce(self) -> List[List[int]]:
        populationCopy = deepcopy(self.population)
        offspring = []

        # Cross Over
        for i in range(0, len(populationCopy), 2):
            offspring.append(populationCopy[i].crossOver(populationCopy[i + 1]))
            offspring.append(populationCopy[i+1].crossOver(populationCopy[i]))

        # Mutate
        for generation in offspring:
            generation.mutate(self.mutationProbability)

        # Pick Best
        newPop = self.population + offspring
        newPop = sorted(newPop, key=lambda item: item.fitnessFunction())
        self.population = newPop[:self.populationSize]
        shuffle(self.population)

    def evolve(self):
        for i in range(0, self.maxIterations):
            if i % (self.maxIterations/5) == 0:
                self.mutationProbability += 0.05
            self.reproduce()

    def printAll(self):
        for genome in sorted(self.population, key=lambda item: item.fitnessFunction()):
            genome.printAll()

    def printBest(self):
        sortedPopulation = sorted(self.population, key=lambda item: item.fitnessFunction())
        print("BEST:")
        sortedPopulation[0].printResult()

def main():
    coins = [1, 2, 5, 10, 20, 50]
    change = 90
    print("Coins set", coins)
    print("Change", change)

    populacja1 = Simulation(coins, change, populationSize=20, mutationProbability=0.05, maxIterations=500)
    populacja1.evolve()
    populacja1.printBest()


if __name__ == '__main__':
    main()

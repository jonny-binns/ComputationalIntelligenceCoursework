package coursework;

import java.util.ArrayList;
import java.lang.Math;

import model.Fitness;
import model.Individual;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;

/**
 * Implements a basic Evolutionary Algorithm to train a Neural Network
 * 
 * You Can Use This Class to implement your EA or implement your own class that extends {@link NeuralNetwork} 
 * 
 */
public class ExampleEvolutionaryAlgorithm extends NeuralNetwork {
	

	/**
	 * The Main Evolutionary Loop
	 */
	@Override
	public void run() {		
		//Initialise a population of Individuals with random weights
		population = initialise();
		//System.out.println("population size = " + population.size());

		//Record a copy of the best Individual in the population
		best = getBest();
		System.out.println("Best From Initialisation " + best);

		/**
		 * main EA processing loop
		 */		
		
		while (evaluations < Parameters.maxEvaluations) {

			/**
			 * this is a skeleton EA - you need to add the methods.
			 * You can also change the EA if you want 
			 * You must set the best Individual at the end of a run
			 * 
			 */

			// Select 2 Individuals from the current population. Currently returns random Individual
			Individual parent1 = select(); 
			Individual parent2 = select();

			// Generate a child by crossover. Not Implemented			
			ArrayList<Individual> children = reproduce(parent1, parent2);			
			
			//mutate the offspring
			mutate(children);
			
			// Evaluate the children
			evaluateIndividuals(children);			

			// Replace children in population
			replace(children);

			// check to see if the best has improved
			best = getBest();
			
			// Implemented in NN class. 
			outputStats();
			
			//Increment number of completed generations			
		}

		//save the trained network to disk
		saveNeuralNetwork();
	}

	

	/**
	 * Sets the fitness of the individuals passed as parameters (whole population)
	 * 
	 */
	private void evaluateIndividuals(ArrayList<Individual> individuals) {
		for (Individual individual : individuals) {
			individual.fitness = Fitness.evaluate(individual, this);
		}
	}


	/**
	 * Returns a copy of the best individual in the population
	 * 
	 */
	private Individual getBest() {
		best = null;;
		for (Individual individual : population) {
			if (best == null) {
				best = individual.copy();
			} else if (individual.fitness < best.fitness) {
				best = individual.copy();
			}
		}
		return best;
	}

	/**
	 * Generates a randomly initialised population
	 * 
	 */
	private ArrayList<Individual> initialise() {
		population = new ArrayList<>();
		for (int i = 0; i < Parameters.popSize; ++i) {
			//chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			population.add(individual);
		}
		evaluateIndividuals(population);
		return population;
	}

	/**
	 * Selection --
	 * 
	 * NEEDS REPLACED with proper selection this just returns a copy of a random
	 * member of the population
	 */
	private Individual select() {	
		//original code
		//Individual parent = population.get(Parameters.random.nextInt(Parameters.popSize));
		//return parent.copy();
		
		//take in tournament size from parameters
		int tSize = 25;
		
		//create list of potential winners
		Individual[] potentialParents = new Individual[tSize];
		
		//for i<tournament size
			//randomly select tournament size individuals from population and add to an array of potential parents
		for(int i=0; i<tSize; i++)
		{
			
			int randInt = Parameters.random.nextInt(Parameters.popSize);
			//System.out.println("RandInt = " + randInt);
			potentialParents[i] = population.get(randInt);
		}
		
		
		//initialize the best fitness potential parents[0]
		double bestFitness = potentialParents[0].fitness;
		
		//init chosen parent
		Individual chosenParent = potentialParents[0];
		
		//for i<tournament size
			//if fitness(potential parent[i]) < best fitness
				//best fitness = fitness 
				//chosen parent = potential parent[i]
		for(int i=0; i<tSize; i++)
		{
			if(potentialParents[i].fitness < bestFitness)
			{
				bestFitness = potentialParents[i].fitness;
				chosenParent = potentialParents[i];
			}
		}
		
		
		//return chosen parent
		return chosenParent;
		
	}

	/**
	 * Crossover / Reproduction
	 * 
	 * NEEDS REPLACED with proper method this code just returns exact copies of the
	 * parents. 
	 */
	private ArrayList<Individual> reproduce(Individual parent1, Individual parent2) {
		
		//original code
		/*
		ArrayList<Individual> children = new ArrayList<>();
		children.add(parent1.copy());
		children.add(parent2.copy());			
		return children;
		*/
		
		/*
		//take in how many points to crossover from parents
		int noCuts = 1;
		
		//pick cut point/s
		//pick random number/s from the chromosome.length
		int cutPoint = (int) (0 + (Math.random() * parent1.chromosome.length));
		
		
		//create empty child array
		double childChromArr[] = new double[parent1.chromosome.length];
		
		
		//genes from parent one
		//for(i<chromosome.length())
			//alternate through cut points and add genes from parents
		
		//for 1pt crossover need to add for more than 1 point
		for(int i=0; i<cutPoint; i++)
		{
			childChromArr[i] = parent1.chromosome[i];
		}
		
		for(int i=cutPoint; i<parent1.chromosome.length; i++)
		{
			childChromArr[i] = parent2.chromosome[i];
		}
		
		ArrayList<Individual> children = new ArrayList<Individual>();
		Individual child = new Individual();
		children.add(child);
		evaluateIndividuals(children);
		children.get(0).chromosome = childChromArr;
		
		return children;
		*/
		
		//uniform crossover
		//potentially add so parameters decide how many children
		double childChromArr[] = new double[parent1.chromosome.length];
		
		for(int i=0; i<parent1.chromosome.length; i++)
		{
			int randInt = Parameters.random.nextInt(2);
			//System.out.println("randInt = " + randInt);
			if(randInt == 0)
			{
				childChromArr[i] = parent1.chromosome[i];
			}
			if(randInt == 1)
			{
				childChromArr[i] = parent2.chromosome[i];
			}
		}
		
		ArrayList<Individual> children = new ArrayList<Individual>();
		Individual child = new Individual();
		child.chromosome = childChromArr;
		children.add(child);
		return children;
		
	} 
	
	/**
	 * Mutation
	 * 
	 * 
	 */
	private void mutate(ArrayList<Individual> individuals) {		
		for(Individual individual : individuals) {
			for (int i = 0; i < individual.chromosome.length; i++) {
				if (Parameters.random.nextDouble() < Parameters.mutateRate) {
					if (Parameters.random.nextBoolean()) {
						individual.chromosome[i] += (Parameters.mutateChange);
					} else {
						individual.chromosome[i] -= (Parameters.mutateChange);
					}
				}
			}
		}		
	}

	/**
	 * 
	 * Replaces the worst member of the population 
	 * (regardless of fitness)
	 * 
	 */
	private void replace(ArrayList<Individual> individuals) {
		//original code
		/*
		for(Individual individual : individuals) {
			int idx = getWorstIndex();		
			population.set(idx, individual);
		}
		*/
		/*
		//replace worst fitness
		//init worst
		double worstFitness = population.get(0).fitness;
		int worstIndex = 0;
		
		for(int i=0; i<population.size(); i++)
		{
			if(population.get(i).fitness > worstFitness)
			{
				worstIndex = i;
			}
		}
		
		population.set(worstIndex, individuals.get(0));
		*/
		
		
		//replace using tournament
		//get tournament size
		int tSize = 15;
		
		//make list of candidates
		Individual[] replaceCandidates = new Individual[tSize];
		
		//select tSize number of individuals from population
		for(int i=0; i<tSize; i++)
		{
			
			int randInt = Parameters.random.nextInt(Parameters.popSize);
			//System.out.println("RandInt = " + randInt);
			replaceCandidates[i] = population.get(randInt);
		}
		
		//init the worst fitness
		double worstFitness = replaceCandidates[0].fitness;
		Individual worst = replaceCandidates[0];
		
		//loop through and find the worst fitness
		for(int i=0; i<replaceCandidates.length; i++)
		{
			if(replaceCandidates[i].fitness > worstFitness)
			{
				worstFitness = replaceCandidates[i].fitness;
				worst = replaceCandidates[i];
			}
		}
		
		//replace worst fitness in population
		for(int i=0; i<population.size(); i++)
		{
			if(population.get(i) == worst)
			{
				population.set(i, individuals.get(0));
			}
		}
		
	}

	

	/**
	 * Returns the index of the worst member of the population
	 * @return
	 */
	private int getWorstIndex() {
		Individual worst = null;
		int idx = -1;
		for (int i = 0; i < population.size(); i++) {
			Individual individual = population.get(i);
			if (worst == null) {
				worst = individual;
				idx = i;
			} else if (individual.fitness > worst.fitness) {
				worst = individual;
				idx = i; 
			}
		}
		return idx;
	}	

	@Override
	public double activationFunction(double x) {
		if (x < -20.0) {
			return -1.0;
		} else if (x > 20.0) {
			return 1.0;
		}
		return Math.tanh(x);
	}
}

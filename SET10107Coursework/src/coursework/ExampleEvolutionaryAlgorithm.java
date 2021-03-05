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
			//replace(children);
			//ReplaceRandomParent requires the parents as parameters
			ReplaceRandomParent(children, parent1, parent2);

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
		/*
		//original code
		Individual parent = population.get(Parameters.random.nextInt(Parameters.popSize));
		return parent.copy();
		*/
		
		Individual individual = TournamentSelect();
		return individual;

		
	}
	
	/**
	 * Tournament Selection
	 * randomly selects tSize number of individuals and finds the fittest individual from that group
	 */
	private Individual TournamentSelect() {
		//take in tournament size from parameters
		int tSize = Parameters.selectTSize;
		
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
		
		//1pt crossover
		//ArrayList<Individual> children = PointCrossover(parent1, parent2);
		//return children;
		
		//uniform crossover
		//ArrayList<Individual> children = UniformCrossover(parent1, parent2);
		//return children;
		
		//arithmetic crossover
		ArrayList<Individual> children = ArithmeticCrossover(parent1, parent2);
		return children;
		
	} 
	
	/**
	 * n pt crossover
	 * chooses a set number of cut point then alternates between parents as to where the childs chromosomes are taken from
	 */
	//change to n point crossover, if someone wants only 1 point then change the parameters
	private ArrayList<Individual> PointCrossover(Individual parent1, Individual parent2) {
		//get no of children from parameters
		int childNo = Parameters.noOfChildren;
		
		//get no of cuts from parameters
		int cutNo = Parameters.cutPoints;
		
		//make children list
		ArrayList<Individual> children = new ArrayList<Individual>();
		
		//for each child
		for(int i=0; i<childNo; i++)
		{
			//make cutpoint list
			ArrayList<Integer> cutPoints = new ArrayList<>();
			for(int j=0; j<cutNo; j++)
			{
				//get cut positions
				cutPoints.add((int) (0 + (Math.random() * parent1.chromosome.length)));
			}
		
			//create chromosome array
			double childChromArr[] = new double[parent1.chromosome.length];
			
			for(int j=0; j<cutPoints.size(); j++)
			{
				for(int k=0; k<parent1.chromosome.length; k++)
				{
					if(i%2==0)
					{
						if(k<cutPoints.get(j))
						{
							childChromArr[k] = parent1.chromosome[k];
						}
					}
					if(i%2!=0)
					{
						if(k<cutPoints.get(j))
						{
							childChromArr[k] = parent2.chromosome[k];
						}
					}
				}
			}
		
			//add child to list
			Individual child = new Individual();
			child.chromosome = childChromArr;
			children.add(child);
		}
		
		//evaluate then return children list
		evaluateIndividuals(children);
		return children;
	}
	
	/**
	 * uniform crossover
	 * for each chromosome randomly choose whether it comes from parent 1 or 2
	 */
	private ArrayList<Individual> UniformCrossover(Individual parent1, Individual parent2) {
		//get number of children
		int childNo = Parameters.noOfChildren;
		
		//make return List
		ArrayList<Individual> children = new ArrayList<Individual>();
		
		//for each child
		for(int i=0; i<childNo; i++)
		{
			//make chromosome array
			double childChromArr[] = new double[parent1.chromosome.length];
			
			//for each chromosome
			for(int j=0; j<parent1.chromosome.length; j++)
			{
				//generate randint
				int randInt = Parameters.random.nextInt(2);
				//get chromosome from corresponding parent
				if(randInt == 0)
				{
					childChromArr[j] = parent1.chromosome[j];
				}
				if(randInt == 1)
				{
					childChromArr[j] = parent2.chromosome[j];
				}
			}
			
			//make child, set chromosome and add to children list
			Individual child = new Individual();
			child.chromosome = childChromArr;
			children.add(child);
		}
			
		
		//evaluate/return list
		evaluateIndividuals(children);
		return children;
	}
	
	/**
	 * Arithmetic crossover
	 * Takes the average of the parents chromosomes for the child
	 * (will only give one set of values so only one child will be created)
	 */
	private ArrayList<Individual> ArithmeticCrossover(Individual parent1, Individual parent2) {
		//create child list
		ArrayList<Individual> children = new ArrayList<Individual>();
		
		//make chromosome array
		double childChromArr[] = new double[parent1.chromosome.length];
		
		//loop through parents chromosomes and average the value
		for(int i=0; i<parent1.chromosome.length; i++)
		{
			childChromArr[i] = ((parent1.chromosome[i] + parent2.chromosome[i])/2);
		}
		
		//set childChromArr to child
		Individual child = new Individual();
		child.chromosome = childChromArr;
		
		//add child to list
		children.add(child);
		
		//evaluate/return list
		evaluateIndividuals(children);
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
	
		//replace worst fitness
		//ReplaceWorst(individuals);
		
		
		//replace using tournament
		//TournamentReplace(individuals);

		//replace random parent (called in run())
		
		//replace random member of population
		//ReplaceRandom(individuals);
		
		
	}

	/**
	 * replace worst
	 * replaces the individual with the worst fitness
	 */
	private void ReplaceWorst(ArrayList<Individual> individuals) {
		//for individuals.length (length of children)
		for(int i=0; i<individuals.size(); i++)
		{
			//init worst
			double worstFitness = population.get(0).fitness;
			int worstIndex = 0;
			
			//loop through population and find the individual with the worst fitness
			for(int j=0; j<population.size(); j++)
			{
				if(population.get(j).fitness > worstFitness)
				{
					worstIndex = j;
				}
			}
	
			//replace with child(i)
			population.set(worstIndex, individuals.get(i));
		}
	}
	
	/**
	 * tournament replace
	 * randomly selects tSize individuals and replaceses the least fit from that selection
	 */
	private void TournamentReplace(ArrayList<Individual> individuals) {	
		//get tournament size
		int tSize = Parameters.replacementTSize;
		
		for(int i=0; i<individuals.size(); i++)
		{
			//make list of candidates
			Individual[] replaceCandidates = new Individual[tSize];
			
			//select tSize number of individuals from population
			for(int j=0; j<tSize; j++)
			{
				
				int randInt = Parameters.random.nextInt(Parameters.popSize);
				replaceCandidates[j] = population.get(randInt);
			}
			
			//init the worst fitness
			double worstFitness = replaceCandidates[0].fitness;
			Individual worst = replaceCandidates[0];
			
			//loop through candidates and find the worst fitness
			for(int j=0; j<replaceCandidates.length; j++)
			{
				if(replaceCandidates[j].fitness > worstFitness)
				{
					worstFitness = replaceCandidates[j].fitness;
					worst = replaceCandidates[j];
				}
			}
			
			//replace individual with worst fitness in the population with child(i)
			for(int j=0; j<population.size(); j++)
			{
				if(population.get(j) == worst)
				{
					population.set(j, individuals.get(i));
				}
			}
		}
	}
	
	/**
	 * Replace random parent
	 * replaces a random parent of the child in question
	 * Will only work for one child, if there are 2 or more children parents may be erased more than once
	 */
	private void ReplaceRandomParent(ArrayList<Individual> individuals, Individual parent1, Individual parent2) {
		
		//generate random int to decide which parent will be replaced
		int randInt = Parameters.random.nextInt(2);
			
		//find parent in population
		//get position in population
		
		//position of parent in the population
		int parentPos = 0;
		
		for(int i=0; i<population.size(); i++)
		{
			if(randInt == 0)
			{
				//get parent 1
				if(population.get(i) == parent1)
				{
					parentPos = i;
				}
			}
			if(randInt == 1)
			{
				//get parent 2
				if(population.get(i) == parent2)
				{
					parentPos = i;
				}
			}
		}
		
		//replace parent in population
		population.set(parentPos, individuals.get(0));
	}
	
	/**
	 * replace random individual
	 * replaces a random individual in the population with the children
	 */
	private void ReplaceRandom(ArrayList<Individual> individuals) {
		
		for(int i=0; i<individuals.size(); i++)
		{
			//generate random position to get replaced
			int randPos = Parameters.random.nextInt(population.size());
			
			//replace individual at that position with child
			population.set(randPos, individuals.get(i));
		}
	}
	
	
	/**
	 * replace oldest
	 * dont use population.set, instead remove position 0 and add child at the end
	 */
	
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

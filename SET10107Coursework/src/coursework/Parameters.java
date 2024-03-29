package coursework;

import java.lang.reflect.Field;
import java.util.Random;
import model.LunarParameters;
import model.NeuralNetwork;
import model.LunarParameters.DataSet;

public class Parameters {
 
	/**
	 * These parameter values can be changed 
	 * You may add other Parameters as required to this class 
	 * 
	 */
	//baseline parameters
	private static int numHidden = 5;	
	private static int numGenes = calculateNumGenes();
	public static double minGene = -3; // specifies minimum and maximum weight values 
	public static double maxGene = +3;
	
	public static int popSize = 40;
	public static int maxEvaluations = 20000;
	
	//parameters for selection
	public static int selectTSize = 4;
	
	//parameters for crossover/reproduction
	public static int noOfChildren = 2; //arithmetic crossover will only give 1 child, replace random parent will work with a max of 2
	public static int cutPoints = 2; //if you want to do 1 point crossover set this to 1
	
	// Parameters for mutation 
	// Rate = probability of changing a gene
	// Change = the maximum +/- adjustment to the gene value
	public static double mutateRate = 0.04; // mutation rate for mutation operator
	public static double mutateChange = 0.1; // delta change for mutation operator
	
	//parameters for replacement
	public static int replacementTSize = 4; //allows for tournament selection and replacement to have different sizes
	
	//Random number generator used throughout the application
	public static long seed = System.currentTimeMillis();
	public static Random random = new Random(seed);

	//set the NeuralNetwork class here to use your code from the GUI
	//public static Class neuralNetworkClass = ExampleHillClimber.class;
	public static Class neuralNetworkClass = ExampleEvolutionaryAlgorithm.class;
	
	/**
	 //1-pt parameters
	private static int numHidden = 5;	
	private static int numGenes = calculateNumGenes();
	public static double minGene = -3; // specifies minimum and maximum weight values 
	public static double maxGene = +3;
	
	public static int popSize = 40;
	public static int maxEvaluations = 20000;
	
	//parameters for selection
	public static int selectTSize = 10;
	
	//parameters for crossover/reproduction
	public static int noOfChildren = 6; //arithmetic crossover will only give 1 child, replace random parent will work with a max of 2
	public static int cutPoints = 1; //if you want to do 1 point crossover set this to 1
	
	// Parameters for mutation 
	// Rate = probability of changing a gene
	// Change = the maximum +/- adjustment to the gene value
	public static double mutateRate = 0.07; // mutation rate for mutation operator
	public static double mutateChange = 0.1; // delta change for mutation operator
	
	//parameters for replacement
	public static int replacementTSize = 4; //allows for tournament selection and replacement to have different sizes
	
	//Random number generator used throughout the application
	public static long seed = System.currentTimeMillis();
	public static Random random = new Random(seed);

	//set the NeuralNetwork class here to use your code from the GUI
	//public static Class neuralNetworkClass = ExampleHillClimber.class;
	public static Class neuralNetworkClass = ExampleEvolutionaryAlgorithm.class;
	 */
	
	/**
	 //2-pt parameters
	 private static int numHidden = 5;	
	private static int numGenes = calculateNumGenes();
	public static double minGene = -3; // specifies minimum and maximum weight values 
	public static double maxGene = +3;
	
	public static int popSize = 40;
	public static int maxEvaluations = 20000;
	
	//parameters for selection
	public static int selectTSize = 10;
	
	//parameters for crossover/reproduction
	public static int noOfChildren = 4; //arithmetic crossover will only give 1 child, replace random parent will work with a max of 2
	public static int cutPoints = 2; //if you want to do 1 point crossover set this to 1
	
	// Parameters for mutation 
	// Rate = probability of changing a gene
	// Change = the maximum +/- adjustment to the gene value
	public static double mutateRate = 0.06; // mutation rate for mutation operator
	public static double mutateChange = 0.1; // delta change for mutation operator
	
	//parameters for replacement
	public static int replacementTSize = 8; //allows for tournament selection and replacement to have different sizes
	
	//Random number generator used throughout the application
	public static long seed = System.currentTimeMillis();
	public static Random random = new Random(seed);

	//set the NeuralNetwork class here to use your code from the GUI
	//public static Class neuralNetworkClass = ExampleHillClimber.class;
	public static Class neuralNetworkClass = ExampleEvolutionaryAlgorithm.class;
	 */
	
	/**
	 //arithmetic parameters
	 	private static int numHidden = 5;	
	private static int numGenes = calculateNumGenes();
	public static double minGene = -3; // specifies minimum and maximum weight values 
	public static double maxGene = +3;
	
	public static int popSize = 40;
	public static int maxEvaluations = 20000;
	
	//parameters for selection
	public static int selectTSize = 11;
	
	//parameters for crossover/reproduction
	public static int noOfChildren = 2; //arithmetic crossover will only give 1 child, replace random parent will work with a max of 2
	public static int cutPoints = 2; //if you want to do 1 point crossover set this to 1
	
	// Parameters for mutation 
	// Rate = probability of changing a gene
	// Change = the maximum +/- adjustment to the gene value
	public static double mutateRate = 0.1; // mutation rate for mutation operator
	public static double mutateChange = 0.1; // delta change for mutation operator
	
	//parameters for replacement
	public static int replacementTSize = 8; //allows for tournament selection and replacement to have different sizes
	
	//Random number generator used throughout the application
	public static long seed = System.currentTimeMillis();
	public static Random random = new Random(seed);

	//set the NeuralNetwork class here to use your code from the GUI
	//public static Class neuralNetworkClass = ExampleHillClimber.class;
	public static Class neuralNetworkClass = ExampleEvolutionaryAlgorithm.class;
	 */
	
	/**
	 //uniform parameters
	 	private static int numHidden = 5;	
	private static int numGenes = calculateNumGenes();
	public static double minGene = -3; // specifies minimum and maximum weight values 
	public static double maxGene = +3;
	
	public static int popSize = 40;
	public static int maxEvaluations = 20000;
	
	//parameters for selection
	public static int selectTSize = 8;
	
	//parameters for crossover/reproduction
	public static int noOfChildren = 5; //arithmetic crossover will only give 1 child, replace random parent will work with a max of 2
	public static int cutPoints = 2; //if you want to do 1 point crossover set this to 1
	
	// Parameters for mutation 
	// Rate = probability of changing a gene
	// Change = the maximum +/- adjustment to the gene value
	public static double mutateRate = 0.09; // mutation rate for mutation operator
	public static double mutateChange = 0.1; // delta change for mutation operator
	
	//parameters for replacement
	public static int replacementTSize = 20; //allows for tournament selection and replacement to have different sizes
	
	//Random number generator used throughout the application
	public static long seed = System.currentTimeMillis();
	public static Random random = new Random(seed);

	//set the NeuralNetwork class here to use your code from the GUI
	//public static Class neuralNetworkClass = ExampleHillClimber.class;
	public static Class neuralNetworkClass = ExampleEvolutionaryAlgorithm.class;
	 */
	
	/**
	 * Do not change any methods that appear below here.
	 * 
	 */
	
	public static int getNumGenes() {					
		return numGenes;
	}

	
	private static int calculateNumGenes() {
		int num = (NeuralNetwork.numInput * numHidden) + (numHidden * NeuralNetwork.numOutput) + numHidden + NeuralNetwork.numOutput;
		return num;
	}

	public static int getNumHidden() {
		return numHidden;
	}
	
	public static void setHidden(int nHidden) {
		numHidden = nHidden;
		numGenes = calculateNumGenes();		
	}

	public static String printParams() {
		String str = "";
		for(Field field : Parameters.class.getDeclaredFields()) {
			String name = field.getName();
			Object val = null;
			try {
				val = field.get(null);
			} catch (IllegalArgumentException | IllegalAccessException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			str += name + " \t" + val + "\r\n";
			
		}
		return str;
	}
	
	public static void setDataSet(DataSet dataSet) {
		LunarParameters.setDataSet(dataSet);
	}
	
	public static DataSet getDataSet() {
		return LunarParameters.getDataSet();
	}
	
	public static void main(String[] args) {
		printParams();
	}
}

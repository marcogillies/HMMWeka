/**
 * 
 */
package weka.estimators;

import java.util.Random;

import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.matrix.DoubleVector;
//import weka.estimators.Estimator;

/**
 * @author marco gillies
 *
 */
public interface HMMEstimator extends RevisionHandler {

	public int getNumStates();

	public void setNumStates(int NumStates);

	public int getNumOutputs();

	public void setNumOutputs(int NumOutputs) throws Exception;

	public int getOutputDimension();
	
	//public void setOutputDimension(int OutputDimension) throws Exception;	
	/**
	   * Add a new time step value to the current estimator.
	   *
	   * @param prev_state the previous HMM state 
	   * @param state the current HMM state
	   * @param output the HMM output  
	   * @param weight the weight assigned to the data value 
	   */
	  void addValue(double prev_state, double state, DoubleVector output, double weight);
	  
	/**
	   * Convenience function for univariate outputs.
	   *
	   * @param prev_state the previous HMM state 
	   * @param state the current HMM state
	   * @param output the HMM output  
	   * @param weight the weight assigned to the data value 
	   */
	  void addValue(double prev_state, double state, double output, double weight) throws Exception;
	  
	  /**
	   * Add a new data value for the first time step to the current estimator.
	   *
	   * @param state the current HMM state
	   * @param output the HMM output  
	   * @param weight the weight assigned to the data value 
	   */
	  void addValue0(double state, DoubleVector output, double weight);
	  
	  /**
	   * Convenience function for univariate outputs.
	   *
	   * @param state the current HMM state
	   * @param output the HMM output  
	   * @param weight the weight assigned to the data value 
	   */
	  void addValue0(double state, double output, double weight) throws Exception;


	  /**
	   * Get a probability for a time step value
	   *
	   * @param prev_state the previous HMM state 
	   * @param state the current HMM state
	   * @param output the HMM output  
	 * @throws Exception 
	   */
	  double getProbability(double prev_state, double state, DoubleVector output) throws Exception;


	  /**
	   * Convenience function for univariate outputs.
	   *
	   * @param prev_state the previous HMM state 
	   * @param state the current HMM state
	   * @param output the HMM output  
	   */
	  double getProbability(double prev_state, double state, double output) throws Exception;


	  /**
	   * Get a probability for a first time step value
	   *
	   * @param state the current HMM state
	   * @param output the HMM output  
	 * @throws Exception 
	   */
	  double getProbability0(double state, DoubleVector output) throws Exception;


	  /**
	   * Convenience function for univariate outputs.
	   *
	   * @param state the current HMM state
	   * @param output the HMM output  
	   */
	  double getProbability0(double state, double output) throws Exception;
	  
	  int Sample0(Instances sequence, Random generator);
	  
	  int Sample(Instances sequence, int prevState, Random generator);
	  
	  void calculateParameters() throws Exception;
}

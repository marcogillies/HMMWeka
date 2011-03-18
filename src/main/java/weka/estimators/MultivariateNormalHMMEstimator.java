package weka.estimators;

import java.io.Serializable;
import java.util.Random;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.DoubleVector;
import weka.core.matrix.Matrix;

public class MultivariateNormalHMMEstimator extends AbstractHMMEstimator
		implements HMMEstimator, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -1123497102759147327L;
	protected boolean m_Tied = true;

	public boolean isTied() {
		return m_Tied;
	}

	public void setTied(boolean tied) {
		m_Tied = tied;
	}

	protected int m_CovarianceType = MultivariateNormalEstimator.COVARIANCE_FULL;
	
	public int getCovarianceType() {
		return m_CovarianceType;
	}

	public void setCovarianceType(int covarianceType) {
		this.m_CovarianceType = covarianceType;
		if(m_outputEstimators != null)
		{
			for (int s = 0; s < getNumStates(); s++)
			{
				m_outputEstimators[s].setCovarianceType(getCovarianceType());
			}
		}
	}
	
	@Override
	public int getOutputDimension() {
		// TODO Auto-generated method stub
		return m_outputEstimators[0].getDimension();
	}

	protected MultivariateNormalEstimator m_outputEstimators[];


	public MultivariateNormalHMMEstimator() {
		super();
	}
	
	public MultivariateNormalHMMEstimator(int numStates, boolean laplace) {
		super(numStates, laplace);
		
		m_outputEstimators = new MultivariateNormalEstimator[numStates];
		for (int s = 0; s < numStates; s++)
		{
			m_outputEstimators[s] = new MultivariateNormalEstimator();
			m_outputEstimators[s].setCovarianceType(getCovarianceType());
		}
	}
	
	public MultivariateNormalHMMEstimator(MultivariateNormalHMMEstimator a) throws Exception
	{
		super(a);
		
		m_outputEstimators = new MultivariateNormalEstimator[a.getNumStates()];
		for (int i = 0; i < m_outputEstimators.length; i++)
			m_outputEstimators[i] = new MultivariateNormalEstimator(a.m_outputEstimators[i]);
	}
	
	public void copyOutputParameters(MultivariateNormalHMMEstimator a) throws Exception
	{
		setCovarianceType(a.getCovarianceType());
		setTied(a.isTied());
		m_outputEstimators = new MultivariateNormalEstimator[a.getNumStates()];
		for (int i = 0; i < m_outputEstimators.length; i++)
			m_outputEstimators[i] = new MultivariateNormalEstimator(a.m_outputEstimators[i]);
	}
	
	@Override
	public void setNumStates(int NumStates) {
		super.setNumStates(NumStates);
		m_outputEstimators = new MultivariateNormalEstimator[m_NumStates];
		for (int s = 0; s < m_NumStates; s++)
		{
			m_outputEstimators[s] = new MultivariateNormalEstimator();
			m_outputEstimators[s].setCovarianceType(getCovarianceType());
		}
	}
	
	public void setState0Probabilities(double probs[])
	{
		for (int i = 0; i < probs.length; i++)
			m_state0Estimator.addValue(i, probs[i]);
	}
	
	public void setStateProbabilities(double probs[][])
	{
		for (int ps = 0; ps < probs.length; ps++)
			for (int s = 0; s < probs[ps].length; s++)
				m_stateEstimators[ps].addValue(s, probs[ps][s]);
	}

	public void setOutputMeans(DoubleVector means[])
	{
		for (int i = 0; i < means.length; i++)
			m_outputEstimators[i].setMean(means[i]);
	}
	
	public void setOutputMean(int state, DoubleVector mean)
	{
		m_outputEstimators[state].setMean(mean);
	}

	public void setOutputVariances(Matrix vars[])
	{
		for (int i = 0; i < vars.length; i++)
			m_outputEstimators[i].setVariance(vars[i]);
	}

	public void setOutputVariance(int state, Matrix var)
	{
		m_outputEstimators[state].setVariance(var);
	}
	
	@Override
	public int Sample(Instances sequence, int prevState, Random generator) {
		int state;
		DoubleVector output;
		
		do {
			state = generator.nextInt(getNumStates());
		} while (generator.nextDouble() > m_stateEstimators[prevState].getProbability(state));
		
		output = m_outputEstimators[state].sample();
		
		sequence.add(new DenseInstance(output.size()));
		Instance frame = sequence.lastInstance();
		
		for(int i = 0; i < output.size(); i++)
			frame.setValue(i, output.get(i));
		
		return state;
	}

	@Override
	public int Sample0(Instances sequence, Random generator) {
		int state;
		DoubleVector output;
		
		do {
			state = generator.nextInt(getNumStates());
			//System.out.println("state "+ state + " prob " + m_state0Estimator.getProbability(state));
		} while (generator.nextDouble() > m_state0Estimator.getProbability(state));
		
		output = m_outputEstimators[state].sample();
		
		sequence.add(new DenseInstance(output.size()));
		Instance frame = sequence.lastInstance();
		
		for(int i = 0; i < output.size(); i++)
			frame.setValue(i, output.get(i));
		
		return state;
	}

	@Override
	public void addValue(double prevState, double state, DoubleVector output,
			double weight) {
		m_stateEstimators[(int)prevState].addValue(state, weight);
		m_outputEstimators[(int)state].addValue(output, weight);
	}

	@Override
	public void addValue0(double state, DoubleVector output, double weight) {
		m_state0Estimator.addValue(state, weight);
		m_outputEstimators[(int)state].addValue(output, weight);
	}

	@Override
	public double getProbability(double prevState, double state, DoubleVector output) throws Exception {
		double ps = m_stateEstimators[(int)prevState].getProbability(state);
		double po = m_outputEstimators[(int)state].getProbability(output);
		double p = ps*po;
		if (Double.isInfinite(p) || Double.isNaN(p))
			throw new Exception("Calculated probability is NaN");
		return p;
	}

	@Override
	public double getProbability0(double state, DoubleVector output) throws Exception {
		return m_state0Estimator.getProbability(state)
		     * m_outputEstimators[(int)state].getProbability(output);
	}


	
	@Override
	public void addValue(double prevState, double state, double output,
			double weight) throws Exception {
		if(getOutputDimension() == 1)
		{
			DoubleVector outputs = new DoubleVector(1, output);
			addValue(prevState, state, outputs, weight);
		}
		else
			throw new Exception("Trying to get the probability of a multivariate output with a single value");

	}

	@Override
	public void addValue0(double state, double output, double weight) throws Exception {
		if(getOutputDimension() == 1)
		{
			DoubleVector outputs = new DoubleVector(1, output);
			addValue0(state, outputs, weight);
		}
		else
			throw new Exception("Trying to get the probability of a multivariate output with a single value");

	}

	@Override
	public double getProbability(double prevState, double state, double output) throws Exception {
		if(getOutputDimension() == 1)
		{
			DoubleVector outputs = new DoubleVector(1, output);
			return getProbability(prevState, state, outputs);
		}
		else
			throw new Exception("Trying to get the probability of a multivariate output with a single value");

	}

	@Override
	public double getProbability0(double state, double output) throws Exception {
		if(getOutputDimension() == 1)
		{
			DoubleVector outputs = new DoubleVector(1, output);
			return getProbability0(state, outputs);
		}
		else
			throw new Exception("Trying to get the probability of a multivariate output with a single value");
	}

	public String  toString() {
	    String s = "MultivariateNormalHMMEstimator\n" +  super.toString();
	    
	    for (int i = 0; i < m_outputEstimators.length; i++)
	    	s = s + "Output Estimator, state " + i + "\n" +  m_outputEstimators[i].toString() + "\n";
	    
	    return s;
	}

	@Override
	public String getRevision() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void calculateParameters() throws Exception {
		if(isTied())
			MultivariateNormalEstimator.calculateTiedParameters(m_outputEstimators);
		else
			for (int i = 0; i < m_outputEstimators.length; i++)
				m_outputEstimators[i].calculateParameters();
	}

}

package weka.estimators;

import java.util.Random;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.DoubleVector;

//import weka.estimators.*;

public class DiscreteHMMEstimator extends AbstractHMMEstimator implements HMMEstimator, java.io.Serializable {

	
	private static final long serialVersionUID = 4585903204046324781L;

	protected Estimator m_outputEstimators[];
	protected int m_NumOutputs;
	


	protected void setupOutputs()
	{
		m_outputEstimators = new Estimator[getNumStates()];
		for (int s = 0; s < getNumStates(); s++)
		{
			m_outputEstimators[s] = new DiscreteEstimator(getNumOutputs(), m_Laplace);
		}
	}
	
	public int getNumOutputs() {
		return m_NumOutputs;
	}
	public void setNumOutputs(int NumOutputs) {
		this.m_NumOutputs = NumOutputs;
		setupOutputs();
	}	

	@Override
	public void setNumStates(int NumStates) {
		super.setNumStates(NumStates);
		setupOutputs();
	}
	
	public DiscreteHMMEstimator() {
		super();
		setNumOutputs(6);
	}
	
	public DiscreteHMMEstimator(int numStates, int numOutputs, boolean laplace) {
		super(numStates, laplace);

		setNumOutputs(numOutputs);
		/*
		m_outputEstimators = new Estimator[numStates];
		for (int s = 0; s < numStates; s++)
		{
			m_outputEstimators[s] = new DiscreteEstimator(numOutputs, laplace);
		}
		*/
	}
	
	

	public DiscreteHMMEstimator(DiscreteHMMEstimator e) throws Exception {
		super(e);

		setNumOutputs(e.getNumOutputs());
		
		//m_outputEstimators = new Estimator[getNumStates()];
		for (int s = 0; s < getNumStates(); s++)
		{
			m_outputEstimators[s] = Estimator.makeCopy(e.m_outputEstimators[s]);
		}
	}
	
	
	
	@Override
	public void addValue(double prevState, double state, DoubleVector output,
			double weight) {
		addValue(prevState, state, output.get(0), weight);
		//m_stateEstimators[(int)prevState].addValue(state, weight);
		//m_outputEstimators[(int)state].addValue(output.get(0), weight);
	}

	@Override
	public void addValue0(double state, DoubleVector output, double weight) {
		addValue0(state, output.get(0), weight);
		//m_state0Estimator.addValue(state, weight);
		//m_outputEstimators[(int)state].addValue(output.get(0), weight);
	}

	@Override
	public double getProbability(double prevState, double state, DoubleVector output) {
		return getProbability(prevState, state, output.get(0));
		//return m_stateEstimators[(int)prevState].getProbability(state)
		//	* m_outputEstimators[(int)state].getProbability(output.get(0));
	}

	@Override
	public double getProbability0(double state, DoubleVector output) {
		return getProbability0(state, output.get(0));
		//return m_state0Estimator.getProbability(state)
		//* m_outputEstimators[(int)state].getProbability(output.get(0));
	}


	
	@Override
	public void addValue(double prevState, double state, double output,
			double weight) {
		m_stateEstimators[(int)prevState].addValue(state, weight);
		m_outputEstimators[(int)state].addValue(output, weight);
	}

	@Override
	public void addValue0(double state, double output, double weight) {
		m_state0Estimator.addValue(state, weight);
		m_outputEstimators[(int)state].addValue(output, weight);
	}

	@Override
	public double getProbability(double prevState, double state, double output) {
		return m_stateEstimators[(int)prevState].getProbability(state)
			* m_outputEstimators[(int)state].getProbability(output);
	}

	@Override
	public double getProbability0(double state, double output) {
		return m_state0Estimator.getProbability(state)
		* m_outputEstimators[(int)state].getProbability(output);
	}



	@Override
	public String getRevision() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public int Sample0(Instances sequence, Random generator) {
		int state;
		int output;
		
		do {
			state = generator.nextInt(getNumStates());
			output = generator.nextInt(getNumOutputs());
		} while (generator.nextDouble() > getProbability0((double)state, (double)output));
		
		sequence.add(new DenseInstance(1));
		Instance frame = sequence.lastInstance();
		
		frame.setValue(0, output);
		
		return state;
	}
	
	@Override
	public int Sample(Instances sequence, int prevState,  Random generator) {
		int state;
		int output;
		
		do {
			state = generator.nextInt(getNumStates());
			output = generator.nextInt(getNumOutputs());
		} while (generator.nextDouble() > getProbability((double)prevState, (double)state, (double)output));
		
		sequence.add(new DenseInstance(1));
		Instance frame = sequence.lastInstance();
		
		frame.setValue(0, output);
		
		return state;
	}
	

	public String  toString() {
	    String s = "DiscreteHMMEstimator\n" + super.toString();
	    
	    for (int i = 0; i < m_outputEstimators.length; i++)
	    	s = s + "Output Estimator, state " + i + " " + m_outputEstimators[i].toString() + "\n";
	    
	    return s;
	}

	@Override
	public void calculateParameters() {
		// do nothing, as discrete estimators use the sufficient
		// statistics directly without calculating parameters
		
	}

}

package weka.classifiers.bayes;


import java.util.Random;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.matrix.DoubleVector;
import weka.core.matrix.Matrix;
import weka.estimators.MultivariateNormalEstimator;
//import junit.framework.TestCase;

public class TestHMMClassifiers {
	
	protected boolean printErrorRates = true;
	
	protected class hmmValue
	{
		public int state;
		public int output;
	}
	
	Random m_rand;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
	}

	@Before
	public void setUp() throws Exception {
		m_rand = new Random(13411);
	}

	@After
	public void tearDown() throws Exception {
	}
	/*
	protected hmmValue sampleHMM0(HMMEstimator model, hmmValue current)
	{
		int state;
		int output;
		
		do {
			state = m_rand.nextInt(model.getNumStates());
			output = m_rand.nextInt(model.getNumOutputs());
		} while (m_rand.nextDouble() > model.getProbability0((double)state, (double)output));
		current.state = state;
		current.output = output;
		
		return current;
	}
	
	protected hmmValue sampleHMM(HMMEstimator model, hmmValue current)
	{
		int state;
		int output;
		
		do {
			state = m_rand.nextInt(model.getNumStates());
			output = m_rand.nextInt(model.getNumOutputs());
		} while (m_rand.nextDouble() > model.getProbability((double)current.state, (double)state, (double)output));
		current.state = state;
		current.output = output;
		
		return current;
	}
	
	
	protected void addSample(Instances sequence, hmmValue sample, ArrayList<String> outputs)
	{
		sequence.add(new DenseInstance(1));
		Instance frame = sequence.lastInstance();
		
		frame.setValue(0, outputs.get(sample.output));
	}
	*/
	/*
	protected Instances sample(List<String> classNames, List<HMMEstimator> models, int numseqs, int length) throws Exception 
	{
		ArrayList<Attribute> attrs = new ArrayList<Attribute>();
		
		ArrayList<String> seqIds = new ArrayList<String>();
		for (int i = 0; i < numseqs; i++)
			seqIds.add("seq_"+i);
		attrs.add(new Attribute("seq-id", seqIds));
		
		attrs.add(new Attribute("class", classNames));
		
		ArrayList<Attribute> seqAttrs = new ArrayList<Attribute>();
		ArrayList<String> outputs = new ArrayList<String>();
		for(int i = 0; i < models.get(0).getNumOutputs(); i++)
			outputs.add("output_"+i);
		seqAttrs.add(new Attribute("output", outputs));
		Instances seqHeader = new Instances("seq", seqAttrs, 0);
		attrs.add(new Attribute("sequence", seqHeader));
		
		Instances seqs = new Instances("test", attrs, numseqs);
		seqs.setClassIndex(1);
		
		for (int seq=0; seq<numseqs; seq++)
		{
			seqs.add(new DenseInstance(3));
			Instance inst = seqs.lastInstance();
			inst.setValue(0, seqIds.get(seq));
			int classId = m_rand.nextInt(classNames.size());
			inst.setValue(1, classNames.get(classId));
			//System.out.print("class "+classId+":");
			
			HMMEstimator model = models.get(classId);
			
			Instances sequence = new Instances(seqIds.get(seq), seqAttrs, length);
			hmmValue current = new hmmValue();
			current = sampleHMM0(model, current);
			addSample(sequence, current, outputs);
			for (int i = 1; i < length; i++)
			{
				current = sampleHMM(model, current);
				addSample(sequence, current, outputs);
				//System.out.print(current.output);
			}
			//System.out.print("\n");
			Attribute seqA = seqs.attribute(2);
			inst.setValue(seqA, seqA.addRelation(sequence));
		}
			
		return seqs;
	}
	*/
	
	protected Instances getSequence1(int numseqs, int length) throws Exception
	{
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		hmm.setNumOutputs(2);
		
		double state0Probs[][] = new double[2][2];
		state0Probs[0][0] = 0.5;
		state0Probs[0][1] = 0.5;
		state0Probs[1][0] = 0.5;
		state0Probs[1][1] = 0.5;
		
		double stateProbs[][][] = new double[2][2][2];
		stateProbs[0][0][0] = 0.9;
		stateProbs[0][0][1] = 0.1;
		stateProbs[0][1][0] = 0.9;
		stateProbs[0][1][1] = 0.1;
		stateProbs[1][0][0] = 0.9;
		stateProbs[1][0][1] = 0.1;
		stateProbs[1][1][0] = 0.5;
		stateProbs[1][1][1] = 0.5;
		
		double outputProbs[][][] = new double[2][2][2];
		outputProbs[0][0][0] = 1.0;
		outputProbs[0][0][1] = 0.0;
		outputProbs[0][1][0] = 1.0;
		outputProbs[0][1][1] = 0.0;
		outputProbs[1][0][0] = 0.0;
		outputProbs[1][0][1] = 1.0;
		outputProbs[1][1][0] = 0.0;
		outputProbs[1][1][1] = 1.0;
		
		hmm.initEstimatorsUnivariateDiscrete(2, state0Probs, stateProbs, outputProbs);	

		return hmm.sample(numseqs, length);
		

	}
	

	
	protected Instances getSequence2(int numseqs, int length) throws Exception
	{
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		hmm.setNumOutputs(2);
		
		double state0Probs[][] = new double[2][2];
		state0Probs[0][0] = 0.5;
		state0Probs[0][1] = 0.5;
		state0Probs[1][0] = 0.5;
		state0Probs[1][1] = 0.5;
		
		double stateProbs[][][] = new double[2][2][2];
		stateProbs[0][0][0] = 0.9;
		stateProbs[0][0][1] = 0.1;
		stateProbs[0][1][0] = 0.1;
		stateProbs[0][1][1] = 0.9;
		stateProbs[1][0][0] = 0.9;
		stateProbs[1][0][1] = 0.1;
		stateProbs[1][1][0] = 0.5;
		stateProbs[1][1][1] = 0.5;
		
		double outputProbs[][][] = new double[2][2][2];
		outputProbs[0][0][0] = 1.0;
		outputProbs[0][0][1] = 0.0;
		outputProbs[0][1][0] = 0.0;
		outputProbs[0][1][1] = 1.0;
		outputProbs[1][0][0] = 1.0;
		outputProbs[1][0][1] = 0.0;
		outputProbs[1][1][0] = 0.0;
		outputProbs[1][1][1] = 1.0;
		
		hmm.initEstimatorsUnivariateDiscrete(2, state0Probs, stateProbs, outputProbs);	

		return hmm.sample(numseqs, length);

	}
	
	protected Instances getSequence3(int numseqs, int length) throws Exception
	{

		HMM hmm = new HMM();
		
		hmm.setNumStates(6);
		hmm.setNumOutputs(6);
		
		double state0Probs[][] = new double[2][6];
		for (int c=0; c < 2; c++)
			for (int s=0; s < 6; s++)
				state0Probs[c][s]=0.0;
		state0Probs[0][0] = 1.0;
		state0Probs[1][0] = 1.0;
		
		double stateProbs[][][] = new double[2][6][6];
		for (int c=0; c < 2; c++)
			for (int ps=0; ps < 6; ps++)
				for (int s=0; s < 6; s++)
					stateProbs[c][ps][s]=0.0;
		
		for (int s=0; s < 5; s++)
		{
			stateProbs[0][s][s]=0.9;
			stateProbs[0][s][s+1]=0.1;
		}
		stateProbs[0][5][5] = 1.0;
		
		
		stateProbs[1][0][0] = 0.7;
		stateProbs[1][0][1] = 0.3;
		stateProbs[1][1][1] = 0.5;
		stateProbs[1][1][2] = 0.5;
		stateProbs[1][2][2] = 0.95;
		stateProbs[1][2][3] = 0.05;
		stateProbs[1][3][3] = 0.7;
		stateProbs[1][3][4] = 0.3;
		stateProbs[1][4][4] = 0.95;
		stateProbs[1][4][5] = 0.05; 
		stateProbs[1][5][5] = 1.0;
		
		double outputProbs[][][] = new double[2][6][6];
		for (int c=0; c < 2; c++)
			for (int s=0; s < 6; s++)
				for (int o=0; o < 6; o++)
					outputProbs[c][s][0]=0.0;
		outputProbs[0][0][0] = 1.0;
		outputProbs[0][1][1] = 1.0;
		outputProbs[0][2][2] = 1.0;
		outputProbs[0][3][3] = 1.0;
		outputProbs[0][4][4] = 1.0;
		outputProbs[0][5][5] = 1.0;
		outputProbs[1][0][0] = 1.0;
		outputProbs[1][1][1] = 1.0;
		outputProbs[1][2][2] = 1.0;
		outputProbs[1][3][3] = 1.0;
		outputProbs[1][4][4] = 1.0;
		outputProbs[1][5][5] = 1.0;
		
		hmm.initEstimatorsUnivariateDiscrete(2, state0Probs, stateProbs, outputProbs);	

		return hmm.sample(numseqs, length);
		
	}
	
	protected Instances getMVSequence1(int numseqs, int length) throws Exception
	{
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		hmm.setNumeric(true);
		hmm.setOutputDimension(4);
		
		double state0Probs[][] = new double[2][2];
		state0Probs[0][0] = 0.5;
		state0Probs[0][1] = 0.5;
		state0Probs[1][0] = 0.5;
		state0Probs[1][1] = 0.5;
		
		double stateProbs[][][] = new double[2][2][2];
		stateProbs[0][0][0] = 0.9;
		stateProbs[0][0][1] = 0.1;
		stateProbs[0][1][0] = 0.9;
		stateProbs[0][1][1] = 0.1;
		stateProbs[1][0][0] = 0.9;
		stateProbs[1][0][1] = 0.1;
		stateProbs[1][1][0] = 0.5;
		stateProbs[1][1][1] = 0.5;
		
		DoubleVector outputMeans[][] = new DoubleVector[2][2];
		Matrix outputVars[][] = new Matrix[2][2];
		
		for (int c=0; c < 2; c++)
		{
			for (int s=0; s < 2; s++)
			{
				outputMeans[c][s] = new DoubleVector(4, 10.0*(double)c);
				outputVars[c][s] = Matrix.identity(4, 4);
			}
		}

		hmm.initEstimatorsMultivariateNormal(2, state0Probs, stateProbs, outputMeans, outputVars, null);	

		return hmm.sample(numseqs, length);
	}
	

	
	protected Instances getMVSequence2(int numseqs, int length) throws Exception
	{
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		hmm.setNumeric(true);
		hmm.setOutputDimension(4);
		
		double state0Probs[][] = new double[2][2];
		state0Probs[0][0] = 0.5;
		state0Probs[0][1] = 0.5;
		state0Probs[1][0] = 0.5;
		state0Probs[1][1] = 0.5;
		
		double stateProbs[][][] = new double[2][2][2];
		stateProbs[0][0][0] = 0.8;
		stateProbs[0][0][1] = 0.2;
		stateProbs[0][1][0] = 0.2;
		stateProbs[0][1][1] = 0.8;
		stateProbs[1][0][0] = 0.9;
		stateProbs[1][0][1] = 0.1;
		stateProbs[1][1][0] = 0.1;
		stateProbs[1][1][1] = 0.9;
		
		
		DoubleVector outputMeans[][] = new DoubleVector[2][2];
		Matrix outputVars[][] = new Matrix[2][2];
		
		for (int c=0; c < 2; c++)
		{
			for (int s=0; s < 2; s++)
			{
				outputMeans[c][s] = new DoubleVector(4, 10.0*(double)s);
				outputVars[c][s] = Matrix.identity(4,4);
			}
		}

		hmm.initEstimatorsMultivariateNormal(2, state0Probs, stateProbs, outputMeans, outputVars, null);	

		return hmm.sample(numseqs, length);

	}
	
//	protected Instances getMVSequence3(int numseqs, int length) throws Exception
//	{
//
//		HMM hmm = new HMM();
//		
//		hmm.setNumStates(6);
//		hmm.setNumOutputs(6);
//		
//		double state0Probs[][] = new double[2][6];
//		for (int c=0; c < 2; c++)
//			for (int s=0; s < 6; s++)
//				state0Probs[c][s]=0.0;
//		state0Probs[0][0] = 1.0;
//		state0Probs[1][0] = 1.0;
//		
//		double stateProbs[][][] = new double[2][6][6];
//		for (int c=0; c < 2; c++)
//			for (int ps=0; ps < 6; ps++)
//				for (int s=0; s < 6; s++)
//					stateProbs[c][ps][s]=0.0;
//		
//		for (int s=0; s < 5; s++)
//		{
//			stateProbs[0][s][s]=0.9;
//			stateProbs[0][s][s+1]=0.1;
//		}
//		stateProbs[0][5][5] = 1.0;
//		
//		
//		stateProbs[1][0][0] = 0.7;
//		stateProbs[1][0][1] = 0.3;
//		stateProbs[1][1][1] = 0.5;
//		stateProbs[1][1][2] = 0.5;
//		stateProbs[1][2][2] = 0.95;
//		stateProbs[1][2][3] = 0.05;
//		stateProbs[1][3][3] = 0.7;
//		stateProbs[1][3][4] = 0.3;
//		stateProbs[1][4][4] = 0.95;
//		stateProbs[1][4][5] = 0.05; 
//		stateProbs[1][5][5] = 1.0;
//		
//		double outputProbs[][][] = new double[2][6][6];
//		for (int c=0; c < 2; c++)
//			for (int s=0; s < 6; s++)
//				for (int o=0; o < 6; o++)
//					outputProbs[c][s][0]=0.0;
//		outputProbs[0][0][0] = 1.0;
//		outputProbs[0][1][1] = 1.0;
//		outputProbs[0][2][2] = 1.0;
//		outputProbs[0][3][3] = 1.0;
//		outputProbs[0][4][4] = 1.0;
//		outputProbs[0][5][5] = 1.0;
//		outputProbs[1][0][0] = 1.0;
//		outputProbs[1][1][1] = 1.0;
//		outputProbs[1][2][2] = 1.0;
//		outputProbs[1][3][3] = 1.0;
//		outputProbs[1][4][4] = 1.0;
//		outputProbs[1][5][5] = 1.0;
//
//		hmm.initEstimatorsMultivariateNormal(2, state0Probs, stateProbs, outputMeans, outputVars);	
//
//		return hmm.sample(numseqs, length);
//		
//	}
	
	protected Instances getMVSequence3(int numseqs, int length) throws Exception
	{

		HMM hmm = new HMM();
		
		hmm.setNumStates(4);
		hmm.setNumeric(true);
		hmm.setOutputDimension(4);
		
		double state0Probs[][] = new double[2][4];
		for (int c=0; c < 2; c++)
			for (int s=0; s < 4; s++)
				state0Probs[c][s]=0.0;
		state0Probs[0][0] = 1.0;
		state0Probs[1][0] = 1.0;
		
		double stateProbs[][][] = new double[2][4][4];
		for (int c=0; c < 2; c++)
			for (int ps=0; ps < 4; ps++)
				for (int s=0; s < 4; s++)
					stateProbs[c][ps][s]=0.0;
		
		for (int s=0; s < 3; s++)
		{
			stateProbs[0][s][s]=0.9;
			stateProbs[0][s][s+1]=0.1;
		}
		stateProbs[0][3][3] = 1.0;
		
		for (int ps=0; ps < 4; ps++)
		{
			for (int s=0; s < 4; s++)
			{
				if(ps == s)
					stateProbs[1][ps][s]=0.7;
				else
					stateProbs[1][ps][s]=0.1;
			}
		}
		
		DoubleVector outputMeans[][] = new DoubleVector[2][4];
		Matrix outputVars[][] = new Matrix[2][4];
		
		for (int c=0; c < 2; c++)
		{
			for (int s=0; s < 4; s++)
			{
				outputMeans[c][s] = new DoubleVector(4, 10.0*(double)s);
				outputVars[c][s] = Matrix.identity(4, 4);
			}
		}
		
		
		hmm.initEstimatorsMultivariateNormal(2, state0Probs, stateProbs, outputMeans, outputVars, null);	

		return hmm.sample(numseqs, length);
		
	}
	
	/*
	protected Instances getSequenceUniform(int numseqs, int length) throws Exception
	{
		DiscreteHMMEstimator model;
		ArrayList<HMMEstimator> models = new ArrayList<HMMEstimator>();
		model = new DiscreteHMMEstimator(2, 2, false);
		model.addValue0(0, 0, 1.0);
		model.addValue0(0, 1, 1.0);
		model.addValue0(1, 0, 1.0);
		model.addValue0(1, 1, 1.0);
		model.addValue(0, 0, 0, 0.5);
		model.addValue(1, 0, 0, 0.5);
		model.addValue(1, 1, 1, 0.5);
		model.addValue(0, 1, 1, 0.5);
		model.addValue(0, 0, 1, 0.0);
		model.addValue(1, 0, 1, 0.0);
		model.addValue(1, 1, 0, 0.0);
		model.addValue(0, 1, 0, 0.0);
		models.add(model);
		model = new DiscreteHMMEstimator(2, 2, false);
		model.addValue0(0, 0, 1.0);
		model.addValue0(0, 1, 1.0);
		model.addValue0(1, 0, 1.0);
		model.addValue0(1, 1, 1.0);
		model.addValue(0, 0, 0, 0.5);
		model.addValue(1, 0, 0, 0.5);
		model.addValue(1, 1, 1, 0.5);
		model.addValue(0, 1, 1, 0.5);
		model.addValue(0, 0, 1, 0.0);
		model.addValue(1, 0, 1, 0.0);
		model.addValue(1, 1, 0, 0.0);
		model.addValue(0, 1, 0, 0.0);
		models.add(model);
		
		ArrayList<String> classNames = new ArrayList<String>();
		classNames.add("one");
		classNames.add("two");
		
		return sample(classNames, models, numseqs, length);
	}
	*/
	
	@Test
	public void TestSequence1() throws Exception
	{
		Instances train = getSequence1(100, 100);
		Instances test1 = getSequence1(20, 100);
		//Instances test2 = getSequenceUniform(20, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		
		hmm.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test Seq 1 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
		
		eval.evaluateModel(hmm, test1);
		errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test Seq 1 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
		
		//eval.evaluateModel(hmm, test2);
		//errorRate = eval.errorRate();
		//assertTrue(errorRate < 0.01);
	}
	

	
	@Test
	public void TestSequence2_1() throws Exception
	{
		Instances train = getSequence2(200, 30);
		Instances test1 = getSequence2(50, 30);
		//Instances test2 = getSequenceUniform(20, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		hmm.setNumOutputs(2);
		
		double outputProbs[][][] = new double[2][2][2];
		outputProbs[0][0][0] = 1.0;
		outputProbs[0][0][1] = 0.0;
		outputProbs[0][1][0] = 0.0;
		outputProbs[0][1][1] = 1.0;
		outputProbs[1][0][0] = 1.0;
		outputProbs[1][0][1] = 0.0;
		outputProbs[1][1][0] = 0.0;
		outputProbs[1][1][1] = 1.0;
		
		hmm.initEstimatorsUnivariateDiscrete(2, null, null, outputProbs);
		/*
		hmm.setProbability0(0, 0, 1, 0.0);
		hmm.setProbability0(0, 1, 0, 0.0);
		hmm.setProbability0(1, 0, 1, 0.0);
		hmm.setProbability0(1, 1, 0, 0.0);
		hmm.setProbability(0, 0, 0, 1, 0.0);
		hmm.setProbability(0, 1, 0, 1, 0.0);
		hmm.setProbability(0, 0, 1, 0, 0.0);
		hmm.setProbability(0, 1, 1, 0, 0.0);
		hmm.setProbability(1, 0, 0, 1, 0.0);
		hmm.setProbability(1, 1, 0, 1, 0.0);
		hmm.setProbability(1, 0, 1, 0, 0.0);
		hmm.setProbability(1, 1, 1, 0, 0.0);
		*/
		
		
		hmm.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test Seq 2_1 error rate " + errorRate);
		assertTrue(errorRate < 0.15);
		
		eval.evaluateModel(hmm, test1);
		errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test Seq 2_1 error rate " + errorRate);
		assertTrue(errorRate < 0.15);
		
		//eval.evaluateModel(hmm, test2);
		//errorRate = eval.errorRate();
		//assertTrue(errorRate < 0.01);
	}
	

	@Test
	public void TestSequence2_2() throws Exception
	{
		Instances train = getSequence2(100, 100);
		Instances test1 = getSequence2(50, 100);
		//Instances test2 = getSequenceUniform(20, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		hmm.setNumOutputs(2);
	
		
		//hmm.initEstimators(2, null);
		/*
		hmm.setProbability0(0, 0, 1, 0.0);
		hmm.setProbability0(0, 1, 0, 0.0);
		hmm.setProbability0(1, 0, 1, 0.0);
		hmm.setProbability0(1, 1, 0, 0.0);
		hmm.setProbability(0, 0, 0, 1, 0.0);
		hmm.setProbability(0, 1, 0, 1, 0.0);
		hmm.setProbability(0, 0, 1, 0, 0.0);
		hmm.setProbability(0, 1, 1, 0, 0.0);
		hmm.setProbability(1, 0, 0, 1, 0.0);
		hmm.setProbability(1, 1, 0, 1, 0.0);
		hmm.setProbability(1, 0, 1, 0, 0.0);
		hmm.setProbability(1, 1, 1, 0, 0.0);
		*/
		
		
		hmm.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test Seq 2_2 error rate " + errorRate);
		assertTrue(errorRate < 0.25);
		
		eval.evaluateModel(hmm, test1);
		errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test Seq 2_2 error rate " + errorRate);
		assertTrue(errorRate < 0.25);
		
		//eval.evaluateModel(hmm, test2);
		//errorRate = eval.errorRate();
		//assertTrue(errorRate < 0.01);
	}


	
	@Test
	public void TestSequence3_1() throws Exception
	{
		Instances train = getSequence3(100, 100);
		Instances test1 = getSequence3(50, 100);
		//Instances test2 = getSequenceUniform(20, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(6);
		hmm.setNumOutputs(6);
		
		double state0Probs[][] = new double[2][6];
		for (int c=0; c < 2; c++)
			for (int s=0; s < 6; s++)
				state0Probs[c][s]=0.0;
		state0Probs[0][0] = 1.0f;
		state0Probs[1][0] = 1.0f;
		
		double stateProbs[][][] = new double[2][6][6];
		for (int c=0; c < 2; c++)
			for (int ps=0; ps < 6; ps++)
				for (int s=0; s < 6; s++)
					stateProbs[c][ps][s]=0.0;
		
		for (int c=0; c < 2; c++)
		{
			for (int s=0; s < 5; s++)
			{
				stateProbs[c][s][s]=0.5;
				stateProbs[c][s][s+1]=0.5;
			}
			stateProbs[c][5][5] = 1.0;
		}
		
		hmm.initEstimatorsUnivariateDiscrete(2, state0Probs, stateProbs, null);	
		
		hmm.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test Seq 3_1 error rate " + errorRate);
		// quite a high error rate as the models are similar
		assertTrue(errorRate < 0.4);
		
		eval.evaluateModel(hmm, test1);
		errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test Seq 3_1 error rate " + errorRate);
		// quite a high error rate as the models are similar
		//System.out.println(errorRate);
		assertTrue(errorRate < 0.4);

	}


	@Test
	public void TestSequence3_2() throws Exception
	{
		Instances train = getSequence3(100, 100);
		Instances test1 = getSequence3(50, 100);
		//Instances test2 = getSequenceUniform(20, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(6);
		hmm.setNumOutputs(6);
		
		
		//hmm.initEstimators(2, null);

		hmm.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test Seq 3_2 error rate " + errorRate);
		// quite a high error rate as the models are similar
		//System.out.println(errorRate);
		assertTrue(errorRate < 0.25);
		
		eval.evaluateModel(hmm, test1);
		errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test Seq 3_2 error rate " + errorRate);
		// quite a high error rate as the models are similar
		//System.out.println(errorRate);
		assertTrue(errorRate < 0.25);
	}
	
	@Test
	public void TestMVSequence1() throws Exception
	{
		Instances train = getMVSequence1(100, 100);
		Instances test1 = getMVSequence1(20, 100);
		//Instances test2 = getSequenceUniform(20, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		hmm.setNumeric(true);
		hmm.setOutputDimension(4);
		hmm.setCovarianceType(new SelectedTag(MultivariateNormalEstimator.COVARIANCE_DIAGONAL, HMM.TAGS_COVARIANCE_TYPE));
		
		hmm.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 1 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
		
		eval.evaluateModel(hmm, test1);
		errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 1 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
		
		//eval.evaluateModel(hmm, test2);
		//errorRate = eval.errorRate();
		//assertTrue(errorRate < 0.01);
	}
	
	@Test
	public void TestSequence3_4() throws Exception
	{
		Instances train = getSequence3(100, 100);
		Instances test1 = getSequence3(50, 100);
		//Instances test2 = getSequenceUniform(20, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(6);
		hmm.setNumOutputs(6);
		hmm.setLeftRight(true);
		
		hmm.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test Seq 3_1 error rate " + errorRate);
		// quite a high error rate as the models are similar
		assertTrue(errorRate < 0.4);
		
		eval.evaluateModel(hmm, test1);
		errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test Seq 3_1 error rate " + errorRate);
		// quite a high error rate as the models are similar
		//System.out.println(errorRate);
		assertTrue(errorRate < 0.4);

	}

	
	@Test
	public void TestMVSequence2_1() throws Exception
	{
		Instances train = getMVSequence2(200, 100);
		Instances test1 = getMVSequence2(100, 100);
		//Instances test2 = getSequenceUniform(20, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		hmm.setNumeric(true);
		hmm.setOutputDimension(4);
		
		double state0Probs[][] = new double[2][2];
		state0Probs[0][0] = 0.5;
		state0Probs[0][1] = 0.5;
		state0Probs[1][0] = 0.5;
		state0Probs[1][1] = 0.5;
		
		double stateProbs[][][] = new double[2][2][2];
		stateProbs[0][0][0] = 0.85;
		stateProbs[0][0][1] = 0.15;
		stateProbs[0][1][0] = 0.85;
		stateProbs[0][1][1] = 0.15;
		stateProbs[1][0][0] = 0.95;
		stateProbs[1][0][1] = 0.05;
		stateProbs[1][1][0] = 0.05;
		stateProbs[1][1][1] = 0.95;
	
		DoubleVector outputMeans[][] = new DoubleVector[2][2];
		Matrix outputVars[][] = new Matrix[2][2];
		
		for (int c=0; c < 2; c++)
		{
			for (int s=0; s < 2; s++)
			{
				outputMeans[c][s] = new DoubleVector(4, 10.0*(double)s);
				outputVars[c][s] = Matrix.identity(4, 4);
				//outputVars[c][s].timesEquals(4.0f);
			}
		}
		
		hmm.initEstimatorsMultivariateNormal(2, state0Probs, stateProbs, outputMeans, outputVars, null);	

		
		//hmm.initEstimatorsUnivariateDiscrete(2, null, null, outputProbs);
		/*
		hmm.setProbability0(0, 0, 1, 0.0);
		hmm.setProbability0(0, 1, 0, 0.0);
		hmm.setProbability0(1, 0, 1, 0.0);
		hmm.setProbability0(1, 1, 0, 0.0);
		hmm.setProbability(0, 0, 0, 1, 0.0);
		hmm.setProbability(0, 1, 0, 1, 0.0);
		hmm.setProbability(0, 0, 1, 0, 0.0);
		hmm.setProbability(0, 1, 1, 0, 0.0);
		hmm.setProbability(1, 0, 0, 1, 0.0);
		hmm.setProbability(1, 1, 0, 1, 0.0);
		hmm.setProbability(1, 0, 1, 0, 0.0);
		hmm.setProbability(1, 1, 1, 0, 0.0);
		*/
		
		
		hmm.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_1 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
		
		eval.evaluateModel(hmm, test1);
		errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_1 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
		
		//eval.evaluateModel(hmm, test2);
		//errorRate = eval.errorRate();
		//assertTrue(errorRate < 0.01);
	}
	
	@Test
	public void TestMVSequence2_2() throws Exception
	{
		Instances train = getMVSequence2(200, 100);
		Instances test1 = getMVSequence2(100, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		hmm.setNumeric(true);
		hmm.setOutputDimension(4);
		
		DoubleVector outputMeans[][] = new DoubleVector[2][2];
		Matrix outputVars[][] = new Matrix[2][2];
		
		for (int c=0; c < 2; c++)
		{
			for (int s=0; s < 2; s++)
			{
				outputMeans[c][s] = new DoubleVector(4, 10.0*(double)s);
				outputVars[c][s] = Matrix.identity(4, 4);
				//outputVars[c][s].timesEquals(4.0f);
			}
		}
		
		hmm.initEstimatorsMultivariateNormal(2, null, null, outputMeans, outputVars, null);	
		
		hmm.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_2 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
		
		eval.evaluateModel(hmm, test1);
		errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_2 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
	}

	@Test
	public void TestMVSequence2_3() throws Exception
	{
		Instances train = getMVSequence2(200, 100);
		Instances test1 = getMVSequence2(100, 100);
		//Instances test2 = getSequenceUniform(20, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		hmm.setNumeric(true);
		hmm.setOutputDimension(4);
		
		double state0Probs[][] = new double[2][2];
		state0Probs[0][0] = 0.5;
		state0Probs[0][1] = 0.5;
		state0Probs[1][0] = 0.5;
		state0Probs[1][1] = 0.5;
		
		double stateProbs[][][] = new double[2][2][2];
		stateProbs[0][0][0] = 0.85;
		stateProbs[0][0][1] = 0.15;
		stateProbs[0][1][0] = 0.85;
		stateProbs[0][1][1] = 0.15;
		stateProbs[1][0][0] = 0.95;
		stateProbs[1][0][1] = 0.05;
		stateProbs[1][1][0] = 0.05;
		stateProbs[1][1][1] = 0.95;
		
		hmm.initEstimatorsMultivariateNormal(2, state0Probs, stateProbs, null, null, null);	

		
		//hmm.initEstimatorsUnivariateDiscrete(2, null, null, outputProbs);
		/*
		hmm.setProbability0(0, 0, 1, 0.0);
		hmm.setProbability0(0, 1, 0, 0.0);
		hmm.setProbability0(1, 0, 1, 0.0);
		hmm.setProbability0(1, 1, 0, 0.0);
		hmm.setProbability(0, 0, 0, 1, 0.0);
		hmm.setProbability(0, 1, 0, 1, 0.0);
		hmm.setProbability(0, 0, 1, 0, 0.0);
		hmm.setProbability(0, 1, 1, 0, 0.0);
		hmm.setProbability(1, 0, 0, 1, 0.0);
		hmm.setProbability(1, 1, 0, 1, 0.0);
		hmm.setProbability(1, 0, 1, 0, 0.0);
		hmm.setProbability(1, 1, 1, 0, 0.0);
		*/
		
		
		hmm.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_3 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
		
		eval.evaluateModel(hmm, test1);
		errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_3 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
		
		//eval.evaluateModel(hmm, test2);
		//errorRate = eval.errorRate();
		//assertTrue(errorRate < 0.01);
	}

	
	@Test
	public void TestMVSequence2_4() throws Exception
	{
		Instances train = getMVSequence2(100, 100);
		Instances test1 = getMVSequence2(50, 100);
		//Instances test2 = getSequenceUniform(20, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		hmm.setNumeric(true);
		hmm.setOutputDimension(4);
		
		//hmm.initEstimators(2);
		/*
		hmm.setProbability0(0, 0, 1, 0.0);
		hmm.setProbability0(0, 1, 0, 0.0);
		hmm.setProbability0(1, 0, 1, 0.0);
		hmm.setProbability0(1, 1, 0, 0.0);
		hmm.setProbability(0, 0, 0, 1, 0.0);
		hmm.setProbability(0, 1, 0, 1, 0.0);
		hmm.setProbability(0, 0, 1, 0, 0.0);
		hmm.setProbability(0, 1, 1, 0, 0.0);
		hmm.setProbability(1, 0, 0, 1, 0.0);
		hmm.setProbability(1, 1, 0, 1, 0.0);
		hmm.setProbability(1, 0, 1, 0, 0.0);
		hmm.setProbability(1, 1, 1, 0, 0.0);
		*/
		
		
		hmm.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_4 error rate " + errorRate);
		assertTrue(errorRate < 0.2);
		
		eval.evaluateModel(hmm, test1);
		errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_4 error rate " + errorRate);
		assertTrue(errorRate < 0.25);
		
		//eval.evaluateModel(hmm, test2);
		//errorRate = eval.errorRate();
		//assertTrue(errorRate < 0.01);
	}


	
	@Test
	public void TestMVSequence2_5() throws Exception
	{
		Instances train = getMVSequence2(200, 100);
		Instances test1 = getMVSequence2(100, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		hmm.setNumeric(true);
		hmm.setOutputDimension(4);
		hmm.setCovarianceType(new SelectedTag(MultivariateNormalEstimator.COVARIANCE_DIAGONAL, HMM.TAGS_COVARIANCE_TYPE));
		
		DoubleVector outputMeans[][] = new DoubleVector[2][2];
		Matrix outputVars[][] = new Matrix[2][2];
		
		for (int c=0; c < 2; c++)
		{
			for (int s=0; s < 2; s++)
			{
				outputMeans[c][s] = new DoubleVector(4, 10.0*(double)s);
				outputVars[c][s] = Matrix.identity(4, 4);
				//outputVars[c][s].timesEquals(4.0f);
			}
		}
		
		hmm.initEstimatorsMultivariateNormal(2, null, null, outputMeans, outputVars, null);	
		
		hmm.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_5 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
		
		eval.evaluateModel(hmm, test1);
		errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_5 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
	}


	@Test
	public void TestMVSequence2_6() throws Exception
	{
		Instances train = getMVSequence2(200, 100);
		Instances test1 = getMVSequence2(100, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		hmm.setNumeric(true);
		hmm.setOutputDimension(4);
		hmm.setCovarianceType(new SelectedTag(MultivariateNormalEstimator.COVARIANCE_SPHERICAL, HMM.TAGS_COVARIANCE_TYPE));
		
		DoubleVector outputMeans[][] = new DoubleVector[2][2];
		Matrix outputVars[][] = new Matrix[2][2];
		
		for (int c=0; c < 2; c++)
		{
			for (int s=0; s < 2; s++)
			{
				outputMeans[c][s] = new DoubleVector(4, 10.0*(double)s);
				outputVars[c][s] = Matrix.identity(4, 4);
				//outputVars[c][s].timesEquals(4.0f);
			}
		}
		
		hmm.initEstimatorsMultivariateNormal(2, null, null, outputMeans, outputVars, null);	
		
		hmm.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_6 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
		
		eval.evaluateModel(hmm, test1);
		errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_6 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
	}

	@Test
	public void TestMVSequence2_7() throws Exception
	{
		Instances train = getMVSequence2(200, 100);
		Instances test1 = getMVSequence2(100, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		hmm.setNumeric(true);
		hmm.setOutputDimension(4);
		hmm.setTied(true);
		
		DoubleVector outputMeans[][] = new DoubleVector[2][2];
		Matrix outputVars[][] = new Matrix[2][2];
		
		for (int c=0; c < 2; c++)
		{
			for (int s=0; s < 2; s++)
			{
				outputMeans[c][s] = new DoubleVector(4, 10.0*(double)s);
				outputVars[c][s] = Matrix.identity(4, 4);
				//outputVars[c][s].timesEquals(4.0f);
			}
		}
		
		hmm.initEstimatorsMultivariateNormal(2, null, null, outputMeans, outputVars, null);	
		
		hmm.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_7 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
		
		eval.evaluateModel(hmm, test1);
		errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_7 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
	}



	@Test
	public void TestMVSequence2_8() throws Exception
	{
		Instances train = getMVSequence2(200, 100);
		Instances test1 = getMVSequence2(100, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(2);
		hmm.setNumeric(true);
		hmm.setOutputDimension(4);
		hmm.setTied(true);
		hmm.setCovarianceType(new SelectedTag(MultivariateNormalEstimator.COVARIANCE_SPHERICAL, HMM.TAGS_COVARIANCE_TYPE));
		
		DoubleVector outputMeans[][] = new DoubleVector[2][2];
		Matrix outputVars[][] = new Matrix[2][2];
		
		for (int c=0; c < 2; c++)
		{
			for (int s=0; s < 2; s++)
			{
				outputMeans[c][s] = new DoubleVector(4, 10.0*(double)s);
				outputVars[c][s] = Matrix.identity(4, 4);
				//outputVars[c][s].timesEquals(4.0f);
			}
		}
		
		hmm.initEstimatorsMultivariateNormal(2, null, null, outputMeans, outputVars, null);	
		
		hmm.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_8 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
		
		eval.evaluateModel(hmm, test1);
		errorRate = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 2_8 error rate " + errorRate);
		assertTrue(errorRate < 0.1);
	}

	
	@Test
	public void TestMVSequence3_1() throws Exception
	{
		//System.out.println("about to sample training set");
		Instances train = getMVSequence3(100, 100);
		//System.out.println("about to sample testing set");
		Instances test1 = getMVSequence3(50, 100);
		//Instances test2 = getSequenceUniform(20, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(4);
		hmm.setNumeric(true);
		hmm.setOutputDimension(4);
		
		
		DoubleVector outputMeans[][] = new DoubleVector[2][4];
		Matrix outputVars[][] = new Matrix[2][4];
		
		for (int c=0; c < 2; c++)
		{
			for (int s=0; s < 4; s++)
			{
				outputMeans[c][s] = new DoubleVector(4, 10.0*(double)s);
				outputVars[c][s] = Matrix.identity(4, 4);
				outputVars[c][s].timesEquals(4.0f);
			}
		}
		
		
		hmm.initEstimatorsMultivariateNormal(2, null, null, outputMeans, outputVars, null);	



		//System.out.println("about to build classifier");
		hmm.buildClassifier(train);

		//System.out.println("about to evaluate");
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate1 = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 3_1 error rate " + errorRate1);
		// quite a high error rate as the models are similar
		//assertTrue(errorRate < 0.25);
		
		eval.evaluateModel(hmm, test1);
		double errorRate2 = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 3_1 error rate " + errorRate2);
		// quite a high error rate as the models are similar
		assertTrue(errorRate1 < 0.25);
		assertTrue(errorRate2 < 0.25);
	}


	@Test
	public void TestMVSequence3_2() throws Exception
	{
		//System.out.println("about to sample training set");
		Instances train = getMVSequence3(100, 100);
		//System.out.println("about to sample testing set");
		Instances test1 = getMVSequence3(50, 100);
		//Instances test2 = getSequenceUniform(20, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(4);
		hmm.setNumeric(true);
		hmm.setOutputDimension(4);
		
		double state0Probs[][] = new double[2][4];
		for (int c=0; c < 2; c++)
			for (int s=0; s < 4; s++)
				state0Probs[c][s]=0.0;
		state0Probs[0][0] = 1.0;
		state0Probs[1][0] = 1.0;
		
		double stateProbs[][][] = new double[2][4][4];
		for (int c=0; c < 2; c++)
			for (int ps=0; ps < 4; ps++)
				for (int s=0; s < 4; s++)
					stateProbs[c][ps][s]=0.0;
		
		for (int s=0; s < 3; s++)
		{
			stateProbs[0][s][s]=0.9;
			stateProbs[0][s][s+1]=0.1;
		}
		stateProbs[0][3][3] = 1.0;
		
		for (int ps=0; ps < 4; ps++)
		{
			for (int s=0; s < 4; s++)
			{
				if(ps == s)
					stateProbs[1][ps][s]=0.7;
				else
					stateProbs[1][ps][s]=0.1;
			}
		}
		
		
		
		
		hmm.initEstimatorsMultivariateNormal(2, state0Probs, stateProbs, null, null, null);	



		//System.out.println("about to build classifier");
		hmm.buildClassifier(train);

		//System.out.println("about to evaluate");
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate1 = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 3_2 error rate " + errorRate1);
		// quite a high error rate as the models are similar
		//assertTrue(errorRate < 0.25);
		
		eval.evaluateModel(hmm, test1);
		double errorRate2 = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 3_2 error rate " + errorRate2);
		// quite a high error rate as the models are similar
		assertTrue(errorRate1 < 0.25);
		assertTrue(errorRate2 < 0.25);
	}


	
	@Test
	public void TestMVSequence3_3() throws Exception
	{
		//System.out.println("about to sample training set");
		Instances train = getMVSequence3(100, 100);
		//System.out.println("about to sample testing set");
		Instances test1 = getMVSequence3(50, 100);
		//Instances test2 = getSequenceUniform(20, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(4);
		hmm.setNumeric(true);
		//hmm.setNumOutputs(6);
	
		//hmm.initEstimators(2);


		//System.out.println("about to build classifier");
		hmm.buildClassifier(train);

		//System.out.println("about to evaluate");
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate1 = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 3_3 error rate " + errorRate1);
		// quite a high error rate as the models are similar
		//assertTrue(errorRate < 0.25);
		
		eval.evaluateModel(hmm, test1);
		double errorRate2 = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 3_3 error rate " + errorRate2);
		// quite a high error rate as the models are similar
		assertTrue(errorRate1 < 0.25);
		assertTrue(errorRate2 < 0.25);
	}
	

	@Test
	public void TestMVSequence3_4() throws Exception
	{
		//System.out.println("about to sample training set");
		Instances train = getMVSequence3(100, 100);
		//System.out.println("about to sample testing set");
		Instances test1 = getMVSequence3(50, 100);
		//Instances test2 = getSequenceUniform(20, 100);
		
		HMM hmm = new HMM();
		
		hmm.setNumStates(4);
		hmm.setNumeric(true);
		hmm.setLeftRight(true);
		//hmm.setNumOutputs(6);
	
		//hmm.initEstimators(2);


		//System.out.println("about to build classifier");
		hmm.buildClassifier(train);

		//System.out.println("about to evaluate");
		Evaluation eval = new Evaluation(train);
		
		eval.evaluateModel(hmm, train);
		double errorRate1 = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 3_3 error rate " + errorRate1);
		// quite a high error rate as the models are similar
		//assertTrue(errorRate < 0.25);
		
		eval.evaluateModel(hmm, test1);
		double errorRate2 = eval.errorRate();
		if(printErrorRates)
			System.out.println("Test MV Seq 3_3 error rate " + errorRate2);
		// quite a high error rate as the models are similar
		assertTrue(errorRate1 < 0.25);
		assertTrue(errorRate2 < 0.25);
	}

	
	@Test
	public void TestMultiVariateNormalEstimator() throws Exception
	{
		// model 1 is a zero mean unit variance gaussian
		MultivariateNormalEstimator model1 = new MultivariateNormalEstimator();
		model1.setMean(new DoubleVector(4, 0.0));
		model1.setVariance(Matrix.identity(4, 4));
		
		DoubleVector train1 [] = new DoubleVector[1000];
		for (int i = 0; i < train1.length; i++)
		{
			train1[i] = model1.sample();
		}
		DoubleVector test1 [] = new DoubleVector[1000];
		for (int i = 0; i < train1.length; i++)
		{
			test1[i] = model1.sample();
		}
	

		// model2 has different means and covariances
		MultivariateNormalEstimator model2 = new MultivariateNormalEstimator();
		DoubleVector mean = DoubleVector.random(4);
		model2.setMean(mean);
		Matrix m = Matrix.identity(4, 4);
		for (int i = 0; i < 3; i++)
		{
			m.set(i,i, 2.0);
			m.set(i, i+1, -1);
			m.set(i+1, i, -1);
		}
		m.set(3,3, 2);
		//m.set(0, 1, 1);
		//m.set(1, 0, 1);
		model2.setVariance(m);
		
		
		DoubleVector train2 [] = new DoubleVector[1000];
		for (int i = 0; i < train2.length; i++)
		{
			train2[i] = model2.sample();
		}
		DoubleVector test2 [] = new DoubleVector[1000];
		for (int i = 0; i < test2.length; i++)
		{
			test2[i] = model2.sample();
		}
		
		MultivariateNormalEstimator est1 = new MultivariateNormalEstimator();
		for (int i = 0; i < train1.length; i++)
		{
			est1.addValue(train1[i], 1.0);
		}
		MultivariateNormalEstimator est2 = new MultivariateNormalEstimator();
		for (int i = 0; i < train2.length; i++)
		{
			est2.addValue(train2[i], 1.0);
		}
		
		double lik1 = 0.0;
		double lik2 = 0.0;
		for (int i = 0; i < test1.length; i++)
		{
			lik1 += Math.log(est1.getProbability(test1[i]));
			lik2 += Math.log(est2.getProbability(test1[i]));
		}
		assertTrue(lik1 > lik2);
		
		lik1 = 0.0;
		lik2 = 0.0;
		for (int i = 0; i < test2.length; i++)
		{
			lik1 += Math.log(est1.getProbability(test2[i]));
			lik2 += Math.log(est2.getProbability(test2[i]));
		}
		assertTrue(lik1 < lik2);
	}
	

}

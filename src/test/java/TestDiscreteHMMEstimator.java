/**
 * 
 */
package weka.estimators;

import static org.junit.Assert.*;

import java.util.Random;

import org.junit.Before;
import org.junit.Test;

import weka.core.matrix.DoubleVector;

/**
 * @author marco
 *
 */
public class TestDiscreteHMMEstimator {

	Random m_rand;
	
	@Before
	public void setUp() throws Exception {
		m_rand = new Random(13411);
	}
	
	/**
	 * Test method for {@link weka.estimators.DiscreteHMMEstimator#setNumOutputs(int)}.
	 */
	@Test
	public void testSetNumOutputs() {
		DiscreteHMMEstimator dhe = new DiscreteHMMEstimator(4, 4, false);
		assertEquals(dhe.getNumOutputs(), 4);
		
		dhe.setNumOutputs(5);
		assertEquals(dhe.getNumOutputs(), 5);
		
		dhe.setNumOutputs(0);
		assertEquals(dhe.getNumOutputs(), 0);
		
		dhe.setNumOutputs(1);
		assertEquals(dhe.getNumOutputs(), 1);
		
		dhe.setNumOutputs(345);
		assertEquals(dhe.getNumOutputs(), 345);
		
	}

	/**
	 * Test method for {@link weka.estimators.DiscreteHMMEstimator#DiscreteHMMEstimator(int, int, boolean)}.
	 */
	@Test
	public void testDiscreteHMMEstimatorIntIntBoolean() {
		DiscreteHMMEstimator dhe = new DiscreteHMMEstimator(17, 14, false);
		assertEquals(dhe.getNumOutputs(), 14);
		assertEquals(dhe.getNumStates(), 17);
		dhe = new DiscreteHMMEstimator(6, 3, false);
		assertEquals(dhe.getNumOutputs(), 3);
		assertEquals(dhe.getNumStates(), 6);
	}

	/**
	 * Test method for {@link weka.estimators.DiscreteHMMEstimator#DiscreteHMMEstimator(weka.estimators.DiscreteHMMEstimator)}.
	 * @throws Exception 
	 */
	@Test
	public void testDiscreteHMMEstimatorDiscreteHMMEstimator() throws Exception {
		DiscreteHMMEstimator dhe1 = new DiscreteHMMEstimator(4, 4, false);
		for(int i = 0; i < 4; i ++)
			for (int k = 0; k < 4; k++)
				dhe1.addValue0(i, k, m_rand.nextDouble());
		for(int i = 0; i < 4; i ++)
			for(int j = 0; j < 4; j++)
				for (int k = 0; k < 4; k++)
					dhe1.addValue(i, j, k, m_rand.nextDouble());
		
		DiscreteHMMEstimator dhe2 = new DiscreteHMMEstimator(dhe1);
		assertEquals(dhe1.getNumOutputs(), dhe2.getNumOutputs());
		assertEquals(dhe2.getNumStates(), dhe2.getNumStates());
		for(int i = 0; i < 4; i ++)
			for (int k = 0; k < 4; k++)
				assertEquals(dhe1.getProbability0(i, k),dhe2.getProbability0(i, k), 0.01) ;
		for(int i = 0; i < 4; i ++)
			for(int j = 0; j < 4; j++)
				for (int k = 0; k < 4; k++)
					assertEquals(dhe1.getProbability(i, j, k),dhe2.getProbability(i,j , k), 0.01) ;
		
	}

	/**
	 * Test method for {@link weka.estimators.DiscreteHMMEstimator#addValue(double, double, weka.core.matrix.DoubleVector, double)}.
	 */
	@Test
	public void testAddValueDoubleDoubleDoubleVectorDouble() {
		DiscreteHMMEstimator dhe1 = new DiscreteHMMEstimator(4, 4, false);
		DiscreteHMMEstimator dhe2 = new DiscreteHMMEstimator(4, 4, false);
		for(int i = 0; i < 4; i ++)
			for (int k = 0; k < 4; k++)
			{
				Double w = m_rand.nextDouble();
				dhe1.addValue0(i, k, w);
				DoubleVector outVec = new DoubleVector(1, k);
				dhe2.addValue0(i, outVec, w);
			}
		for(int i = 0; i < 4; i ++)
			for(int j = 0; j < 4; j++)
				for (int k = 0; k < 4; k++)
				{
					Double w = m_rand.nextDouble();
					dhe1.addValue(i, j, k, w);
					DoubleVector outVec = new DoubleVector(1, k);
					dhe2.addValue(i, j, outVec, w);
				}
		

		for (int i = 0; i < 30; i++)
		{
			int s = m_rand.nextInt(4);
			int o = m_rand.nextInt(4);

			assertEquals(dhe1.getProbability0(s, o), dhe2.getProbability0(s, o), 0.001);
		}
		
		for (int i = 0; i < 30; i++)
		{
			int s = m_rand.nextInt(4);
			int s1 = m_rand.nextInt(4);
			int o = m_rand.nextInt(4);

			assertEquals(dhe1.getProbability(s, s1, o), dhe2.getProbability(s, s1, o), 0.001);
		}
	}

	/**
	 * Test method for {@link weka.estimators.DiscreteHMMEstimator#getProbability(double, double, weka.core.matrix.DoubleVector)}.
	 */
	@Test
	public void testGetProbabilityDoubleDoubleDoubleVector() {
		DiscreteHMMEstimator dhe1 = new DiscreteHMMEstimator(4, 4, false);
		for(int i = 0; i < 4; i ++)
			for (int k = 0; k < 4; k++)
			{
				//Double w = m_rand.nextDouble();
				dhe1.addValue0(i, k, m_rand.nextDouble());
			}
		for(int i = 0; i < 4; i ++)
			for(int j = 0; j < 4; j++)
				for (int k = 0; k < 4; k++)
					dhe1.addValue(i, j, k, m_rand.nextDouble());

		for (int i = 0; i < 30; i++)
		{
			int s = m_rand.nextInt(4);
			int o = m_rand.nextInt(4);
			DoubleVector outVec = new DoubleVector(1, o);
			assertEquals(dhe1.getProbability0(s, o), dhe1.getProbability0(s, outVec), 0.001);
		}
		
		for (int i = 0; i < 30; i++)
		{
			int s = m_rand.nextInt(4);
			int s1 = m_rand.nextInt(4);
			int o = m_rand.nextInt(4);
			DoubleVector outVec = new DoubleVector(1, o);
			assertEquals(dhe1.getProbability(s, s1, o), dhe1.getProbability(s, s1, outVec), 0.001);
		}
	}

	/**
	 * Test method for {@link weka.estimators.DiscreteHMMEstimator#addValue(double, double, double, double)}.
	 */
	@Test
	public void testAddValueDoubleDoubleDoubleDouble() {

		for(int testRun = 0; testRun < 10; testRun++)
		{
			DiscreteHMMEstimator dhe1 = new DiscreteHMMEstimator(m_rand.nextInt(20), m_rand.nextInt(20), false);
			double weights[][][] = new double[dhe1.getNumStates()][dhe1.getNumStates()][dhe1.getNumOutputs()];
			double sum = 0;
			double stateSum[] = new double[dhe1.getNumStates()];
			double stateProb[][] = new double[dhe1.getNumStates()][dhe1.getNumStates()];
			double outputSum[] = new double[dhe1.getNumStates()];
			double outputProb[][] = new double[dhe1.getNumStates()][dhe1.getNumOutputs()];
			for(int s = 0; s < dhe1.getNumStates(); s++)
			{
				//sum = 0.0;
				for(int s1 = 0; s1 < dhe1.getNumStates(); s1++)
				{
	
					for (int o = 0; o < dhe1.getNumOutputs(); o++)
					{
						
						weights[s][s1][o] = m_rand.nextDouble();
						sum += weights[s][s1][o];
						dhe1.addValue(s, s1, o, weights[s][s1][o]);
						
						stateSum[s] += weights[s][s1][o];
						stateProb[s][s1] += weights[s][s1][o];
						outputSum[s1] += weights[s][s1][o];
						outputProb[s1][o] += weights[s][s1][o];
					}
				}
				//for(int s1 = 0; s1 < dhe1.getNumStates(); s1++)
				//	for (int o = 0; o < dhe1.getNumOutputs(); o++)
				//	{
				//		weights[s][s1][o] /= sum;
				//	}
			}
			
			
			
			for(int s = 0; s < dhe1.getNumStates(); s ++)
			{
				for(int s1 = 0; s1 < dhe1.getNumStates(); s1++)
				{
					for (int o = 0; o < dhe1.getNumOutputs(); o++)
					{
						System.out.println(dhe1.getProbability(s, s1, o) + " " + weights[s][s1][o]);
						double prob = stateProb[s][s1]/stateSum[s];
						prob *= outputProb[s1][o]/outputSum[s1];
						assertEquals("probability " + s + " " + s1 + " " + o + " " + stateSum[s], 
								dhe1.getProbability(s, s1, o), prob, 0.001);
					}
				}
			}
		}

//		dhe1 = new DiscreteHMMEstimator(m_rand.nextInt(20), m_rand.nextInt(20), false);
//		weights = new double[dhe1.getNumStates()][dhe1.getNumStates()][dhe1.getNumOutputs()];
//		sum = 0;
//		for(int s = 0; s < dhe1.getNumStates(); s ++)
//		{
//			sum = 0.0;
//			for(int s1 = 0; s1 < dhe1.getNumStates(); s1 ++)
//				for (int o = 0; o < dhe1.getNumOutputs(); o++)
//			{
//				weights[s][s1][o] = m_rand.nextInt(100);
//				sum += weights[s][s1][o];
//				for(int i = 0; i < weights[s][s1][o]; i++)	
//					dhe1.addValue(s, s1, o, 1.0);
//			}
//			//for(int s1 = 0; s1 < dhe1.getNumStates(); s1 ++)
//			//	for (int o = 0; o < dhe1.getNumOutputs(); o++)
//			//{
//			//	weights[s][s1][o] /= sum;
//			//}
//		}
//		for(int s = 0; s < dhe1.getNumStates(); s ++)
//			for(int s1 = 0; s1 < dhe1.getNumStates(); s1 ++)
//				for (int o = 0; o < dhe1.getNumOutputs(); o++)
//			{
//				assertEquals(dhe1.getProbability(s, s1, o), weights[s][s1][o], 0.001);
//			}
	}

	/**
	 * Test method for {@link weka.estimators.DiscreteHMMEstimator#addValue0(double, double, double)}.
	 */
	@Test
	public void testAddValue0DoubleDoubleDouble() {
		DiscreteHMMEstimator dhe1 = new DiscreteHMMEstimator(m_rand.nextInt(20), m_rand.nextInt(20), false);
		double weights[][] = new double[dhe1.getNumStates()][dhe1.getNumOutputs()];
		double sum = 0;
		for(int s = 0; s < dhe1.getNumStates(); s ++)
			for (int o = 0; o < dhe1.getNumOutputs(); o++)
			{
				weights[s][o] = m_rand.nextDouble();
				sum += weights[s][o];
				dhe1.addValue0(s, o, weights[s][o]);
			}
		for(int s = 0; s < dhe1.getNumStates(); s ++)
			for (int o = 0; o < dhe1.getNumOutputs(); o++)
			{
				weights[s][o] /= sum;
			}
		for(int s = 0; s < dhe1.getNumStates(); s ++)
			for (int o = 0; o < dhe1.getNumOutputs(); o++)
			{
				assertEquals(dhe1.getProbability0(s, o), weights[s][o], 0.01);
			}

		dhe1 = new DiscreteHMMEstimator(m_rand.nextInt(20),m_rand.nextInt(20), false);
		weights = new double[dhe1.getNumStates()][dhe1.getNumOutputs()];
		sum = 0;
		for(int s = 0; s < dhe1.getNumStates(); s ++)
			for (int o = 0; o < dhe1.getNumOutputs(); o++)
			{
				weights[s][o] = m_rand.nextInt(100);
				sum += weights[s][o];
				for(int i = 0; i < weights[s][o]; i++)	
					dhe1.addValue0(s, o, 1.0);
			}
		for(int s = 0; s < dhe1.getNumStates(); s ++)
			for (int o = 0; o < dhe1.getNumOutputs(); o++)
			{
				weights[s][o] /= sum;
			}
		for(int s = 0; s < dhe1.getNumStates(); s ++)
			for (int o = 0; o < dhe1.getNumOutputs(); o++)
			{
				assertEquals(dhe1.getProbability0(s, o), weights[s][o], 0.01);
			}
	}

	/**
	 * Test method for {@link weka.estimators.DiscreteHMMEstimator#getProbability(double, double, double)}.
	 */
	@Test
	public void testGetProbabilityDoubleDoubleDouble() {
		DiscreteHMMEstimator dhe1 = new DiscreteHMMEstimator(4, 4, false);
		for(int i = 0; i < 4; i ++)
			for (int k = 0; k < 4; k++)
			{
				//Double w = m_rand.nextDouble();
				dhe1.addValue0(i, k, m_rand.nextDouble());
			}
		for(int i = 0; i < 4; i ++)
			for(int j = 0; j < 4; j++)
				for (int k = 0; k < 4; k++)
					dhe1.addValue(i, j, k, m_rand.nextDouble());

		double sum = 0.0;
		for(int s = 0; s < 4; s ++)
		{
			for (int o = 0; o < 4; o++)
			{
				sum += dhe1.getProbability0(s, o);
				assertTrue(dhe1.getProbability0(s, o) >= 0);
				assertTrue(dhe1.getProbability0(s, o) <= 1);
			}
		}
		assertEquals(sum, 1.0, 0.001);
		
		for(int s = 0; s < 4; s ++)
		{
			sum = 0.0;
			for(int s1 = 0; s1 < 4; s1++)
			{
				for (int o = 0; o < 4; o++)
				{
					sum += dhe1.getProbability(s, s1, o);
					assertTrue(dhe1.getProbability(s, s1, o) >= 0);
					assertTrue(dhe1.getProbability(s, s1, o) <= 1);
				}
			}
			assertEquals(sum, 1.0, 0.001);
		}
	}

	/**
	 * Test method for {@link weka.estimators.DiscreteHMMEstimator#Sample0(weka.core.Instances, java.util.Random)}.
	 */
	
//	@Test
//	public void testSample0() {
//		fail("Not yet implemented");
//	}

	/**
	 * Test method for {@link weka.estimators.DiscreteHMMEstimator#Sample(weka.core.Instances, int, java.util.Random)}.
	 */
//	@Test
//	public void testSample() {
//		fail("Not yet implemented");
//	}

	/**
	 * Test method for {@link weka.estimators.AbstractHMMEstimator#setNumStates(int)}.
	 */
	@Test
	public void testSetNumStates() {

		DiscreteHMMEstimator dhe = new DiscreteHMMEstimator(4, 4, false);
		assertEquals(dhe.getNumStates(), 4);
		
		dhe.setNumStates(5);
		assertEquals(dhe.getNumStates(), 5);
		
		dhe.setNumStates(0);
		assertEquals(dhe.getNumStates(), 0);
		
		dhe.setNumStates(1);
		assertEquals(dhe.getNumStates(), 1);
		
		dhe.setNumStates(345);
		assertEquals(dhe.getNumStates(), 345);
	}

	/**
	 * Test method for {@link weka.estimators.AbstractHMMEstimator#getOutputDimension()}.
	 */
	@Test
	public void testGetOutputDimension() {
		DiscreteHMMEstimator dhe = new DiscreteHMMEstimator(4, 4, false);
		assertEquals(dhe.getOutputDimension(), 1);
	}

}

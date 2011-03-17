/**
 * 
 */
package weka.estimators;

import static org.junit.Assert.*;

import org.junit.Test;

import weka.core.matrix.DoubleVector;
import weka.core.matrix.Matrix;

/**
 * @author marco
 *
 */
public class TestMultivariateNormalEstimator {

	/**
	 * Test method for {@link weka.estimators.MultivariateNormalEstimator#setCovarianceType(weka.estimators.MultivariateNormalEstimator.CovarianceType)}.
	 * @throws Exception 
	 */
	@Test
	public void testSetCovarianceType() throws Exception {
		MultivariateNormalEstimator mne = new MultivariateNormalEstimator();
		assertEquals(mne.getCovarianceType(), MultivariateNormalEstimator.COVARIANCE_FULL);
		
		mne.setCovarianceType(MultivariateNormalEstimator.COVARIANCE_DIAGONAL);
		assertEquals(mne.getCovarianceType(), MultivariateNormalEstimator.COVARIANCE_DIAGONAL);
		
		for(int i = 0; i < 50.0; i++)
		{
			DoubleVector v = DoubleVector.random(6);
			mne.addValue(v, 1.0);
		}
		mne.calculateParameters();
		Matrix cov = mne.getVariance();
		for(int i = 0; i < 6; i ++)
			for (int j = 0; j < 6; j++)
			{
				if(i!=j)
					assertEquals(cov.get(i,j), 0.0, 0.0001);
			}
		
		mne.setCovarianceType(MultivariateNormalEstimator.COVARIANCE_SPHERICAL);
		assertEquals(mne.getCovarianceType(), MultivariateNormalEstimator.COVARIANCE_SPHERICAL);
		
		for(int i = 0; i < 50.0; i++)
		{
			DoubleVector v = DoubleVector.random(6);
			mne.addValue(v, 1.0);
		}
		mne.calculateParameters();
		cov = mne.getVariance();
		for(int i = 0; i < 6; i ++)
			for (int j = 0; j < 6; j++)
			{
				if(i!=j)
					assertEquals(cov.get(i,j), 0.0, 0.0001);
				else
					assertEquals(cov.get(i,j), cov.get(0,0), 0.0001);
			}
		
		mne.setCovarianceType(MultivariateNormalEstimator.COVARIANCE_FULL);
		assertEquals(mne.getCovarianceType(), MultivariateNormalEstimator.COVARIANCE_FULL);
	}

	/**
	 * Test method for {@link weka.estimators.MultivariateNormalEstimator#MultivariateNormalEstimator()}.
	 * just checks that it doesn't throw and exception
	 */
	@Test
	public void testMultivariateNormalEstimator() {
		MultivariateNormalEstimator mne = new MultivariateNormalEstimator();
		assertEquals(mne.getCovarianceType(), MultivariateNormalEstimator.COVARIANCE_FULL);
	}
	
	void assertVectorsEqual(DoubleVector v1, DoubleVector v2, double epsilon)
	{
		DoubleVector diff = v1.minus(v2);
		assertTrue(Math.abs(diff.norm2()/v1.norm2()) < epsilon);
	}
	

	void assertMatricesEqual(Matrix m1, Matrix m2, double epsilon)
	{
		Matrix diff = m1.minus(m2);
		assertTrue(Math.abs(diff.normF()/m1.normF()) < epsilon);
	}

	/**
	 * Test method for {@link weka.estimators.MultivariateNormalEstimator#MultivariateNormalEstimator(weka.estimators.MultivariateNormalEstimator)}.
	 * @throws Exception 
	 */
	@Test
	public void testMultivariateNormalEstimatorMultivariateNormalEstimator() throws Exception {
		MultivariateNormalEstimator mne1 = new MultivariateNormalEstimator();
		DoubleVector mean = DoubleVector.random(6);
		mne1.setMean(mean);
		
		Matrix cov = Matrix.identity(6, 6);
		for (int i = 0; i < 5; i++)
		{
			cov.set(i,i, 2.0);
			cov.set(i, i+1, -1);
			cov.set(i+1, i, -1);
		}
		cov.set(5,5, 2);
		mne1.setVariance(cov);
		
		MultivariateNormalEstimator mne2 = new MultivariateNormalEstimator(mne1);
		assertVectorsEqual( mne1.getMean(), mne2.getMean(), 0.001);
		assertVectorsEqual( mean, mne2.getMean(), 0.001);
		
		assertMatricesEqual(mne1.getVariance(), mne2.getVariance(), 0.001);
		assertMatricesEqual(cov, mne2.getVariance(), 0.001);
	}

	/**
	 * Test method for {@link weka.estimators.MultivariateNormalEstimator#setMean(weka.core.matrix.DoubleVector)}.
	 */
	@Test
	public void testSetMean() {
		MultivariateNormalEstimator mne = new MultivariateNormalEstimator();
		DoubleVector mean = DoubleVector.random(6);
		mne.setMean(mean);
		
		assertVectorsEqual( mean, mne.getMean(), 0.001);
	}

	/**
	 * Test method for {@link weka.estimators.MultivariateNormalEstimator#setVariance(weka.core.matrix.Matrix)}.
	 */
	@Test
	public void testSetVariance() {
		MultivariateNormalEstimator mne = new MultivariateNormalEstimator();
		DoubleVector mean = DoubleVector.random(6);
		mne.setMean(mean);
		
		Matrix cov = Matrix.identity(6, 6);
		for (int i = 0; i < 5; i++)
		{
			cov.set(i,i, 2.0);
			cov.set(i, i+1, -1);
			cov.set(i+1, i, -1);
		}
		cov.set(5,5, 2);
		mne.setVariance(cov);
		
		assertMatricesEqual(cov, mne.getVariance(), 0.001);
	}

	/**
	 * Test method for {@link weka.estimators.MultivariateNormalEstimator#addValue(weka.core.matrix.DoubleVector, double)}.
	 * @throws Exception 
	 */
	@Test
	public void testAddValue() throws Exception {
		MultivariateNormalEstimator mne = new MultivariateNormalEstimator();
		
		DoubleVector avg = new DoubleVector(6,0.0);
		for(int i = 0; i < 50.0; i++)
		{
			DoubleVector v = DoubleVector.random(6);
			mne.addValue(v, 1.0);
			avg.plusEquals(v);
		}
		avg.timesEquals(1.0/50.0);
		
		mne.calculateParameters();
		
		assertVectorsEqual(avg, mne.getMean(), 0.001);
	}

	/**
	 * Test method for {@link weka.estimators.MultivariateNormalEstimator#getProbability(weka.core.matrix.DoubleVector)}.
	 * @throws Exception 
	 */
	@Test
	public void testGetProbability() throws Exception {
		MultivariateNormalEstimator mne = new MultivariateNormalEstimator();
		DoubleVector mean = DoubleVector.random(6);
		DoubleVector other = DoubleVector.random(6);
		mne.setMean(mean);
		
		Matrix cov = Matrix.identity(6, 6);
		for (int i = 0; i < 5; i++)
		{
			cov.set(i,i, 2.0);
			cov.set(i, i+1, -1);
			cov.set(i+1, i, -1);
		}
		cov.set(5,5, 2);
		mne.setVariance(cov);

		double meanProb = mne.getProbability(mean);
		double otherProb = mne.getProbability(other);
		
		assertTrue(meanProb >= 0);
		assertTrue(meanProb <= 1);
		assertTrue(otherProb >= 0);
		assertTrue(otherProb <= 1);
		
		assertTrue(meanProb > otherProb);
	}

	/**
	 * Test method for {@link weka.estimators.MultivariateNormalEstimator#sample()}.
	 * @throws Exception 
	 */
	@Test
	public void testSample() throws Exception {
		MultivariateNormalEstimator mne1 = new MultivariateNormalEstimator();
		DoubleVector mean = DoubleVector.random(6);
		mne1.setMean(mean);
		
		Matrix cov = Matrix.identity(6, 6);
		for (int i = 0; i < 5; i++)
		{
			cov.set(i,i, 2.0);
			cov.set(i, i+1, -1);
			cov.set(i+1, i, -1);
		}
		cov.set(5,5, 2);
		mne1.setVariance(cov);
		
		MultivariateNormalEstimator mne2 = new MultivariateNormalEstimator();
		for(int i = 0; i < 2000; i ++)
		{
			DoubleVector v = mne1.sample();
			mne2.addValue(v, 1.0);
		}
		mne2.calculateParameters();

		assertVectorsEqual( mne1.getMean(), mne2.getMean(), 0.1);
		assertMatricesEqual(mne1.getVariance(), mne2.getVariance(), 0.1);
	}


}

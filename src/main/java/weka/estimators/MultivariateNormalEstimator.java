package weka.estimators;

import java.io.Serializable;

import weka.core.matrix.*;

public class MultivariateNormalEstimator implements Serializable {

	//public static enum CovarianceType
    //{
    //    FULL,
    //    DIAGONAL,
    //    SPHERICAL,
    //    ;
    //}
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 5472266864312430693L;
	
	
	/** Covariance Type: Full matrix (unconstrained) */
	public static final int COVARIANCE_FULL = 0;
	/** Covariance Type: Diagonal matrix (no correlation between data attributes)  */
	public static final int COVARIANCE_DIAGONAL = 1;
	/** Covariance Type:       */
	public static final int COVARIANCE_SPHERICAL = 2;
	
	// sufficient statistics
	double 			m_SumOfWeights;
	SerializableDoubleVector 	m_SumOfValues;
	Matrix 		 	m_SumOfSquareValues;
	int 			m_NumObservations;
	
	SerializableDoubleVector m_Mean;
	Matrix m_Var;
	Matrix m_InvVar;
	Matrix m_CholeskyL;
	double m_DetVar;
	
	protected int m_CovarianceType = COVARIANCE_FULL;
	
	public int getCovarianceType() {
		return m_CovarianceType;
	}

	public void setCovarianceType(int Type) {
		this.m_CovarianceType = Type;
	}


	boolean m_Dirty = false;
	
	public MultivariateNormalEstimator()
	{
		m_Dirty = true;
	}
	
	public MultivariateNormalEstimator(MultivariateNormalEstimator e) throws Exception
	{
		if(e.m_Dirty)
			e.calculateParameters();
		m_Mean = new SerializableDoubleVector(e.m_Mean.copy());
		m_Var = e.m_Var.copy();
		m_InvVar = e.m_InvVar.copy();
		m_CholeskyL = e.m_CholeskyL.copy();
		m_DetVar = e.m_DetVar;
		m_CovarianceType = e.m_CovarianceType;
		m_Dirty = false;
	}
	
	int getDimension()
	{
		if (m_Mean != null)
			return m_Mean.size();
		else
			return 0;
	}
	
	void init(int n)
	{
		m_SumOfWeights = 0.0;
		m_SumOfValues = new SerializableDoubleVector(new DoubleVector(n, 0.0));
		m_SumOfSquareValues = new Matrix(n,n, 0.0);
	}
	
	void calculateVarianceFull()
	{
		 m_Var = new Matrix(m_Mean.size(), m_Mean.size());
	      for (int i = 0; i < m_Mean.size(); i++)
			  for (int j = 0; j < m_Mean.size(); j++)
			  {
				  m_Var.set(i,j, (m_SumOfSquareValues.get(i,j)/m_SumOfWeights));
			  }

	      for (int i = 0; i < m_Mean.size(); i++)
			  for (int j = 0; j < m_Mean.size(); j++)
			  {
				  m_Var.set(i,j, m_Var.get(i,j) - m_Mean.get(i) * m_Mean.get(j));
			  }
	}
	
	void calculateVarianceDiagonal()
	{
		 m_Var = new Matrix(m_Mean.size(), m_Mean.size());
	      for (int i = 0; i < m_Mean.size(); i++)
			  {
				  m_Var.set(i,i, (m_SumOfSquareValues.get(i,i)/m_SumOfWeights));
			  }

	      for (int i = 0; i < m_Mean.size(); i++)
			  {
				  m_Var.set(i,i, m_Var.get(i,i) - m_Mean.get(i) * m_Mean.get(i));
			  }
	}
	
	void calculateVarianceSpherical()
	{
		double sigma = 0;
		for(int i = 0; i < m_Mean.size(); i++)
		{
			//for (int j = 0; j < m_Mean.size(); j++)
			sigma += m_SumOfSquareValues.get(i,i);
		}
		sigma = sigma/m_SumOfWeights;
		for(int i = 0; i < m_Mean.size(); i++)
			sigma -= m_Mean.get(i)*m_Mean.get(i);
		sigma = sigma/m_Mean.size();
		
		m_Var = Matrix.identity(m_Mean.size(), m_Mean.size());
		m_Var.timesEquals(sigma);
	}
	
	public void calculateParameters() throws Exception
	{

		m_Dirty = false; 
		if (m_SumOfWeights > 0.00001) {
		      m_Mean = new SerializableDoubleVector(m_SumOfValues.times(1.0 / m_SumOfWeights));
		      
		      switch(getCovarianceType())
		      {
		      case COVARIANCE_FULL:
		    	  calculateVarianceFull();
		    	  break;
		      case COVARIANCE_DIAGONAL:
		    	  calculateVarianceDiagonal();
		    	  break;
		      case COVARIANCE_SPHERICAL:
		    	  calculateVarianceSpherical();
		    	  break;
		      default:
		    	  throw new Exception("Unhandled covariance type");
		      }
		      
			  m_DetVar = m_Var.det();
			  if (m_DetVar < 1.0E-200)
			  {
				//System.out.println(this);
				  //throw new Exception("Covariance matrix has zero determinant");
				  System.err.println("Covariance matrix has zero determinant");
				  return;
			  }
		      m_InvVar = m_Var.inverse();
			  m_CholeskyL = m_Var.chol().getL();
			  
		}
 
	}
	
	public static void calculateTiedParameters(MultivariateNormalEstimator ests[]) throws Exception
	{
		if (ests.length == 0)
			return;
		Matrix Sigma = new Matrix(ests[0].m_Mean.size(),ests[0].m_Mean.size(),0.0);
		double M = 0;
		for(int i = 0; i < ests.length; i++)
		{
			ests[i].calculateParameters();
			Sigma.plusEquals(ests[i].m_Var.times(ests[i].m_SumOfWeights));
			M += ests[i].m_SumOfWeights;
		}
		Sigma.timesEquals(1.0/M);
		for(int i = 0; i < ests.length; i++)
		{
			ests[i].m_Var = Sigma.copy();
		}
	}
	
	public 	DoubleVector getMean()
	{
		return m_Mean;
	}
	
	public 	void setMean(DoubleVector v)
	{
		m_Mean = new SerializableDoubleVector(v.copy());
	}
	
	public 	Matrix getVariance()
	{
		return m_Var;
	}
	
	public 	void setVariance(Matrix m)
	{
		m_Var = m.copy();
	    m_InvVar = m_Var.inverse();
	    m_DetVar = m_Var.det();
	    m_CholeskyL = m_Var.chol().getL();
	}
	
	
	
	
	  public void addValue(DoubleVector data, double weight)
	  {
		  if (weight == 0) {
		      return;
		  }

		  if(m_SumOfValues == null)
			  init(data.size());
		  m_SumOfWeights += weight;
		  m_SumOfValues.plusEquals(data.times(weight));
		  m_NumObservations += 1;
		  for (int i = 0; i < data.size(); i++)
			  for (int j = 0; j < data.size(); j++)
			  {
				  m_SumOfSquareValues.set(i,j, m_SumOfSquareValues.get(i,j) + data.get(i) * data.get(j) * weight);
			  }
		  //System.out.println(m_SumOfValues + "      " + data);
		  // make sure you recalculate the mean
		  m_Dirty = true;
	  }
	  


	
	  public double getProbability(DoubleVector data) throws Exception
	  {
		  if(m_Dirty)
			  calculateParameters();
		  
		  if (m_DetVar < 1.0E-200)
		  {
			  return 0.0;
		  }
		  
		  double coef = 1.0/Math.pow(2*Math.PI, m_Mean.size()/2.0);
		  coef = coef * 1.0/(Math.sqrt(m_DetVar));
		  
		  double product = 0.0;
		  for (int i = 0; i < m_Mean.size(); i++)
			  for (int j = 0; j < m_Mean.size(); j++)
				  product += (data.get(i)-m_Mean.get(i)) * m_InvVar.get(i, j) * (data.get(j)-m_Mean.get(j));
		  
		  double p = coef*Math.exp(-0.5*product);
		  if (Double.isInfinite(p) || Double.isNaN(p))
				throw new Exception("Calculated probability is NaN");
		  return p;
	  }
	  
	  public DoubleVector boxMuller()
	  {
		  DoubleVector v;
		  double r2 = 0.0;
		  do
		  {
			v  = DoubleVector.random(2); 
			v.timesEquals(2.0);
			v.minusEquals(1);
			r2 = v.sum2();
		  }while(r2>1.0);
		  for(int i = 0; i < v.size(); i++)
		  {
			  //v.set(i, v.get(i)*Math.sqrt(-2*Math.log(Math.abs(v.get(i)))/r2));
			  v.set(i, v.get(i)*Math.sqrt(-2*Math.log(r2)/r2));
		  }
		  return v;
	  }
	
	  public DoubleVector sample()
	  {
		  //get a vector of samples between -1 and 10
		  DoubleVector v = new DoubleVector(m_Mean.size());
		  for (int i = 0; i < v.size()/2; i++)
		  {
			  DoubleVector pair = boxMuller();
			  v.set(2*i, pair.get(0));
			  v.set(2*i+1, pair.get(1));
		  }
		  if(v.size()%2 == 1)
		  {
			  DoubleVector pair = boxMuller();
			  v.set(v.size()-1, pair.get(0));
		  }  
		  DoubleVector result = new DoubleVector(m_Mean.size());
		  for(int i = 0; i < result.size(); i++)
			  for(int j = 0; j < result.size(); j++)
				  result.set(i, result.get(i)+m_CholeskyL.get(i, j)*v.get(j));// + m_Mean.get(i));
		  result.plusEquals(m_Mean);
		  //System.out.println(result);
		  return result;
	  }
	
	  
	  public String  toString() {
		   String covString = "";
		   switch(getCovarianceType())
		      {
		      case COVARIANCE_FULL:
		    	  covString  = m_Var.toString();
		    	  break;
		      case COVARIANCE_DIAGONAL:
		    	  for (int i = 0; i < m_Mean.size(); i++)
		    		  covString += m_Var.get(i, i) + " ";
		    	  break;
		      case COVARIANCE_SPHERICAL:
		    	  covString += m_Var.get(0, 0);
		    	  break;
		      default:
		    	  break;
		      }
		   return "Mean\n" + m_Mean.toString() + "\nCovariance\n" + covString;
	  }
}

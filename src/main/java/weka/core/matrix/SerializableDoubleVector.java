package weka.core.matrix;

import java.io.Serializable;

/**
 * This class simply adds the Serializable interface to Weka's DoubleVector class
 */
public class SerializableDoubleVector extends DoubleVector implements
		Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1399306906998193091L;

	public SerializableDoubleVector(DoubleVector v)
	{
		super(v.size());
		set(v);
	}
	
}

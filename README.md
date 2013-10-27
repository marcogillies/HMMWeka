HMMWeka
=======

A Hidden Markov Model package for the Weka machine learning toolkit.

This is a standard Weka classifier that can be used in all the standard weka tools (with the caveats listed below). See the [Weka documentation](http://www.cs.waikato.ac.nz/ml/weka/) for more details of how to run weka. Full documentation on the API for this package can be found in the [javadoc reference.](http://doc.gold.ac.uk/~mas02mg/software/hmmweka/HMM/doc/index.html)

Hidden Markov Models are used differently from other classifiers as they work on temporal sequences of data rather than on individual data items. It is therefore important to have a good understanding of HMMs before using them, you cannot just use them in place of another Weka classifier. There are plenty of good resources on Hidden Markov Models, so I won't attempt to repeat the theory. Here are a selection of the resources I have personally found useful when implementing HMMs.

* [Pattern Recognition and Machine Learning](http://research.microsoft.com/en-us/um/people/cmbishop/PRML/index.htm) by Chris Bishop (I largely based my implementation on the discussion in this book)
* [The hidden markov model tutorial](http://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf) by Lawrence Rabiner This is still the definitive text
* Kevin Murphy maintains [an excellent set of links on HMMs](http://www.cs.ubc.ca/~murphyk/Software/HMM/hmm.html) and I would particularly recommend this [PhD Thesis](http://www.cs.ubc.ca/~murphyk/Thesis/thesis.html) which has an excellent summary of learning algorithms in the appendix

The HMM classifiers only work on sequence data, which in Weka is represented as a [relational attribute](http://weka.wikispaces.com/Multi-instance+classification). Each sequence has a single nominal attribute for the class and a relational attribute for the sequence data. Sequences are vary length strings of data items. Each data item can either be a single, nominal attribute or multiple numerical attributes.

If the class is either 0 or 1 and the sequence consists of a string of three elements: (a[1], b[1], c[1]), (a[2], b[2], c[2])…. The data would look like this:

			1, "a1, b1, c1\n a2, b2, c2…"
			
the sequence data is is enclosed in quotes ("), each attribute is separated by a comma and each element of the sequence is separated by a new line (\n)

The Hidden Markov Model classifier accepts the following options:

* States: -S number of HMM states to use
* Iteration Cutoff: -I the proportional minimum change of likelihood at which to stop the EM iteractions
* Covariance Type: -C whether the covariances of gaussian outputs should be full matrices or limited to diagonal or spherical matrices
* Tied Covariance: -D whether the covariances of gaussian outputs are tied to be the same across all outputs
* Left Right: -L whether the state transitions are constrained to go only to the next state in numerical order
* Random Initialisation: -R whether the state transition probabilities are intialized randomly (if this is false they are initialised by performing a k-means clustering on the data)

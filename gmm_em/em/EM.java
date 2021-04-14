package em;

public class EM {
	private static final double EPS = 0.01; // Stopping delta
	
	// Helper class for representing clusters
	public static class Cluster {
		public double mean;
		public double variance;
		public double weight;
		public double m;
		
		public Cluster(double mean, double variance, double weight) {
			this.mean = mean;
			this.variance = variance;
			this.weight = weight;
			this.m = 0;
		}
		
		public String toString() {
  		  return String.format("[mean=%f, variance=%f, weight=%f]", mean, variance, weight);
		}
	}
	
    public static void main(String[] args) {	
    		double[] x = {2.0, 3.0, 9.0, 10.0, 11.0};
    		
    		Cluster[] clusters = trainModel(x, 2);
    		
		System.out.printf("---------- result ----------\n");
  	  	for(int c = 0; c < clusters.length; ++c) {
		  System.out.printf("    Cluster %d: %s\n", c, clusters[c]);
  	  	}
	}

	/**
	 * Compute the normal (Gaussian) distribution.
	 * 
	 * @param x the point to compute the probability for.
	 * @param mean the mean of the distribution.
	 * @param variance the variance ("sigma-squared") of the distribution.
	 * @return N(x; mean; sigma*sigma)
	 */
    private static double normDist(double x, double mean, double variance) {
		return Math.exp(-(x-mean)*(x-mean) / (2*variance)) / Math.sqrt(2 * Math.PI * variance);
    }

    /**
     * Train an EM model.
     * 
     * @param x input points.
     * @param k number of clusters to compute
     * @return the clusters found.
     */
	public static Cluster[] trainModel(double[] x, int k) {
		double[][] r = new double[x.length][k];
		Cluster[] clusters = new Cluster[k];
		
		// Initialize the Clusters
		// TODO
			
		// Iterate
		int iteration = 0;
		double prevL;
		double newL = computeL(x, clusters);
		printDebug(iteration, newL, clusters, r);
		do {
			prevL = newL;
			doEStep(x, clusters, r);
			doMStep(x, clusters, r);
			newL = computeL(x, clusters);
			++iteration;
			printDebug(iteration, newL, clusters, r);
		} while(Math.abs(prevL-newL)>=EPS);
		
		return clusters;
	}
	
	private static double computeL(double[] x, Cluster[] cluster) {
		// TODO
		return 0;
	}
	
	private static void doEStep(double[] x, Cluster[] cluster, double[][] r) {
		// TODO
	}
	
	private static void doMStep(double[] x, Cluster[] cluster, double[][] r) {
		// TODO
	}
	
    private static void printDebug(int iteration, double L, Cluster[] cluster, double[][] r) {
    	  System.out.printf("\nIteration %d, L=%f:\n", iteration, L);
    	  for(int c = 0; c < cluster.length; ++c) {
    		  System.out.printf("    ");
        	  for(int i = 0; i < r.length; ++i) {
    			  System.out.printf("r[%d][%d]=%f ", i, c, r[i][c]);
    		  }
    		  System.out.printf("\n");
    	  }
    	  for(int c = 0; c < cluster.length; ++c) {
    		  System.out.printf("    Cluster %d: %s\n", c, cluster[c]);
    	  }
    }
}

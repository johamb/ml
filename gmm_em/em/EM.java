package em;

import java.util.Arrays;

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
		for (int i = 0; i < clusters.length; i++){
			clusters[i] = new Cluster(i, 1, 0.5);
		}
			
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
	
	private static double computeL(double[] x, Cluster[] clusters) {
		double sumLogs = 0;

		for (double xi: x) {
			double sumDistributions = 0;

			for (Cluster cj: clusters) {
				sumDistributions += computeProbability(xi, cj.mean, cj.variance);
			}

			sumLogs += Math.log(sumDistributions);
		}

		return sumLogs;
	}

	private static double computeProbability(double x, double mean, double variance) {
		double deviation = Math.sqrt(variance);
		return 1 / ((deviation * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * (Math.pow((x - mean) / deviation, 2)));
	}

	private static void doEStep(double[] x, Cluster[] clusters, double[][] r) {
		for (int i = 0; i < x.length; i++) {
			double xi = x[i];
			for (int j = 0; j < clusters.length; j++) {
				Cluster cj = clusters[j];
				double numerator = cj.weight * computeProbability(xi, cj.mean, cj.variance);

				double denominator = 0;
				for (Cluster ci: clusters){
					denominator += ci.weight * computeProbability(xi, ci.mean, ci.variance);
				}

				r[i][j] = numerator / denominator;
			}
		}
	}
	
	private static void doMStep(double[] x, Cluster[] clusters, double[][] r) {

		Cluster[] newClusters = new Cluster[clusters.length];

		// compute responsibility for each cluster
		double[] responsibilities = new double[clusters.length];
		for (int j = 0; j < clusters.length; j++) {
			Cluster cj = clusters[j];

			for (int i = 0; i < r.length; i++) {
				responsibilities[j] += r[i][j];
			}
		}

		double sumResponsibilities = Arrays.stream(responsibilities).sum();

		// update parameters for each cluster
		for (int j = 0; j < clusters.length; j++) {

			// compute new weight
			Cluster cj = clusters[j];
			double responsibility = responsibilities[j];
			cj.weight = responsibility / sumResponsibilities;

			// compute new mean
			double sum1 = 0;
			for (int i = 0; i < r.length; i++) {
				sum1 += r[i][j] * x[i];
			}
			cj.mean = 1 / responsibility * sum1;

			// compute new variance
			double sum2 = 0;
			for (int i = 0; i < r.length; i++) {
				sum2 += r[i][j] * Math.pow(x[i] - cj.mean, 2);
			}
			cj.variance = 1 / responsibility * sum2;

			newClusters[j] = cj;
		}
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

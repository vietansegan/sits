package segmentation;

import core.AbstractSampler;

/**
 *
 * @author vietan
 */
public abstract class AbstractTopicShiftSampler extends AbstractSampler {

    /**
     * Output and input topic-word multinomials
     */
    public abstract void outputPhi(String outputFile) throws Exception;

    public abstract double[][] inputPhi(String inputFile) throws Exception;

    /**
     * Output and input author-shift multinomials
     */
    public abstract void outputPi(String outputFile) throws Exception;

    public abstract double[][] inputPi(String inputFile) throws Exception;

    /**
     * Output log likelihoods over iterations
     */
    public abstract void outputLogLikelihoods(String outputFile) throws Exception;

    /**
     * Output hyperparameters
     */
    public abstract void outputHyperparameters(String outputFile) throws Exception;

    /**
     * Output and input probabilities that a turn is shifted
     */
    public abstract void outputAvgSampledL(String outputFile) throws Exception;

    public abstract double[] inputAvgSampledL(String inputFile) throws Exception;

    /**
     * Get topic-word multinomials
     */
    public abstract double[][] getPhi();

    /**
     * Get author-shift multinomials
     */
    public abstract double[][] getPi();
}

package segmentation.parametric.sampler;

import core.AbstractSampler;
import java.util.ArrayList;
import sampling.likelihood.DirichletMultinomialModel;
import util.IOUtils;
import util.SamplerUtils;

/**
 *
 * @author vietan
 */
public class LDAGibbsSampler extends AbstractSampler {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    protected int K;
    protected int V; // vocabulary size
    protected int D; // number of documents
    protected int[][] words;  // [D] x [Nd]: words
    protected int[][] z;
    protected DirichletMultinomialModel[] doc_topics;
    protected DirichletMultinomialModel[] topic_words;

    public void configure(String folder, int[][] words,
            int V, int K,
            double alpha,
            double beta,
            AbstractSampler.InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag) {
        this.folder = folder;
        this.words = words;

        this.K = K;
        this.V = V;
        this.D = this.words.length;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(beta);

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;

        this.paramOptimized = paramOpt;
        this.prefix = initState.toString();
        this.setName();
    }

    protected void setName() {
        this.name = this.prefix
                + "_LDA"
                + "_K-" + K
                + "_B-" + BURN_IN
                + "_M-" + MAX_ITER
                + "_L-" + LAG
                + "_a-" + formatter.format(this.hyperparams.get(ALPHA))
                + "_b-" + formatter.format(this.hyperparams.get(BETA))
                + "_opt-" + this.paramOptimized;
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }

        initializeHierarchies();

        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }
    }

    protected void initializeHierarchies() {
        if (verbose) {
            logln("--- Initializing topic hierarchy ...");
        }

        doc_topics = new DirichletMultinomialModel[D];
        for (int d = 0; d < D; d++) {
            doc_topics[d] = new DirichletMultinomialModel(K, hyperparams.get(ALPHA) * K, 1.0 / K);
        }

        topic_words = new DirichletMultinomialModel[K];
        for (int k = 0; k < K; k++) {
            topic_words[k] = new DirichletMultinomialModel(V, hyperparams.get(BETA) * V, 1.0 / V);
        }
    }

    protected void initializeAssignments() {
        if (verbose) {
            logln("--- Initializing assignments ...");
        }

        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
            for (int n = 0; n < words[d].length; n++) {
                z[d][n] = rand.nextInt(K);
                doc_topics[d].increment(z[d][n]);
                topic_words[z[d][n]].increment(words[d][n]);
            }
        }
    }

    @Override
    public void sample() {
        openLogger();

        initialize();

        iterate();

        closeLogger();
    }

    @Override
    protected void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        logLikelihoods = new ArrayList<Double>();

        for (iter = 0; iter < MAX_ITER; iter++) {
            for (int d = 0; d < D; d++) {
                for (int t = 0; t < words[d].length; t++) {
                    sampleZ(d, t);
                }
            }

            if (debug) {
                validate("Iter " + iter);
            }

            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);
            if (verbose && iter % (LAG) == 0) {
                if (iter < BURN_IN) {
                    logln("--- Burning in. Iter " + iter + ". llh = " + loglikelihood);
                } else {
                    logln("--- Sampling. Iter " + iter + ". llh = " + loglikelihood);
                }
            }
        }
    }

    private void sampleZ(int d, int n) {
        doc_topics[d].decrement(z[d][n]);
        topic_words[z[d][n]].decrement(words[d][n]);

        double[] logprobs = new double[K];
        for (int k = 0; k < K; k++) {
            logprobs[k] =
                    doc_topics[d].getLogLikelihood(k)
                    + topic_words[k].getLogLikelihood(words[d][n]);
        }

        z[d][n] = SamplerUtils.logScaleSample(logprobs);
        doc_topics[d].increment(z[d][n]);
        topic_words[z[d][n]].increment(words[d][n]);
    }

    @Override
    public double getLogLikelihood() {
        double docTopicLlh = 0;
        for (int d = 0; d < D; d++) {
            docTopicLlh += doc_topics[d].getLogLikelihood();
        }
        double topicWordLlh = 0;
        for (int k = 0; k < K; k++) {
            topicWordLlh += topic_words[k].getLogLikelihood();
        }
        return docTopicLlh + topicWordLlh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        if (newParams.size() != this.hyperparams.size()) {
            throw new RuntimeException("Number of hyperparameters mismatched");
        }
        double llh = 0;
        for (int d = 0; d < D; d++) {
            llh += doc_topics[d].getLogLikelihood(newParams.get(ALPHA) * K, 1.0 / K);
        }
        for (int k = 0; k < K; k++) {
            llh += topic_words[k].getLogLikelihood(newParams.get(BETA) * V, 1.0 / V);
        }
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        this.hyperparams = newParams;
        for (int d = 0; d < D; d++) {
            this.doc_topics[d].setConcentration(this.hyperparams.get(ALPHA) * K);
        }
        for (int k = 0; k < K; k++) {
            this.topic_words[k].setConcentration(this.hyperparams.get(BETA) * V);
        }
    }

    public void outputTopicTopWords(String filepath, ArrayList<String> wordVocab, int numTopWords) throws Exception {
        double[][] distrs = new double[K][];
        for (int k = 0; k < K; k++) {
            distrs[k] = topic_words[k].getDistribution();
        }
        IOUtils.outputTopWords(distrs, wordVocab, numTopWords, filepath);
    }

    public void outputTopicTopWordsCummProbs(String filepath, ArrayList<String> wordVocab, int numTopWords) throws Exception {
        double[][] distrs = new double[K][];
        for (int k = 0; k < K; k++) {
            distrs[k] = topic_words[k].getDistribution();
        }
        IOUtils.outputTopWordsCummProbs(distrs, wordVocab, numTopWords, filepath);
    }

    public void outputTopicWordDistribution(String outputFile) throws Exception {
        double[][] pi = new double[K][];
        for (int k = 0; k < K; k++) {
            pi[k] = this.topic_words[k].getDistribution();
        }
        IOUtils.outputDistributions(pi, outputFile);
    }

    public double[][] inputTopicWordDistribution(String inputFile) throws Exception {
        return IOUtils.inputDistributions(inputFile);
    }

    public void outputDocumentTopicDistribution(String outputFile) throws Exception {
        double[][] theta = new double[D][];
        for (int d = 0; d < D; d++) {
            theta[d] = this.doc_topics[d].getDistribution();
        }
        IOUtils.outputDistributions(theta, outputFile);
    }

    public double[][] inputDocumentTopicDistribution(String inputFile) throws Exception {
        return IOUtils.inputDistributions(inputFile);
    }

    @Override
    public void validate(String msg) {
    }
}

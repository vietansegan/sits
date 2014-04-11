package segmentation.parametric.sampler;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.util.ArrayList;
import segmentation.AbstractTopicShiftSampler;
import util.IOUtils;
import util.SamplerUtils;
import util.sampling.FiniteMultinomial;
import util.sampling.Segment;
import util.sampling.SymmetricFiniteMultinomials;
import util.sampling.TurnVector;

/**
 *
 * @author vietan
 */
public class SeededAuthorShiftSampler extends AbstractTopicShiftSampler {

    private double[][] seedWordPriors;
    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public static final int GAMMA = 2;
    public static final int HAS_SHIFT = 1;
    public static final int NO_SHIFT = 0;
    private int K; // number of topics (input parameter)
    private int J; // number of authors
    private int T; // total number of turns (including separated turns)
    private int V; // vocabulary size
    private int[] authors; // multiple conversations are separated by -1's
    private int[][] words; // multiple conversations are separated by empty arrays
    private int[] l;
    private int[][] z;
    private int[] segment_index; // store the segment for each turn [T x 1]
    private ArrayList<Segment> segments;
    private FiniteMultinomial[] topic_word;
    private SymmetricFiniteMultinomials author_shift;
    private ArrayList<int[]> sampledLs = new ArrayList<int[]>();
    private int iter;

    public void configure(String folder, int[][][] words, int[][] authors,
            int K, int J, int V,
            double alpha, double beta, double gamma) {
        // change format
        int totalTurns = 0;
        for (int i = 0; i < authors.length; i++) {
            totalTurns += authors[i].length + 1;
        }
        int[][] newWords = new int[totalTurns][];
        int[] newSpeakers = new int[totalTurns];
        int index = 0;
        for (int c = 0; c < words.length; c++) {
            for (int t = 0; t < words[c].length; t++) {
                newWords[index] = words[c][t];
                newSpeakers[index] = authors[c][t];
                index++;
            }
            newWords[index] = null;
            newSpeakers[index] = -1;
            index++;
        }
        configure(folder, newWords, newSpeakers, K, J, V, alpha, beta, gamma);
    }

    public void configure(String folder, int[][] words, int[] authors,
            int K, int J, int V,
            double alpha, double beta, double gamma) {
        this.words = words;
        this.authors = authors;

        this.K = K;
        this.J = J;
        this.V = V;
        this.T = words.length;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(beta);
        this.hyperparams.add(gamma);

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(this.cloneHyperparameters());

//        this.hyperparameters = new double[3];
//        this.hyperparameters[ALPHA] = alpha;
//        this.hyperparameters[BETA] = beta;
//        this.hyperparameters[GAMMA] = gamma;
//
//        this.initHyperparams = new double[3];
//        this.initHyperparams[ALPHA] = alpha;
//        this.initHyperparams[BETA] = beta;
//        this.initHyperparams[GAMMA] = gamma;

        this.folder = folder;
    }

    @Override
    public void sample() {
        openLogger();

        initialize();

        iterate();

        closeLogger();
    }

    public void setSeededTopics(double[][] st) {
        this.seedWordPriors = st;
    }

    private void initialize(int[][] initZ, int[] initL) {
        if (debug) {
            logln("Initializing ...");
        }
        segments = new ArrayList<Segment>();
        //topic_word = new Multinomials(K, V, hyperparameters[BETA]);
        topic_word = new FiniteMultinomial[K];
        for (int k = 0; k < K; k++) {
            if (k < seedWordPriors.length) {
                topic_word[k] = new FiniteMultinomial(V, hyperparams.get(BETA) * V, seedWordPriors[k]);
            } else {
                topic_word[k] = new FiniteMultinomial(V, hyperparams.get(BETA) * V, 1.0 / V);
            }
        }
        author_shift = new SymmetricFiniteMultinomials(J, 2, hyperparams.get(GAMMA));

        z = new int[T][];
        l = new int[T];
        segment_index = new int[T];

        for (int t = 0; t < T; t++) {
            if (authors[t] == -1) { // if this is a break between 2 conversation units
                z[t] = new int[0];
                l[t] = -1;
                segment_index[t] = -1;
                continue;
            }

            int Nt = words[t].length; // size of the current turn
            z[t] = new int[Nt];

            // initialize l's
            if (t == 0 || l[t - 1] == -1) // beginning of each conversation
            {
                l[t] = 1;
            } else {
                l[t] = initL[t];
            }
            author_shift.increment(authors[t], l[t]);
            if (l[t] == 1) // create new segment
            {
                segments.add(new Segment(K, hyperparams.get(ALPHA)));
            }

            int cur_segment = segments.size() - 1;
            segment_index[t] = cur_segment;

            // initialize z's
            TurnVector turn = new TurnVector(t, K);
            for (int n = 0; n < Nt; n++) {
                z[t][n] = initZ[t][n];
                topic_word[z[t][n]].increment(words[t][n]);
                turn.increment(z[t][n]);
            }
            segments.get(cur_segment).addTurn(turn);
        }

        if (debug) {
            validate("Initializing");
        }
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        this.hyperparams = newParams;
        for (int k = 0; k < K; k++) {
            this.topic_word[k].setConcentration(this.hyperparams.get(BETA));
        }
        this.author_shift.setHyperparameter(this.hyperparams.get(GAMMA));
        for (Segment segm : segments) {
            segm.setHyperparameter(this.hyperparams.get(ALPHA));
        }
    }

    @Override
    protected void initialize() {
        if (debug) {
            logln("Initializing ...");
        }
        segments = new ArrayList<Segment>();
        topic_word = new FiniteMultinomial[K];
        for (int k = 0; k < K; k++) {
            if (k < seedWordPriors.length) {
                topic_word[k] = new FiniteMultinomial(V, this.hyperparams.get(BETA), seedWordPriors[k]);
            } else {
                topic_word[k] = new FiniteMultinomial(V, this.hyperparams.get(BETA), 1.0 / V);
            }
        }
        author_shift = new SymmetricFiniteMultinomials(J, 2, this.hyperparams.get(ALPHA));

        z = new int[T][];
        l = new int[T];
        segment_index = new int[T];

        for (int t = 0; t < T; t++) {
            if (authors[t] == -1) { // if this is a break between 2 conversation units
                z[t] = new int[0];
                l[t] = -1;
                segment_index[t] = -1;
                continue;
            }

            int Nt = words[t].length; // size of the current turn
            z[t] = new int[Nt];

            // initialize l's
            if (t == 0 || l[t - 1] == -1) // beginning of each conversation
            {
                l[t] = HAS_SHIFT;
            } else {
                l[t] = NO_SHIFT;
            }
            //l[t] = rand.nextInt(2);
            author_shift.increment(authors[t], l[t]);
            if (l[t] == HAS_SHIFT) // create new segment
            {
                segments.add(new Segment(K, this.hyperparams.get(ALPHA)));
            }

            int cur_segment = segments.size() - 1;
            segment_index[t] = cur_segment;

            // initialize z's
            TurnVector turn = new TurnVector(t, K);
            //int[] turnCount = new int[K]; // topic asssignments for the current turn
            for (int n = 0; n < Nt; n++) {
                z[t][n] = rand.nextInt(K);
                topic_word[z[t][n]].increment(words[t][n]);
                //turnCount[z[t][n]] ++;
                turn.increment(z[t][n]);
            }
            segments.get(cur_segment).addTurn(turn);
        }

        if (debug) {
            validate("Initializing");
        }
    }

    @Override
    protected void iterate() {
        if (debug) {
            logln("Iterating ...");
        }

        for (iter = 0; iter < MAX_ITER; iter++) {
            //if (i % (INTERVAL) == 0)
            //    System.out.println("--- iter " + i);

            for (int t = 0; t < T; t++) {
                if (authors[t] == -1) {
                    continue;
                }

                if (t != 0 && authors[t - 1] != -1) // only sample t for non-first turns
                {
                    sampleL(t);
                }

                for (int n = 0; n < words[t].length; n++) {
                    sampleZ(t, n);
                }

                if (debug) {
                    validate("After Z: Error in iter: " + iter + "; turn: " + t);
                }
            }

            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);
            System.out.println("--- Iter " + iter + " --- loglikelihood\t" + loglikelihood);

            // slice sampling for parameters
            if (paramOptimized) {
                sliceSample();
                this.sampledParams.add(cloneHyperparameters());
//                double[] sampledHyperparameter = new double[hyperparams.size()];
//                for(int x=0; x<hyperparams.size(); x++)
//                    sampledHyperparameter[x] = hyperparams.get(x);
//                sampledHyperparameters.add(sampledHyperparameter);
            }

            // debug: print out the loglikelihood
            if (iter % (LAG) == 0) {
                if (iter >= BURN_IN) {
                    int[] sampledL = new int[l.length];
                    for (int x = 0; x < sampledL.length; x++) {
                        sampledL[x] = l[x];
                    }
                    sampledLs.add(sampledL);
                }
            }
        }
    }

    /**
     * Sample the shift indicator at a given turn
     *
     * @param t The turn to be sampled at
     */
    private void sampleL(int t) {
        int cur_author = authors[t];
        int cur_shift = l[t];
        author_shift.decrement(cur_author, cur_shift);

        int cur_segment = segment_index[t];

        Segment preSegment, posSegment, mergedSegment;
        if (cur_shift == HAS_SHIFT) {
            preSegment = segments.get(cur_segment - 1);
            posSegment = segments.get(cur_segment);
            mergedSegment = new Segment(preSegment);
            mergedSegment.mergeSegment(posSegment);
        } else {
            Segment[] splitSegments = segments.get(cur_segment).splitSegment(t);
            preSegment = splitSegments[0];
            posSegment = splitSegments[1];
            mergedSegment = segments.get(cur_segment);
        }

        // probabilities
        double logP0_merge = mergedSegment.getLogLikelihood();
        double logP0_author = Math.log(author_shift.getCount(cur_author, 0) + hyperparams.get(GAMMA))
                - Math.log(author_shift.getCountSum(cur_author) + 2 * hyperparams.get(GAMMA));
        double logP0 = logP0_merge + logP0_author;

        double logP1_pre = preSegment.getLogLikelihood();
        double logP1_pos = posSegment.getLogLikelihood();
        double logP1_author = Math.log(author_shift.getCount(cur_author, 1) + hyperparams.get(GAMMA))
                - Math.log(author_shift.getCountSum(cur_author) + 2 * hyperparams.get(GAMMA));
        double logP1 = logP1_pre + logP1_pos + logP1_author;

        double[] p = new double[2];
        p[0] = Math.exp(logP0 - logP1);
        p[1] = 1;
        int sampled_shift = SamplerUtils.scaleSample(p);

        if (debug) {
            System.out.println("cur_shift = " + cur_shift + " -> sampled_shift = " + sampled_shift);
            System.out.println("logP0 = " + logP0
                    + ". logP0_merge = " + logP0_merge
                    + ". logP0_author = " + logP0_author);
            System.out.println("logP1 = " + logP1
                    + ". logP1_pre = " + logP1_pre
                    + ". logP1_pos = " + logP1_pos
                    + ". logP1_author = " + logP1_author);
        }

        // update after sampling
        author_shift.increment(cur_author, sampled_shift);
        if (cur_shift == NO_SHIFT && sampled_shift == HAS_SHIFT) { // split
            segments.set(cur_segment, preSegment);
            segments.add(cur_segment + 1, posSegment);

            // update indices of all turns from t onwards
            for (int turn = t; turn < l.length; turn++) {
                if (segment_index[turn] != -1) {
                    segment_index[turn]++;
                }
            }
        } else if (cur_shift == HAS_SHIFT && sampled_shift == NO_SHIFT) { // merge
            segments.set(cur_segment - 1, mergedSegment);
            segments.remove(cur_segment);

            // update indices of all turns from t onwards
            for (int turn = t; turn < l.length; turn++) {
                if (segment_index[turn] != -1) {
                    segment_index[turn]--;
                }
            }
        }
        l[t] = sampled_shift;
    }

    /**
     * Sample a topic assignment for a given word
     *
     * @param t The turn containing the word
     * @param n The position of the word in turn t
     */
    private void sampleZ(int t, int n) {
        int cur_topic = z[t][n];
        int cur_word = words[t][n];
        int cur_segment = segment_index[t];

        topic_word[cur_topic].decrement(cur_word);
        segments.get(cur_segment).decrement(t, cur_topic); // update the turn counts

        double[] p = new double[K];
        for (int k = 0; k < K; k++) {
            p[k] = topic_word[k].getLogLikelihood(cur_word)
                    * ((segments.get(cur_segment).getCount(k) + hyperparams.get(ALPHA))
                    / (segments.get(cur_segment).getSum() + K * hyperparams.get(ALPHA)));
        }
        int sampled_topic = SamplerUtils.scaleSample(p);

        topic_word[sampled_topic].increment(cur_word);
        segments.get(cur_segment).increment(t, sampled_topic);

        z[t][n] = sampled_topic;
    }

    @Override
    public void validate(String location) {
        // assert the number of words
        int word_count = 0;
        for (int t = 0; t < T; t++) {
            if (words[t].length != 0) {
                word_count += words[t].length;
            }
        }

        int topic_word_count = 0;
        for (int k = 0; k < K; k++) {
            topic_word_count += topic_word[k].getCountSum();
        }
        if (word_count != topic_word_count) {
            throw new RuntimeException(location + " Error topic_word: " + topic_word_count);
        }

        /*int author_shift_count = author_shift.getTotalCount();
         if(T - this.words.length != author_shift_count)
         throw new RuntimeException(location + " Error author_shift: " + author_shift_count
         + " T: " + T + " # docs: " + words.length);*/

        int segment_total = 0;
        for (Segment s : segments) {
            segment_total += s.getSum();
        }
        if (word_count != segment_total) {
            throw new RuntimeException(location + " Error segment: " + segment_total + "; word_count: " + word_count);
        }

        /*int total_segment_turns = 0;
         for(Segment s : segments)
         total_segment_turns += s.size();
         if(total_segment_turns != MainParametricSampler.getNumActualTurns())
         throw new RuntimeException(location + " Error segment turns: " + total_segment_turns
         + ". correct: " + MainParametricSampler.getNumActualTurns());
         */

        for (Segment s : segments) {
            for (int k = 0; k < K; k++) {
                if (s.getCount(k) < 0) {
                    throw new RuntimeException(location + " Error segment count: " + k + "; " + s.getCount(k));
                }
            }
        }
    }

    public int[][] getTopicAssignments() {
        return z;
    }

    public int[] getShiftIndicators() {
        return l;
    }

    /**
     * Return the topic_word multinomial distribution
     */
    @Override
    public double[][] getPhi() {
        double[][] phi = new double[K][];
        for (int k = 0; k < K; k++) {
            phi[k] = topic_word[k].getDistribution();
        }
        return phi;
    }

    /**
     * Return the author_shift multinomial distribution
     */
    @Override
    public double[][] getPi() {
        return author_shift.getDistribution();
    }

    public double[][] getTheta() {
        double[][] thetas = new double[T][K];
        for (Segment segment : segments) {
            for (TurnVector turn : segment.getTurns().values()) {
                int turnIndex = turn.getTurnIndex();
                for (int k = 0; k < K; k++) {
                    thetas[turnIndex][k] =
                            (turn.getTopicCounts()[k] + hyperparams.get(ALPHA))
                            / (turn.getSum() + K * hyperparams.get(ALPHA));
                }
            }
        }
        return thetas;
    }

    /**
     * Get the log likelihood if a new value of alpha is sampled
     */
    @Override
    public double getLogLikelihood(ArrayList<Double> testHyperparameters) {
        double val = 0.0;
        val += author_shift.getLogLikelihood(testHyperparameters.get(GAMMA));
        for (int k = 0; k < K; k++) {
            val += topic_word[k].getLogLikelihood(hyperparams.get(BETA) * V, 1.0 / V);
        }
        for (Segment s : segments) {
            val += s.getLogLikelihood(testHyperparameters.get(ALPHA));
        }
        return val;
    }

    /**
     * Get the current log likelihood
     */
    @Override
    public double getLogLikelihood() {
        double val = 0.0;
        val += author_shift.getLogLikelihood();
        for (int k = 0; k < K; k++) {
            val += topic_word[k].getLogLikelihood();
        }
        for (Segment s : segments) {
            val += s.getLogLikelihood();
        }

        return val;
    }

    public ArrayList<Double> getLogLikelihoods() {
        return this.logLikelihoods;
    }

    @Override
    public void outputHyperparameters(String outputFile) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        writer.write("Iter\talpha\tbeta\tgamma\n");
        for (int i = 0; i < sampledParams.size(); i++) {
            writer.write(Integer.toString(i));
            for (double h : sampledParams.get(i)) {
                writer.write("\t" + h);
            }
            writer.write("\n");
        }
        writer.close();
    }

    @Override
    public void outputAvgSampledL(String outputFile) throws Exception {
        double[] values = new double[T];
        for (int[] sampledL : sampledLs) {
            for (int t = 0; t < T; t++) {
                values[t] += sampledL[t];
            }
        }
        for (int t = 0; t < T; t++) {
            values[t] /= sampledLs.size();
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (int t = 0; t < T; t++) {
            writer.write(values[t] + "\n");
        }
        writer.close();
    }

    @Override
    public double[] inputAvgSampledL(String inputFile) throws Exception {
        ArrayList<Double> value_list = new ArrayList<Double>();
        BufferedReader reader = IOUtils.getBufferedReader(inputFile);
        String line;
        while ((line = reader.readLine()) != null) {
            value_list.add(Double.parseDouble(line));
        }
        reader.close();

        double[] avgSampledLs = new double[value_list.size()];
        for (int i = 0; i < avgSampledLs.length; i++) {
            avgSampledLs[i] = value_list.get(i);
        }
        return avgSampledLs;
    }

    public void outputTheta(String outputFile) throws Exception {
        IOUtils.outputLatentVariables(this.getTheta(), outputFile);
    }

    public double[][] inputTheta(String inputFile) throws Exception {
        return IOUtils.inputLatentVariables(inputFile);
    }

    @Override
    public void outputPi(String outputFile) throws Exception {
        IOUtils.outputLatentVariables(this.getPi(), outputFile);
    }

    @Override
    public double[][] inputPi(String inputFule) throws Exception {
        return IOUtils.inputLatentVariables(inputFule);
    }

    @Override
    public void outputPhi(String outputFile) throws Exception {
        IOUtils.outputLatentVariables(this.getPhi(), outputFile);
    }

    @Override
    public double[][] inputPhi(String inputFile) throws Exception {
        return IOUtils.inputLatentVariables(inputFile);
    }

    @Override
    public void outputLogLikelihoods(String outputFile) throws Exception {
        IOUtils.outputLogLikelihoods(this.getLogLikelihoods(), outputFile);
    }

    public void outputShiftAssignments(String outputFile) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (int t = 0; t < T; t++) {
            writer.write(l[t] + "\n");
        }
        writer.close();
    }

    public int[] inputShiftAssignments(String inputFile) throws Exception {
        ArrayList<Integer> inflnList = new ArrayList<Integer>();
        BufferedReader reader = IOUtils.getBufferedReader(inputFile);
        String line;
        while ((line = reader.readLine()) != null) {
            inflnList.add(Integer.parseInt(line));
        }
        reader.close();

        int[] infln_asg = new int[inflnList.size()];
        for (int i = 0; i < infln_asg.length; i++) {
            infln_asg[i] = inflnList.get(i);
        }
        return infln_asg;
    }

    public void outputTopicAssignments(String outputFile) throws Exception {
        IOUtils.outputLatentVariableAssignment(this.getTopicAssignments(), outputFile);
    }

    public int[][] inputTopicAssignments(String inputFile) throws Exception {
        return IOUtils.inputLatentVariableAssignment(inputFile);
    }

    @Override
    public String getSamplerName() {
        return this.getPrefix()
                + "_seeded_asm"
                + "_B-" + BURN_IN
                + "_M-" + MAX_ITER
                + "_L-" + LAG
                + "_a-" + formatter.format(hyperparams.get(ALPHA))
                + "_b-" + formatter.format(hyperparams.get(BETA))
                + "_g-" + formatter.format(hyperparams.get(GAMMA))
                + "_K-" + K
                + "_opt-" + paramOptimized;
    }
}

package segmentation.parametric.sampler;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import segmentation.AbstractTopicShiftSampler;
import util.IOUtils;
import util.SamplerUtils;
import util.StatisticsUtils;
import util.sampling.Segment;
import util.sampling.SymmetricFiniteMultinomials;
import util.sampling.TurnVector;

/**
 * Parametric SITS
 *
 * @author vietan
 */
public class AuthorShiftSampler extends AbstractTopicShiftSampler {

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
    private SymmetricFiniteMultinomials topic_word;
    private SymmetricFiniteMultinomials author_shift;
    private ArrayList<int[]> sampledLs = new ArrayList<int[]>();

    @Override
    public String getSamplerName() {
        return this.getPrefix()
                + "_asm"
                + "_B-" + BURN_IN
                + "_M-" + MAX_ITER
                + "_L-" + LAG
                + "_a-" + formatter.format(this.hyperparams.get(ALPHA))
                + "_b-" + formatter.format(this.hyperparams.get(BETA))
                + "_g-" + formatter.format(this.hyperparams.get(GAMMA))
                + "_K-" + K
                + "_opt-" + paramOptimized;
    }

    public void configure(String folder, int[][][] words, int[][] authors,
            int K, int J, int V,
            double alpha, double beta, double gamma,
            int burnin, int maxiter, int samplelag) {
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
        configure(folder, newWords, newSpeakers, K, J, V, alpha, beta, gamma, burnin, maxiter, samplelag);
    }

    public void configure(String folder, int[][] words, int[] authors,
            int K, int J, int V,
            double alpha, double beta, double gamma,
            int burnin, int maxiter, int samplelag) {
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

        this.setSamplerConfiguration(burnin, maxiter, samplelag);

        this.folder = folder;
    }

    @Override
    public void sample() {
        openLogger();

        initialize();

        iterate();

        closeLogger();
    }

    public void sample(int[][] initAssignments) {
        openLogger();

        int[] initL = new int[T];
        for (int t = 0; t < T; t++) {
            if (authors[t] == -1) {
                initL[t] = -1;
            } else if (t == 0 || initL[t - 1] == -1) {
                initL[t] = HAS_SHIFT;
            } else {
                initL[t] = NO_SHIFT;
            }
        }

        initialize(initAssignments, initL);

        iterate();

        closeLogger();
    }

    /**
     * Initialize with some seeded assignments
     */
    private void initialize(int[][] initZ, int[] initL) {
        if (debug) {
            logln("Initializing with assignments ...");
        }
        logLikelihoods = new ArrayList<Double>();

        segments = new ArrayList<Segment>();
        topic_word = new SymmetricFiniteMultinomials(K, V, this.hyperparams.get(BETA));
        author_shift = new SymmetricFiniteMultinomials(J, 2, this.hyperparams.get(GAMMA));

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
                l[t] = initL[t];
            }
            author_shift.increment(authors[t], l[t]);
            if (l[t] == 1) // create new segment
            {
                segments.add(new Segment(K, this.hyperparams.get(ALPHA)));
            }

            int cur_segment = segments.size() - 1;
            segment_index[t] = cur_segment;

            // initialize z's
            TurnVector turn = new TurnVector(t, K);
            for (int n = 0; n < Nt; n++) {
                z[t][n] = initZ[t][n];
                topic_word.increment(z[t][n], words[t][n]);
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
        this.topic_word.setHyperparameter(this.hyperparams.get(BETA));
        this.author_shift.setHyperparameter(this.hyperparams.get(GAMMA));
        for (Segment segm : segments) {
            segm.setHyperparameter(this.hyperparams.get(ALPHA));
        }
    }

    /**
     * Initialize with random assignments
     */
    @Override
    protected void initialize() {
        if (debug) {
            logln("Initializing ...");
        }
        logLikelihoods = new ArrayList<Double>();

        segments = new ArrayList<Segment>();
        topic_word = new SymmetricFiniteMultinomials(K, V, this.hyperparams.get(BETA));
        author_shift = new SymmetricFiniteMultinomials(J, 2, this.hyperparams.get(GAMMA));

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
                topic_word.increment(z[t][n], words[t][n]);
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
            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);

            if (verbose) {
                if (iter % LAG == 0) {
                    if (iter < BURN_IN) {
                        logln("--- Burning in. Iter " + iter
                                + "\t llh = " + loglikelihood);
                    } else {
                        logln("--- Sampling. Iter " + iter
                                + "\t llh = " + loglikelihood);
                    }
                }
            }

            for (int t = 0; t < T; t++) {
                if (authors[t] == -1) {
                    continue;
                }

                if (t != 0 && authors[t - 1] != -1) { // only sample t for non-first turns
                    if (words[t].length > 5) {
                        sampleL(t);
                    } else {
                        l[t] = 0;
                    }
                }
                for (int n = 0; n < words[t].length; n++) {
                    sampleZ(t, n);
                }

                if (debug) {
                    validate("After Z: Error in iter: " + iter + "; turn: " + t);
                }
            }

            if (iter >= BURN_IN) {
                if (iter % LAG == 0) {
                    // record sampled topic shift indicators
                    int[] sampledL = new int[l.length];

//                    System.out.println("iter = " + iter + ". sampledL size = " + sampledL.length);

                    for (int x = 0; x < sampledL.length; x++) {
                        sampledL[x] = l[x];
                    }
                    sampledLs.add(sampledL);

                    // slice sampling for parameters
                    if (paramOptimized) {
                        if (verbose) {
                            logln("--- --- Slice sampling ...");
                        }

                        sliceSample();
                        this.sampledParams.add(this.cloneHyperparameters());
                    }
                }
            }

            if (debug) {
                validate("Error in iter: " + iter);
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
        double logP0_author = Math.log(author_shift.getCount(cur_author, 0) + this.hyperparams.get(GAMMA))
                - Math.log(author_shift.getCountSum(cur_author) + 2 * this.hyperparams.get(GAMMA));
        double logP0 = logP0_merge + logP0_author;

        double logP1_pre = preSegment.getLogLikelihood();
        double logP1_pos = posSegment.getLogLikelihood();
        double logP1_author = Math.log(author_shift.getCount(cur_author, 1) + this.hyperparams.get(GAMMA))
                - Math.log(author_shift.getCountSum(cur_author) + 2 * this.hyperparams.get(GAMMA));
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

        topic_word.decrement(cur_topic, cur_word);
        segments.get(cur_segment).decrement(t, cur_topic); // update the turn counts

        double[] p = new double[K];
        for (int k = 0; k < K; k++) {
            p[k] = ((topic_word.getCount(k, cur_word) + this.hyperparams.get(BETA)) / (topic_word.getCountSum(k) + V * this.hyperparams.get(BETA)))
                    * ((segments.get(cur_segment).getCount(k) + this.hyperparams.get(ALPHA)) / (segments.get(cur_segment).getSum() + K * this.hyperparams.get(ALPHA)));
        }
        int sampled_topic = SamplerUtils.scaleSample(p);

        topic_word.increment(sampled_topic, cur_word);
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

        int topic_word_count = topic_word.getTotalCount();
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
        return topic_word.getDistribution();
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
                            (turn.getTopicCounts()[k] + this.hyperparams.get(ALPHA))
                            / (turn.getSum() + K * this.hyperparams.get(ALPHA));
                }
            }
        }
        return thetas;
    }

    /**
     * Get the current log likelihood
     */
    @Override
    public double getLogLikelihood() {
        double val = 0.0;
        val += author_shift.getLogLikelihood();
        val += topic_word.getLogLikelihood();
        for (Segment s : segments) {
            val += s.getLogLikelihood();
        }

        return val;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        if (newParams.size() != this.hyperparams.size()) {
            throw new RuntimeException("Number of hyperparameters mismatched");
        }

        double val = 0.0;
        val += author_shift.getLogLikelihood(newParams.get(GAMMA));
        val += topic_word.getLogLikelihood(newParams.get(BETA));
        for (Segment s : segments) {
            val += s.getLogLikelihood(newParams.get(ALPHA));
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
        for (int i = 0; i < this.sampledParams.size(); i++) {
            writer.write(Integer.toString(i));
            for (double h : this.sampledParams.get(i)) {
                writer.write("\t" + h);
            }
            writer.write("\n");
        }
        writer.close();
    }

    @Override
    public void outputAvgSampledL(String outputFile) throws Exception {
        double[] values = new double[T];

        //debug
//        System.out.println("In AuthorShiftSampler: T = " + T);

        for (int[] sampledL : sampledLs) {
//            System.out.println("sampledL size: " + sampledL.length);

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

    public void outputAllShiftAssignments(String outputFile) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (int j = 0; j < sampledLs.get(0).length; j++) {
            for (int i = 0; i < sampledLs.size(); i++) {
                writer.write(sampledLs.get(i)[j] + " ");
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void outputTopicAssignments(String outputFile) throws Exception {
        IOUtils.outputLatentVariableAssignment(this.getTopicAssignments(), outputFile);
    }

    public int[][] inputTopicAssignments(String inputFile) throws Exception {
        return IOUtils.inputLatentVariableAssignment(inputFile);
    }

    public HashMap<String, Double> inputSpeakerTopicShiftScore(String inputFile,
            ArrayList<String> speakerVocab) throws Exception {
        HashMap<String, Double> speakerTopicShiftScore = new HashMap<String, Double>();
        BufferedReader reader = IOUtils.getBufferedReader(inputFile);
        String line;
        int count = 0;
        while ((line = reader.readLine()) != null) {
            String[] sline = line.split(" ");
            double score = Double.parseDouble(sline[1]);
            speakerTopicShiftScore.put(speakerVocab.get(count), score);
            count++;
        }
        reader.close();
        return speakerTopicShiftScore;
    }

    public HashMap<String, Double> outputSpeakerTopicShiftScore(String outputFile,
            double[][] pi, ArrayList<String> speakerVocab) throws Exception {
        HashMap<String, Double> speakerPiTable = new HashMap<String, Double>();

        // turn counts
        int[] author_turncount = new int[speakerVocab.size()];
        for (int i = 0; i < authors.length; i++) {
            if (authors[i] == -1) {
                continue;
            }
            author_turncount[authors[i]]++;
        }

        // conversation counts
        int[] author_convcount = new int[speakerVocab.size()];
        Set<Integer> convSpeakerSet = new HashSet<Integer>();
        for (int i = 0; i < authors.length; i++) {
            if (authors[i] == -1) {
                for (int s : convSpeakerSet) {
                    author_convcount[s]++;
                }
                convSpeakerSet = new HashSet<Integer>();
            } else {
                convSpeakerSet.add(authors[i]);
            }
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        // Header
        writer.write("Speaker\tPi_0\tPi_1\tTurnCount\tConversationCount\n");

        for (int i = 0; i < speakerVocab.size(); i++) {
//            if(author_turncount[i] < 30 || author_convcount[i] < 5)
//                continue;

            String au = speakerVocab.get(i);
            if (au.equals("NULL")) {
                continue;
            }
            writer.write(au);
            for (int j = 0; j < pi[i].length; j++) {
                writer.write("\t" + pi[i][j]);
            }
            writer.write("\t" + author_turncount[i]);
            writer.write("\t" + author_convcount[i]);
            writer.write("\n");

            speakerPiTable.put(au, pi[i][1]);
        }
        writer.close();

        return speakerPiTable;
    }

    /**
     * Return the average topic entropy of each speaker
     */
    public HashMap<String, Double> getSpeakerTopicEntropy(String thetaFilepath,
            ArrayList<String> speakerVocab) throws Exception {
        double[][] thetas = inputTheta(thetaFilepath);
        HashMap<String, ArrayList<double[]>> speakerTopicDistrs = new HashMap<String, ArrayList<double[]>>();
        for (int i = 0; i < this.authors.length; i++) {
            if (this.authors[i] == -1) {
                continue;
            }
            String speaker = speakerVocab.get(this.authors[i]);
            ArrayList<double[]> singleSpeakerTopicDistrs = speakerTopicDistrs.get(speaker);
            if (singleSpeakerTopicDistrs == null) {
                singleSpeakerTopicDistrs = new ArrayList<double[]>();
            }
            singleSpeakerTopicDistrs.add(thetas[i]);
            speakerTopicDistrs.put(speaker, singleSpeakerTopicDistrs);
        }

        HashMap<String, Double> speakerTopicEntropy = new HashMap<String, Double>();
        for (String speaker : speakerTopicDistrs.keySet()) {
            ArrayList<Double> entropies = new ArrayList<Double>();
            for (double[] topic : speakerTopicDistrs.get(speaker)) {
                double entropy = StatisticsUtils.entropy(topic);
                entropies.add(entropy);
            }
            double avgEntropy = StatisticsUtils.mean(entropies);
            speakerTopicEntropy.put(speaker, avgEntropy);
        }

        return speakerTopicEntropy;
    }

    public HashMap<String, Double> getSpeakerTopicShiftIndicator(String avgSampledLFilepath,
            ArrayList<String> speakerVocab) throws Exception {
        double[] avgShiftInd = this.inputAvgSampledL(avgSampledLFilepath);

        HashMap<String, ArrayList<Double>> speakerTopicShiftIndList = new HashMap<String, ArrayList<Double>>();
        for (int i = 0; i < this.authors.length; i++) {
            if (this.authors[i] == -1) {
                continue;
            }
            String speaker = speakerVocab.get(this.authors[i]);
            ArrayList<Double> singleSpeakerIndList = speakerTopicShiftIndList.get(speaker);
            if (singleSpeakerIndList == null) {
                singleSpeakerIndList = new ArrayList<Double>();
            }
            singleSpeakerIndList.add(avgShiftInd[i]);
            speakerTopicShiftIndList.put(speaker, singleSpeakerIndList);
        }

        HashMap<String, Double> speakerTopicShiftInd = new HashMap<String, Double>();
        for (String speaker : speakerTopicShiftIndList.keySet()) {
            double avgInd = StatisticsUtils.mean(speakerTopicShiftIndList.get(speaker));
            speakerTopicShiftInd.put(speaker, avgInd);
        }
        return speakerTopicShiftInd;
    }

    public HashMap<String, HashMap<String, ArrayList<Double>>> getConversationSpeakerTopicShiftIndicators(
            String avgSampledLFilepath,
            ArrayList<String> speakerVocab,
            ArrayList<String> convIds)
            throws Exception {
        double[] avgShiftInd = this.inputAvgSampledL(avgSampledLFilepath);

        HashMap<String, HashMap<String, ArrayList<Double>>> convSpeakerTSInds =
                new HashMap<String, HashMap<String, ArrayList<Double>>>();

        String curConvId = convIds.get(0); // first conversation's id
        int nextConvIndex = 1;
        for (int i = 0; i < this.authors.length; i++) {
            if (this.authors[i] == -1) {
                if (nextConvIndex == convIds.size()) { // last conversation 
                    break;
                }
                curConvId = convIds.get(nextConvIndex);
                nextConvIndex++;
                continue;
            }

            String speaker = speakerVocab.get(this.authors[i]);

            HashMap<String, ArrayList<Double>> singleConvMap = convSpeakerTSInds.get(curConvId);
            if (singleConvMap == null) {
                singleConvMap = new HashMap<String, ArrayList<Double>>();
            }
            ArrayList<Double> indList = singleConvMap.get(speaker);
            if (indList == null) {
                indList = new ArrayList<Double>();
            }
            indList.add(avgShiftInd[i]);
            singleConvMap.put(speaker, indList);
            convSpeakerTSInds.put(curConvId, singleConvMap);
        }
        return convSpeakerTSInds;
    }
}

package segmentation.nonparametric.sampler;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import segmentation.AbstractTopicShiftSampler;
import segmentation.TopicSegmentation;
import segmentation.TopicSegmentation.InternalAssumption;
import util.IOUtils;
import util.SamplerUtils;
import util.sampling.ResizableMultinomials;
import util.sampling.SymmetricFiniteMultinomials;

/**
 * Nonparametric SITS
 *
 * @author vietan
 */
public class NonparametricAuthorShiftPathAssumptionSampler extends AbstractTopicShiftSampler {

    public static final int HAS_SHIFT = 1;
    public static final int NO_SHIFT = 0;
    // indices for hyperparameters    
    public static final int ALPHA = 0;  // for top level DP
    public static final int ALPHA_0 = 1;// for across conversation DP
    public static final int ALPHA_C = 2;// for conversation-specific (across turn) DP
    public static final int LAMBDA = 3; // for topic_word multinomial
    public static final int GAMMA = 4;  // for author_shift multinomial
    protected int M; // number of authors
    protected int V; // vocabulary size
    protected int C; // number of conversations
    protected int K = 100; // initial number of topics
    // observed variables
    protected int[][] authors; // [C x T_c]
    protected int[][][] words; // [C x Tc x Nct]
    // latent vairables
    protected int[][] l; // [C x T_c] l_ct: shift indicator
    protected int[][] s; // [C x T_c] s_ct: segment indices for each turn
    protected int[][][] z; // [C x T_c x N_tc] z_ctn: global indices
    protected ResizableMultinomials topic_word; // topic_word multinomial
    protected SymmetricFiniteMultinomials author_shift; // author_shift multinomial
    protected Hierarchy hier_prior; // hierarchical prior of topic
    protected Restaurant[][] turnPseudoRests;
    //protected double[] topic_hier_hyperparams;
    //protected ArrayList<Double> logLikelihoods;
    protected ArrayList<int[][]> sampledLs;
    //protected ArrayList<double[]> sampledHyperparameters = new ArrayList<double[]>();
    protected InternalAssumption pathAssumption = InternalAssumption.MINIMAL;

    public void setPathAssumption(InternalAssumption assumption) {
        this.pathAssumption = assumption;
    }

    public InternalAssumption getPathAssumption() {
        return this.pathAssumption;
    }

    public void setK(int K) {
        this.K = K;
    }

    @Override
    public String getSamplerName() {
        String initStr = this.getPrefix();
        if (this.prefix.equals(TopicSegmentation.RANDOM_INIT)) {
            initStr += "_K-" + this.K;
        }
        return initStr
                + "_np-asm"
                + "_B-" + BURN_IN
                + "_M-" + MAX_ITER
                + "_L-" + LAG
                + "_l-" + formatter.format(this.hyperparams.get(LAMBDA))
                + "_g-" + formatter.format(this.hyperparams.get(GAMMA))
                + "_a-" + formatter.format(this.hyperparams.get(ALPHA))
                + "_a0-" + formatter.format(this.hyperparams.get(ALPHA_0))
                + "_ac-" + formatter.format(this.hyperparams.get(ALPHA_C))
                + "_path-" + this.pathAssumption
                + "_opt-" + this.paramOptimized;
    }

    public void configure(String folder, int[][][] words, int[][] authors,
            int M, int V,
            double lambda, double gamma,
            double alpha, double alpha_0, double alpha_c,
            InternalAssumption assumption) {
        this.debug = false;

        if (debug) {
            logln("\nConfiguring ...");
        }

        this.words = words;
        this.authors = authors;

        this.M = M;
        this.V = V;
        this.C = words.length;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(alpha_0);
        this.hyperparams.add(alpha_c);
        this.hyperparams.add(lambda);
        this.hyperparams.add(gamma);

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.pathAssumption = assumption;
        this.folder = folder;
    }

    public void configure(String folder, int[][] words, int[] authors,
            int M, int V,
            double lambda, double gamma,
            double alpha, double alpha_0, double alpha_c,
            InternalAssumption assumption) {

        ArrayList<ArrayList<Integer>> temp_authors = new ArrayList<ArrayList<Integer>>();
        ArrayList<ArrayList<int[]>> temp_words = new ArrayList<ArrayList<int[]>>();
        int cur_author = 0;
        ArrayList<Integer> cur_conv_authors = new ArrayList<Integer>();
        ArrayList<int[]> cur_conv_words = new ArrayList<int[]>();
        for (int i = 0; i < authors.length;) {
            while (cur_author != -1) {
                cur_conv_authors.add(authors[i]);
                cur_conv_words.add(words[i]);

                i++;
                cur_author = authors[i];
            }

            temp_authors.add(cur_conv_authors);
            temp_words.add(cur_conv_words);
            cur_conv_authors = new ArrayList<Integer>();
            cur_conv_words = new ArrayList<int[]>();
            i++;
            if (i == authors.length) {
                break;
            }
            cur_author = authors[i];
        }

        int[][][] ws = new int[temp_words.size()][][];
        int[][] as = new int[temp_authors.size()][];
        for (int i = 0; i < as.length; i++) {
            int[][] single_conv_ws = new int[temp_words.get(i).size()][];
            int[] single_conv_as = new int[temp_authors.get(i).size()];
            for (int j = 0; j < single_conv_as.length; j++) {
                single_conv_as[j] = temp_authors.get(i).get(j);
                single_conv_ws[j] = temp_words.get(i).get(j);
            }
            as[i] = single_conv_as;
            ws[i] = single_conv_ws;
        }
        configure(folder, ws, as, M, V, lambda, gamma, alpha, alpha_0, alpha_c, assumption);
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        this.hyperparams = newParams;
        this.author_shift.setHyperparameter(this.hyperparams.get(GAMMA));
        this.topic_word.setHyperparameter(this.hyperparams.get(LAMBDA));
        this.hier_prior.setHyperparameters(this.getHierParams());
    }

    private double[] getHierParams() {
        double[] topic_hier_hyperparams = new double[3];
        topic_hier_hyperparams[0] = this.hyperparams.get(ALPHA);
        topic_hier_hyperparams[1] = this.hyperparams.get(ALPHA_0);
        topic_hier_hyperparams[2] = this.hyperparams.get(ALPHA_C);
        return topic_hier_hyperparams;
    }
    private int[][] initLs = null;

    public void setInitLs(int[][] il) {
        this.initLs = il;
    }

    @Override
    protected void initialize() {
        logln("Initializing ...");

        l = new int[C][];
        s = new int[C][];
        z = new int[C][][];
        author_shift = new SymmetricFiniteMultinomials(M, 2, this.hyperparams.get(GAMMA));
        topic_word = new ResizableMultinomials(V, this.hyperparams.get(LAMBDA));
        turnPseudoRests = new Restaurant[C][];

        // initialize l's
        for (int c = 0; c < C; c++) {
            int segm_count = 0;
            l[c] = new int[words[c].length];
            s[c] = new int[words[c].length];
            for (int t = 0; t < words[c].length; t++) {
                // initialize l's
                if (t == 0) {
                    l[c][t] = HAS_SHIFT;
                } else {
                    if (initLs == null) {
                        l[c][t] = NO_SHIFT;
                    } else {
                        l[c][t] = initLs[c][t];
                    }
                }
                //l[c][t] = rand.nextInt(2); // uniformly random
                author_shift.increment(authors[c][t], l[c][t]);

                if (l[c][t] == HAS_SHIFT) {
                    segm_count++;
                }
                s[c][t] = segm_count - 1;
            }
        }

        // initialize the hierachy
        double[] topic_hier_hyperparams = getHierParams();
        hier_prior = new Hierarchy(topic_hier_hyperparams, pathAssumption);
        for (int c = 0; c < C; c++) {
            turnPseudoRests[c] = new Restaurant[words[c].length];
            for (int t = 0; t < words[c].length; t++) {
                turnPseudoRests[c][t] = new Restaurant(t, null, -1, false, null);
            }
        }
        initializeHierarchy();

        // initialize z's
        for (int c = 0; c < C; c++) {
            z[c] = new int[words[c].length][];
            for (int t = 0; t < words[c].length; t++) {
                z[c][t] = new int[words[c][t].length];
                for (int n = 0; n < words[c][t].length; n++) {
                    if (c == 0 && t == 0 && n == 0) { // add the first word
                        z[c][t][n] = 0;
                        Table wordPseudoTable = turnPseudoRests[c][t].getNewTable();
                        wordPseudoTable.setGlobalIndex(z[c][t][n]);
                        hier_prior.addCustomer(
                                this.turnPseudoRests[c][t].getParent(), // leave of the path
                                wordPseudoTable); // word pseudo table with updated global index
                        topic_word.increment(z[c][t][n], words[c][t][n]);
                    } else if (this.prefix.equals(TopicSegmentation.RANDOM_INIT)) {
                        initializeZRandom(c, t, n);
                    } else if (this.prefix.equals(TopicSegmentation.PRIOR_INIT)) {
                        initializeZPrior(c, t, n);
                    } else if (this.prefix.equals(TopicSegmentation.MAXIMAL_INIT)) {
                        initializeZMaximal(c, t, n);
                    } else {
                        throw new RuntimeException("Unknown initialization for Z's");
                    }
                }
            }
        }

        double loglikelihood = this.getLogLikelihood();
        logln("--- Iter " + iter + ". " + hier_prior.getState() + " llh = " + loglikelihood);

        if (debug) {
            validate("Initializing");
        }
    }

    /**
     * Create the hierarchy
     */
    private void initializeHierarchy() {
        // construct the tree structure
        hier_prior.root = new Restaurant(0, null,
                hier_prior.hyperparameters[0], false,
                hier_prior.pathAssumption);
        for (int c = 0; c < l.length; c++) {
            Restaurant conv_rest = new Restaurant(c, hier_prior.root,
                    hier_prior.hyperparameters[1], false,
                    hier_prior.pathAssumption);
            hier_prior.root.addChild(conv_rest);

            MergeSplitRestaurant cur_segm_rest = null;
            for (int t = 0; t < l[c].length; t++) {
                if (l[c][t] == NonparametricAuthorShiftPathAssumptionSampler.HAS_SHIFT) {
                    cur_segm_rest = new MergeSplitRestaurant(s[c][t], conv_rest,
                            hier_prior.hyperparameters[2], true,
                            hier_prior.pathAssumption);
                    conv_rest.addChild(cur_segm_rest);
                }

                // create pseudo restaurant
                turnPseudoRests[c][t].setParent(cur_segm_rest);
                cur_segm_rest.addTurnRestaurant(t, turnPseudoRests[c][t]);
            }
        }
    }

    /**
     * Sample the topic assignment from prior
     */
    private void initializeZPrior(int c, int t, int n) {
        if (debug) {
            logln("Initializing Z: c=" + c + "; t=" + t + "; n=" + n);
        }

        ArrayList<Integer> table_indices = new ArrayList<Integer>();
        ArrayList<Double> table_probs = new ArrayList<Double>();

        // prob assigning new table
        table_indices.add(topic_word.getNumDistributions());
        table_probs.add(1.0 / topic_word.getDimension()
                * hier_prior.getScore(
                this.turnPseudoRests[c][t].getParent(), // the leave of the path
                Table.UNASSIGNED_GLOBAL_INDEX,
                1.0 / hier_prior.getNumTopics()));

        // prob assigning existing topics
        for (int globalIndex : topic_word.getDistributionIndices()) {
            double prob = topic_word.getLikelihood(globalIndex, words[c][t][n])
                    * hier_prior.getScore(
                    this.turnPseudoRests[c][t].getParent(), // the leave of the path
                    globalIndex,
                    1.0 / hier_prior.getNumTopics());

            table_indices.add(globalIndex);
            table_probs.add(prob);
        }

        int array_index = SamplerUtils.scaleSample(table_probs); // should use logScaleSample
        int sampled_z = table_indices.get(array_index);

        // add word
        Table wordPseudoTable = turnPseudoRests[c][t].getNewTable();
        wordPseudoTable.setGlobalIndex(sampled_z);
        hier_prior.addCustomer(
                this.turnPseudoRests[c][t].getParent(), // leave of the path
                wordPseudoTable); // word pseudo table with updated global index
        topic_word.increment(sampled_z, words[c][t][n]);
        z[c][t][n] = sampled_z;

        if (debug) {
            validate("Initializing Z: c=" + c + "; t=" + t + "; n=" + n);
        }
    }

    /**
     * Randomly choose topic assignments
     */
    private void initializeZRandom(int c, int t, int n) {
        z[c][t][n] = rand.nextInt(K);
        topic_word.increment(z[c][t][n], words[c][t][n]);

        // create pseudo table
        Table wordPseudoTable = turnPseudoRests[c][t].getNewTable();
        wordPseudoTable.setGlobalIndex(z[c][t][n]);
        turnPseudoRests[c][t].addCustomerToTable(wordPseudoTable);

        // add to the hierarchy
        hier_prior.addCustomer(turnPseudoRests[c][t].getParent(), wordPseudoTable);
    }

    /**
     * 
     */
    private void initializeZMaximal(int c, int t, int n) {
    }

    @Override
    protected void iterate() {
        logln("Iterating ...");

        logLikelihoods = new ArrayList<Double>();
        sampledLs = new ArrayList<int[][]>();

        for (iter = 0; iter < MAX_ITER; iter++) {
            if (debug && iter % (LAG) == 0) {
                if (iter < BURN_IN) {
                    logln("--- Burning in. ");
                } else {
                    logln("--- Sampling. ");
                }
            }

            for (int c = 0; c < C; c++) {
                for (int t = 0; t < words[c].length; t++) {
                    if (t != 0) // sample l only from the 2nd turn onwards
                    {
                        sampleL(c, t);
                    }

                    for (int n = 0; n < words[c][t].length; n++) {
                        sampleZ(c, t, n);
                    }
                }
            }

            if (debug) {
                validate("Iter " + iter);
            }

            double loglikelihood = this.getLogLikelihood();
            logLikelihoods.add(loglikelihood);
            logln("--- Iter " + iter + ". " + hier_prior.getState() + " llh = " + loglikelihood);

            if (iter % (LAG) == 0) {
                if (iter >= BURN_IN) {
                    int[][] sampledL = new int[l.length][];
                    for (int x = 0; x < sampledL.length; x++) {
                        sampledL[x] = new int[l[x].length];
                        for (int y = 0; y < sampledL[x].length; y++) {
                            sampledL[x][y] = l[x][y];
                        }
                    }
                    sampledLs.add(sampledL);

                    if (paramOptimized) { // slice sampling for parameters
                        sliceSample();
                        this.sampledParams.add(cloneHyperparameters());
                    }
                }
            }
        }
    }

    private void sampleL(int c, int t) {
        if (debug) {
            logln("Iter " + iter + ". Sampling l: c=" + c + "; t=" + t);
        }

        // only work if initializing all with 0 (except for the 1st turn)
        if (words[c][t].length < 5) {
            return;
        }

        int cur_author = authors[c][t];
        int cur_shift = l[c][t];

        // decrement
        author_shift.decrement(cur_author, cur_shift);

        // sample l
        int sampled_shift;
        if (cur_shift == HAS_SHIFT) {
            // since the current turn is the beginning of a segment, the previous
            // turn belongs to the previous segment
            MergeSplitRestaurant preRest = (MergeSplitRestaurant) this.turnPseudoRests[c][t - 1].getParent();
            MergeSplitRestaurant curRest = (MergeSplitRestaurant) this.turnPseudoRests[c][t].getParent();

            // merge proposal
            MergeRestaurantProposal mergeProposal = new MergeRestaurantProposal(preRest, curRest);
            mergeProposal.propose();

            // sample
            // compute probabilities and sample
            double logP0_mergedRest = mergeProposal.getMergedRestaurantLogPrior();
            double logP0_author = author_shift.getLogSamplingProbability(cur_author, NO_SHIFT);
            double logP0_newParent = mergeProposal.getNewParentLogPrior();
            double logP0 = logP0_mergedRest + logP0_author + logP0_newParent;

            double logP1_preRest = preRest.getLogPrior();
            double logP1_posRest = curRest.getLogPrior();
            double logP1_author = author_shift.getLogSamplingProbability(cur_author, HAS_SHIFT);
            double logP1_curParent = mergeProposal.getCurParentLogPrior();
            double logP1 = logP1_preRest + logP1_posRest + logP1_author + logP1_curParent;

            double[] p = new double[2];
            p[0] = Math.exp(logP0 - logP1);
            p[1] = 1;

            if (Double.isInfinite(p[0]) && p[0] > 0) {
                sampled_shift = 0;
            } else {
                sampled_shift = SamplerUtils.scaleSample(p);
            }

            if (debug) {
                logln("cur_shift: " + cur_shift + " -> sampled_shift: " + sampled_shift);
                logln("pre rest: " + mergeProposal.preRest.toString());
                logln("pos rest: " + mergeProposal.posRest.toString());
                logln("mrg rest: " + mergeProposal.mergedRest.toString());

                logln("logP0 = " + logP0
                        + ". logP0_mergedRest = " + logP0_mergedRest
                        + ". logP0_newParentRest = " + logP0_newParent
                        + ". logP0_author = " + logP0_author);
                logln("logP1 = " + logP1
                        + ". logP1_preRest = " + logP1_preRest
                        + ". logP1_posRest = " + logP1_posRest
                        + ". logP1_oriConvRest = " + logP1_curParent
                        + ". logP1_author = " + logP1_author);
            }

            if (sampled_shift == NO_SHIFT) { // if merged, update the hierarchy
                merge(preRest, curRest);

                // update segm restaurant indices
                for (int turn_idx = t; turn_idx < l[c].length; turn_idx++) {
                    s[c][turn_idx]--;
                }
            }
        } else {
            assert cur_shift == NO_SHIFT;

            // current restaurant
            MergeSplitRestaurant curRest = (MergeSplitRestaurant) turnPseudoRests[c][t].getParent();

            // split proposal
            SplitRestaurantProposal splitProposal = new SplitRestaurantProposal(curRest, t);
            splitProposal.split();

            // compute probabilities and sample
            double logP0_curRest = curRest.getLogPrior();
            double logP0_author = author_shift.getLogSamplingProbability(cur_author, NO_SHIFT);
            double logP0_curParent = splitProposal.getCurParentLogPrior();
            double logP0 = logP0_curRest + logP0_author + logP0_curParent;

            double logP1_preRest = splitProposal.getPreSplitRestaurantLogPrior();
            double logP1_posRest = splitProposal.getPosSplitRestaurantLogPrior();
            double logP1_author = author_shift.getLogSamplingProbability(cur_author, HAS_SHIFT);
            double logP1_newParent = splitProposal.getNewParentLogPrior();
            double logP1 = logP1_preRest + logP1_posRest + logP1_author + logP1_newParent;

            double[] p = new double[2];
            p[0] = Math.exp(logP0 - logP1);
            p[1] = 1;

            if (Double.isInfinite(p[0]) && p[0] > 0) {
                sampled_shift = 0;
            } else {
                sampled_shift = SamplerUtils.scaleSample(p);
            }

            if (debug) {
                logln("cur_shift: " + cur_shift + " -> sampled_shift: " + sampled_shift);
                logln("ori rest: " + splitProposal.oriRest.toString());
                logln("pre rest: " + splitProposal.splitRests[0].toString());
                logln("pos rest: " + splitProposal.splitRests[1].toString());

                logln("logP0 = " + logP0
                        + ". logP0_curRest = " + logP0_curRest
                        + ". logP0_curConvRest = " + logP0_curParent
                        + ". logP0_author = " + logP0_author);
                logln("logP1 = " + logP1
                        + ". logP1_preRest = " + logP1_preRest
                        + ". logP1_posRest = " + logP1_posRest
                        + ". logP1_newConvRest = " + logP1_newParent
                        + ". logP1_author = " + logP1_author);
            }

            if (sampled_shift == HAS_SHIFT) { // if split, update the hierarchy
                split(curRest, t);

                // update indices
                for (int turn_idx = t; turn_idx < l[c].length; turn_idx++) {
                    s[c][turn_idx]++;
                }
            }
        }

        // update after sampling
        author_shift.increment(cur_author, sampled_shift);
        l[c][t] = sampled_shift; // reassign sampled value of l

        // update indices of all segment restaurants
        for (int turn_idx = 0; turn_idx < l[c].length; turn_idx++) {
            if (l[c][turn_idx] == HAS_SHIFT) {
                turnPseudoRests[c][turn_idx].getParent().setIndex(s[c][turn_idx]);
            }
        }

        if (debug) {
            validate("sampling l_" + c + "," + t);
        }
    }

    private void merge(MergeSplitRestaurant preRest, MergeSplitRestaurant posRest) {
        Restaurant parent = preRest.getParent();
        if (!parent.equals(posRest.getParent())) {
            throw new RuntimeException("Uncommon parent restaurant when merging");
        }

        MergeSplitRestaurant mergeRest = new MergeSplitRestaurant(
                Restaurant.UNASSIGNED_REST_INDEX,
                parent,
                preRest.getHyperparameter(),
                preRest.isTerminal(),
                preRest.getPathAssumption());

        parent.addChild(mergeRest);

        MergeSplitRestaurant[] oldRests = new MergeSplitRestaurant[2];
        oldRests[0] = preRest;
        oldRests[1] = posRest;

        // change the path of all pseudo word tables
        for (MergeSplitRestaurant oldRest : oldRests) {
            for (Restaurant pseudoTurnRest : oldRest.getTurnRestaurants()) {
                for (Table pseudoWordTable : pseudoTurnRest.getActiveTables()) {
                    hier_prior.removeCustomer(oldRest, pseudoWordTable);
                    hier_prior.addCustomer(mergeRest, pseudoWordTable);
                }
                pseudoTurnRest.setParent(mergeRest);
                mergeRest.addTurnRestaurant(pseudoTurnRest.getIndex(), pseudoTurnRest);
            }
            parent.removeChild(oldRest);
        }
    }

    private void split(MergeSplitRestaurant oriRest, int t) {
        Restaurant parent = oriRest.getParent();

        MergeSplitRestaurant[] newRests = new MergeSplitRestaurant[2];
        for (int i = 0; i < newRests.length; i++) {
            newRests[i] = new MergeSplitRestaurant(
                    Restaurant.UNASSIGNED_REST_INDEX,
                    parent,
                    oriRest.getHyperparameter(),
                    oriRest.isTerminal(),
                    oriRest.getPathAssumption());
        }

        for (MergeSplitRestaurant newRest : newRests) {
            parent.addChild(newRest);
        }

        for (int turn_index : oriRest.getTurnRestaurantIndices()) {
            int ii = turn_index < t ? 0 : 1;
            newRests[ii].addTurnRestaurant(turn_index, oriRest.getTurnRestaurant(turn_index));

            for (Table pseudoWordTable : oriRest.getTurnRestaurant(turn_index).getActiveTables()) {
                hier_prior.removeCustomer(oriRest, pseudoWordTable);
                hier_prior.addCustomer(newRests[ii], pseudoWordTable);
            }
            oriRest.getTurnRestaurant(turn_index).setParent(newRests[ii]);
        }

        parent.removeChild(oriRest);
    }

    private void sampleZ(int c, int t, int n) {
        if (debug) {
            logln("Iter " + iter + ". Sampling Z: c=" + c + "; t=" + t + "; n=" + n);
        }

        // remove word
        topic_word.decrement(z[c][t][n], words[c][t][n]);
        hier_prior.removeCustomer(
                this.turnPseudoRests[c][t].getParent(), // leave of the path
                this.turnPseudoRests[c][t].getTable(n));

        // sample z
        ArrayList<Integer> table_indices = new ArrayList<Integer>();
        ArrayList<Double> table_probs = new ArrayList<Double>();

        // prob assigning new table
        table_indices.add(topic_word.getNumDistributions());
        table_probs.add(1.0 / topic_word.getDimension()
                * hier_prior.getScore(
                this.turnPseudoRests[c][t].getParent(), // the leave of the path
                Table.UNASSIGNED_GLOBAL_INDEX,
                1.0 / hier_prior.getNumTopics()));

        // prob assigning existing topics
        for (int globalIndex : topic_word.getDistributionIndices()) {
            double prob = topic_word.getLikelihood(globalIndex, words[c][t][n])
                    * hier_prior.getScore(
                    this.turnPseudoRests[c][t].getParent(), // the leave of the path
                    globalIndex,
                    1.0 / hier_prior.getNumTopics());

            table_indices.add(globalIndex);
            table_probs.add(prob);
        }
        int array_index = SamplerUtils.scaleSample(table_probs); // should use logScaleSample
        int sampled_z = table_indices.get(array_index);

        // add word
        Table wordPseudoTable = this.turnPseudoRests[c][t].getTable(n);
        wordPseudoTable.setGlobalIndex(sampled_z);
        hier_prior.addCustomer(
                this.turnPseudoRests[c][t].getParent(), // leave of the path
                wordPseudoTable); // word pseudo table with updated global index
        topic_word.increment(sampled_z, words[c][t][n]);
        z[c][t][n] = sampled_z;

        if (debug) {
            validate("Sampling Z: c=" + c + "; t=" + t + "; n=" + n);
        }
    }

    @Override
    public void sample() {
        openLogger();

        if (verbose) {
            logln("lambda = " + this.hyperparams.get(LAMBDA));
            logln("gamma = " + this.hyperparams.get(LAMBDA));
            logln("alpha = " + this.hyperparams.get(ALPHA));
            logln("alpha_0 = " + this.hyperparams.get(ALPHA_0));
            logln("alpha_c = " + this.hyperparams.get(ALPHA_C));
            logln("M (author vocab size) = " + this.M);
            logln("V (word vocab size) = " + this.V);
            logln("C (# conversations) = " + this.C);
            int au = 0;
            for (int[] a : this.authors) {
                au += a.length;
            }
            logln("T (total # turns) = " + au);
            logln("Path assumption: " + this.pathAssumption.toString());
            logln("Max_iter = " + MAX_ITER + ". Burn_in = " + BURN_IN + ". Lag = " + LAG);
        }

        initialize();

        iterate();

        closeLogger();
    }

    @Override
    public void outputLogLikelihoods(String outputFile) throws Exception {
        IOUtils.outputLogLikelihoods(this.logLikelihoods, outputFile);
    }

    @Override
    public void outputAvgSampledL(String outputFile) throws Exception {
        double[] values = null;
        for (int[][] sampledL : sampledLs) {
            int[] straightenL = straightenArray(sampledL);
            if (values == null) {
                values = new double[straightenL.length];
            }
            for (int i = 0; i < straightenL.length; i++) {
                values[i] += straightenL[i];
            }
        }

        for (int i = 0; i < values.length; i++) {
            values[i] /= sampledLs.size();
        }

        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        for (int t = 0; t < values.length; t++) {
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

    protected static int[] straightenArray(int[][] arr) {
        ArrayList<Integer> straighten_list = new ArrayList<Integer>();

        for (int[] conv_l : arr) {
            for (int single_l : conv_l) {
                straighten_list.add(single_l);
            }
            straighten_list.add(-1);
        }

        int[] straighten_arr = new int[straighten_list.size()];
        for (int i = 0; i < straighten_arr.length; i++) {
            straighten_arr[i] = straighten_list.get(i);
        }
        return straighten_arr;
    }

    public void outputShiftAssignments(String outputFile) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        int[] straightenL = straightenArray(l);
        for (int i : straightenL) {
            writer.write(i + "\n");
        }
        writer.close();
    }

    @Override
    public double[][] getPhi() {
        return topic_word.getDistribution();
    }

    public double[][] getTheta() {
        int numTopic = hier_prior.getNumTopics();
        ArrayList<Integer> topicIndices = getTopicIndices();

        ArrayList<double[]> thetaList = new ArrayList<double[]>();
        for (int c = 0; c < C; c++) {
            for (int t = 0; t < words[c].length; t++) {
                Restaurant turnRest = turnPseudoRests[c][t];
                Hashtable<Integer, Integer> globalTopicCount = new Hashtable<Integer, Integer>();
                for (Table wordTable : turnRest.getActiveTables()) {
                    int topicIndex = wordTable.getGlobalIndex();
                    Integer count = globalTopicCount.get(topicIndex);
                    if (count == null) {
                        globalTopicCount.put(topicIndex, 1);
                    } else {
                        globalTopicCount.put(topicIndex, count + 1);
                    }
                }

                double[] turn_topic_distr = new double[numTopic];
                for (int i = 0; i < topicIndices.size(); i++) {
                    Integer count = globalTopicCount.get(topicIndices.get(i));
                    if (count == null) {
                        count = 0;
                    }
                    turn_topic_distr[i] = (count + this.hyperparams.get(ALPHA_C))
                            / (turnRest.getNumTables() + numTopic * this.hyperparams.get(ALPHA_C));
                }
                thetaList.add(turn_topic_distr);
            }
            thetaList.add(null);
        }

        double[][] theta = new double[thetaList.size()][];
        for (int i = 0; i < thetaList.size(); i++) {
            theta[i] = thetaList.get(i);
        }
        return theta;
    }

    public ArrayList<Integer> getTopicIndices() {
        return this.topic_word.getSortedTopicIndices();
    }

    public void outputTopicIndices(String outputFile) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        ArrayList<Integer> sortedTopicIndices = topic_word.getSortedTopicIndices();
        for (int i : sortedTopicIndices) {
            writer.write(i + "\n");
        }
        writer.close();
    }

    public void outputTopicAssignments(String outputFile) throws Exception {
        int totalNumTurns = 0;
        for (int c = 0; c < C; c++) {
            totalNumTurns += words[c].length + 1;
        }
        int[][] topicAssignments = new int[totalNumTurns][];

        int turnCount = 0;
        for (int c = 0; c < C; c++) {
            for (int t = 0; t < words[c].length; t++) {
                topicAssignments[turnCount] = z[c][t];
                turnCount++;
            }
            topicAssignments[turnCount] = new int[0];
            turnCount++;
        }

        IOUtils.outputLatentVariableAssignment(topicAssignments, outputFile);
    }

    public int[][] inputTopicAssignments(String inputFile) throws Exception {
        return IOUtils.inputLatentVariableAssignment(inputFile);
    }

    @Override
    public double[][] getPi() {
        return author_shift.getDistribution();
    }

    @Override
    public void outputPhi(String outputFile) throws Exception {
        IOUtils.outputLatentVariables(this.getPhi(), outputFile);
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
    public double[][] inputPhi(String inputFile) throws Exception {
        return IOUtils.inputLatentVariables(inputFile);
    }

    @Override
    public void outputHyperparameters(String outputFile) throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
        writer.write("Iter\talpha\talpha_0\talpha_c\tlambda\tgamma\n");
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
    public double getLogLikelihood() {
        double val = 0.0;
        val += author_shift.getLogLikelihood();
        val += topic_word.getLogLikelihood();
        val += hier_prior.getLogPrior();
        return val;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> testHyperparams) {
        double val = 0.0;
        val += author_shift.getLogLikelihood(testHyperparams.get(GAMMA));
        val += topic_word.getLogLikelihood(testHyperparams.get(LAMBDA));

        double[] testHierHyperparameters = new double[3];
        testHierHyperparameters[0] = testHyperparams.get(ALPHA);
        testHierHyperparameters[1] = testHyperparams.get(ALPHA_0);
        testHierHyperparameters[2] = testHyperparams.get(ALPHA_C);

        val += hier_prior.getLogPrior(testHierHyperparameters);
        return val;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(hier_prior.toString());
        return str.toString();
    }

    @Override
    public void validate(String location) {
        hier_prior.validate();
        topic_word.validate();
        author_shift.validate();
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
}

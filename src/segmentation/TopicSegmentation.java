package segmentation;

import core.AbstractExperiment;
import java.io.BufferedReader;
import java.io.File;
import java.util.*;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import segmentation.nonparametric.sampler.NonparametricAuthorShiftPathAssumptionSampler;
import segmentation.parametric.sampler.AuthorShiftSampler;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class TopicSegmentation extends AbstractExperiment {

    public static final long RAND_SEED = 1123581322;
    protected static CommandLineParser parser;
    protected static Options options;
    protected static CommandLine cmd;

    public static enum InternalAssumption {

        MAXIMAL, MINIMAL, NORMAL
    }
    public static final String RANDOM_INIT = "rand";
    public static final String PRIOR_INIT = "prior";
    public static final String MAXIMAL_INIT = "max";
    protected static final String ResultFolder = "results/";
    protected static final String SummaryFolder = "summary/";
    protected static final String QualResultFolder = "qual_results/";
    protected static final String WindowDiffResult = "WindowDiff";
    protected static final String PkResult = "Pk";
    protected static final String LevenshteinDistanceResult = "LevenshteinDistance";
    protected static final String DamerauLevenshteinDistanceResult = "DamerauLevenshteinDistance";
    protected static final String EMDResult = "EMD";
    protected static final String AverageResultsFile = "AverageResults.txt";
    protected static int[][] documents;
    protected static int[] authors;
    protected static String[] conversation_names;
    protected static String[] texts;
    protected static int[] turn_indices;
    protected static int[] groundtruth_segments;
    protected static double[] groundtruth_topicshift_scores;
    protected static ArrayList<String> author_vocab;
    protected static ArrayList<String> word_vocab;
    protected static Hashtable<String, Integer> doc_indices; // store where each document starts in the documents array
    protected static int vocab_size;
    protected static int num_unique_authors;
    protected static double[] idfs;
    protected static ArrayList<String> selectedExptDocIds;
    protected static ArrayList<AbstractSegmentationModel> models;
    // current experimental doc
    protected static String exptDocId;
    protected static int[][] exptDoc;
    protected static int[] exptAuthors;
    protected static int[] exptTurnIndices;
    protected static int[] exptGroundtruthSegments;
    protected static double[] expGroundtruthTopicShiftScores;
    protected static String ldaFormatInputFolder;
    protected static boolean hasGroundtruth = false;
    protected static boolean hasGroundtruthScore = false;
    protected static int[] window_sizes = {2, 3, 5, 7, 8, 9, 10, 12, 15, 20};
    protected static String model;

    protected static void addOption(String optName, String optDesc) {
        options.addOption(OptionBuilder.withLongOpt(optName)
                .withDescription(optDesc)
                .hasArg()
                .withArgName(optName)
                .create());
    }

    @Override
    public void setup() throws Exception {
        if (verbose) {
            System.out.println("Setting up ...");
        }
        loadData(ldaFormatInputFolder + datasetName);
    }

    /**
     * Will need to replace this by creating a Dataset object
     */
    private void loadData(String path) throws Exception {
        if (verbose) {
            System.out.println("--- Loading data from " + datasetName + " ...");
        }

        Scanner word_scanner = new Scanner(new File(path + ".words"));
        Scanner author_scanner = new Scanner(new File(path + ".authors"));

        int num_docs = word_scanner.nextInt();
        int num_turns = word_scanner.nextInt();
        int total_num_words = 0; // FYI

        documents = new int[num_turns][];
        authors = new int[num_turns];
        turn_indices = new int[num_turns];
        conversation_names = new String[num_turns];

        num_unique_authors = 0;
        vocab_size = 0;

        for (int ii = 0; ii < num_turns; ii++) {
            authors[ii] = author_scanner.nextInt();
            if (authors[ii] > num_unique_authors) {
                num_unique_authors = authors[ii];
            }

            // Shows a break between conversational units
            if (authors[ii] < 0) {
                documents[ii] = new int[0];
                //word_scanner.nextInt();
                continue;
            }

            int num_words = word_scanner.nextInt();
            total_num_words += num_words;
            documents[ii] = new int[num_words];
            for (int jj = 0; jj < num_words; jj++) {
                documents[ii][jj] = word_scanner.nextInt();
                if (documents[ii][jj] > vocab_size) {
                    vocab_size = documents[ii][jj];
                }
            }
        }

        vocab_size++;
        num_unique_authors++;
        word_scanner.close();
        author_scanner.close();

        File turnIndexFile = new File(path + ".turnindex");
        if (turnIndexFile.exists()) {
            BufferedReader reader = IOUtils.getBufferedReader(path + ".turnindex");
            String line;
            int lineCount = 0;
            while ((line = reader.readLine()) != null) {
                turn_indices[lineCount] = Integer.parseInt(line);
                lineCount++;
            }
            reader.close();
        }

        // index document positions
        doc_indices = new Hashtable<String, Integer>();
        BufferedReader reader = IOUtils.getBufferedReader(path + ".shows");
        String line;
        String pre_show = "";
        int index = 0;
        while ((line = reader.readLine()) != null) {
            if (pre_show.equals("")) {
                doc_indices.put(line, index);
            }
            conversation_names[index] = line;
            index++;
            pre_show = line;
        }
        reader.close();

        // load author vocab
        reader = IOUtils.getBufferedReader(path + ".whois");
        author_vocab = new ArrayList<String>();
        while ((line = reader.readLine()) != null) {
            author_vocab.add(line);
        }
        reader.close();

        // load word vocab
        word_vocab = new ArrayList<String>();
        if (new File(path + ".voc").exists()) {
            reader = IOUtils.getBufferedReader(path + ".voc");
            while ((line = reader.readLine()) != null) {
                word_vocab.add(line);
            }
            reader.close();
        }

        File textFile = new File(path + ".text");
        texts = new String[num_turns];
        if (textFile.exists()) {
            reader = IOUtils.getBufferedReader(path + ".text");
            int linecount = 0;
            while ((line = reader.readLine()) != null) {
                if (!line.equals("")) {
                    texts[linecount] = line.split("\t")[2];
                }
                linecount++;
            }
            reader.close();
        }

        int numBoundaries = 0;
        if (hasGroundtruth) {
            groundtruth_segments = new int[authors.length];
            int linecount = 0;
            reader = IOUtils.getBufferedReader(path + ".segment");
            while ((line = reader.readLine()) != null) {
                if (line.trim().equals("")) {
                    groundtruth_segments[linecount] = -2;
                } else {
                    groundtruth_segments[linecount] = Integer.parseInt(line.split("\t")[0]);
                }

                if (groundtruth_segments[linecount] == 1) {
                    numBoundaries++;
                }
                linecount++;
            }
            reader.close();
        }
        if (hasGroundtruthScore) {
            groundtruth_topicshift_scores = new double[authors.length];
            int linecount = 0;
            reader = IOUtils.getBufferedReader(path + ".topic");
            while ((line = reader.readLine()) != null) {
                if (line.trim().equals("")) {
                    groundtruth_topicshift_scores[linecount] = -1;
                } else {
                    groundtruth_topicshift_scores[linecount] = Double.parseDouble(line.split("\t")[0]);
                }
                linecount++;
            }
            reader.close();
        }

        if (verbose) {
            System.out.println("--- --- # documents: " + num_docs + "; " + doc_indices.size());
            if (hasGroundtruth) {
                System.out.println("--- --- # boundaries: " + numBoundaries);
            }
            System.out.println("--- --- # turns: " + documents.length);
            System.out.println("--- --- vocab size: " + vocab_size + "; " + word_vocab.size());
            System.out.println("--- --- # unique authors: " + num_unique_authors + "; " + author_vocab.size());
            System.out.println("--- --- # turn indices: " + turn_indices.length + "\t" + authors.length);
            System.out.println("--- --- total number of words: " + total_num_words);
        }

        // compute TFIDF
        if (verbose) {
            System.out.println("--- Computing TF-IDF ...");
        }

        idfs = new double[vocab_size];
        int num_acutal_shows = 0; // exclude boundary empty shows
        for (int[] doc : documents) {
            if (doc.length == 0) {
                continue;
            }
            num_acutal_shows++;
            Set<Integer> doc_word_set = new HashSet<Integer>(); // set of unique words in the show
            for (int word : doc) {
                doc_word_set.add(word);
            }
            for (int word : doc_word_set) {
                idfs[word]++; // first count dfs
            }
        }
        for (int i = 0; i < idfs.length; i++) {
            idfs[i] = Math.log((double) (num_acutal_shows + 1) / (idfs[i] + 1)); // compute idfs
        }


        // select experimental shows
        if (verbose) {
            System.out.println("--- Seleting experimental shows ...");
        }
        selectedExptDocIds = new ArrayList<String>();
        //selectedExptDocIds.add(exptDocId);
        for (String showId : doc_indices.keySet()) {
            selectedExptDocIds.add(showId);
        }
    }

    @Override
    public void run() throws Exception {
        if (verbose) {
            System.out.println("Running ...");
        }
        if (model.equals("param")) {
            runParametricSITS();
        } else if (model.equals("non-param")) {
            runNonparametricSITS();
        } else {
            throw new RuntimeException("Model " + model + " is not suppported");
        }
    }

    private void runParametricSITS() throws Exception {
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double gamma = CLIUtils.getDoubleArgument(cmd, "gamma", 0.25);
        int K = CLIUtils.getIntegerArgument(cmd, "K", 25);

        AuthorShiftSampler sampler = new AuthorShiftSampler();
        sampler.configure(experimentPath, documents, authors,
                K, author_vocab.size(), word_vocab.size(),
                alpha, beta, gamma,
                burn_in, max_iters, sample_lag);
        sampler.setPrefix("RANDOM"); // random intialization

        String asm_folder = experimentPath + sampler.getSamplerFolder();
        IOUtils.createFolder(asm_folder);

        sampler.sample();
        sampler.outputLogLikelihoods(asm_folder + "loglikelihood.txt");
        sampler.outputShiftAssignments(asm_folder + "shift_asgn.txt");
        sampler.outputTopicAssignments(asm_folder + "topic_asgn.txt");

        sampler.outputPhi(asm_folder + "phi.txt");
        sampler.outputPi(asm_folder + "pi.txt");
        sampler.outputTheta(asm_folder + "theta.txt");
        IOUtils.outputTopWords(sampler.getPhi(), word_vocab, 20, asm_folder + "topwords.txt");
    }

    private void runNonparametricSITS() throws Exception {
        int K = CLIUtils.getIntegerArgument(cmd, "K", 25); // for initialization
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double gamma = CLIUtils.getDoubleArgument(cmd, "gamma", 0.25);
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double alpha_0 = CLIUtils.getDoubleArgument(cmd, "alpha_0", 0.1);
        double alpha_c = CLIUtils.getDoubleArgument(cmd, "alpha_C", 0.1);

        NonparametricAuthorShiftPathAssumptionSampler sampler = new NonparametricAuthorShiftPathAssumptionSampler();
        sampler.configure(experimentPath, documents, authors,
                author_vocab.size(), word_vocab.size(),
                beta, gamma, alpha, alpha_0, alpha_c,
                InternalAssumption.MINIMAL);
        sampler.setSamplerConfiguration(burn_in, max_iters, sample_lag);
        sampler.setK(K);
        sampler.setPrefix(TopicSegmentation.RANDOM_INIT);
        sampler.setParamsOptimized(true);
        sampler.setDebug(false);
        String nonparametricSamplerFolder = experimentPath + sampler.getSamplerName() + "/";
        System.out.println("Sampling " + sampler.getSamplerName());

        IOUtils.createFolder(nonparametricSamplerFolder);

        sampler.sample();

        sampler.outputLogLikelihoods(nonparametricSamplerFolder + "loglikelihood.txt");
        sampler.outputAvgSampledL(nonparametricSamplerFolder + "avg_sampled_shift.txt");
        sampler.outputShiftAssignments(nonparametricSamplerFolder + "shift_asgn.txt");
        sampler.outputTopicAssignments(nonparametricSamplerFolder + "topic_asgn.txt");
        sampler.outputTopicIndices(nonparametricSamplerFolder + "topic_indices.txt");

        sampler.outputPhi(nonparametricSamplerFolder + "phi.txt");
        sampler.outputPi(nonparametricSamplerFolder + "pi.txt");

        sampler.outputHyperparameters(nonparametricSamplerFolder + "hyperparameters.txt");
        IOUtils.outputTopWords(sampler.getPhi(), word_vocab, 20, nonparametricSamplerFolder + "topwords.txt");
    }

    @Override
    public void evaluate() throws Exception {
    }

    public static void main(String[] args) {
        try {

            // create the command line parser
            parser = new BasicParser();

            // create the Options
            options = new Options();

            // input data
            addOption("dataset", "Dataset name");
            addOption("input", "Input folder");
            addOption("output", "Outputfolder");

            // sampling
            addOption("burnIn", "Burn-in");
            addOption("maxIter", "Maximum number of iterations");
            addOption("sampleLag", "Sample lag");

            // parametric
            addOption("K", "Number of topics");

            // hyperparameters
            addOption("alpha", "alpha");
            addOption("alpha_0", "alpha_0");
            addOption("alpha_C", "alpha_C");
            addOption("lambda", "lambda");
            addOption("beta", "beta");
            addOption("gamma", "gamma");

            addOption("model", "Model");

            options.addOption("v", false, "verbose");
            options.addOption("d", false, "debug");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(), options);
                return;
            }

            datasetName = CLIUtils.getStringArgument(cmd, "dataset", "debate2008");
            ldaFormatInputFolder = CLIUtils.getStringArgument(cmd, "input", "data/" + datasetName + "/ldaformat/");
            experimentPath = CLIUtils.getStringArgument(cmd, "output", "data/segmentation/" + datasetName + "/");
            hasGroundtruth = false;
            hasGroundtruthScore = false;

            burn_in = CLIUtils.getIntegerArgument(cmd, "burnIn", 2500);
            max_iters = CLIUtils.getIntegerArgument(cmd, "maxIter", 5000);
            sample_lag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 100);

            model = CLIUtils.getStringArgument(cmd, "model", "non-param");

            TopicSegmentation expt = new TopicSegmentation();
            expt.setup();
            expt.run();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static String getHelpString() {
        return "java -cp dist/teaparty.jar " + TopicSegmentation.class.getName() + " -help";
    }
}

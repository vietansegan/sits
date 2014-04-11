package segmentation.parametric.sampler;

import java.io.BufferedReader;
import java.util.ArrayList;
import util.IOUtils;

/**
 * Separate main class for running parametric sampler. To use the whole
 * pipeline, look at MainSegmentationExperiment
 *
 * @author vietan
 */
public class MainParametricSampler {

    protected static int[][] words;
    protected static int[] authors;
    protected static ArrayList<String> show_ids;
    protected static ArrayList<String> word_vocab;
    protected static ArrayList<String> author_vocab;
    protected static int K = 25;
    protected static double alpha = 0.1;
    protected static double beta = 0.1;
    protected static double gamma = 0.1;
    protected static int burnIn = 5; // burn-in
    protected static int maxIter = 100; // maximum number of iterations
    protected static int sampleLag = 1; // for outputing log-likelihood
    protected static String samplingFolder;
    protected static String ldaFormatInputFolder;
    protected static String datasetName;

    public static void main(String[] arg) {
        try {
            datasetName = "icsi";
            run();

            //datasetName = "cf";
            //run();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void run() throws Exception {
        ldaFormatInputFolder = "data/" + datasetName + "/ldaformat/";
        samplingFolder = "data/segmentation/" + datasetName + "/";
        IOUtils.createFolder(samplingFolder);

        System.out.println("\n Loading data ...");
        loadData(ldaFormatInputFolder + datasetName);

        System.out.println("\n Sampling ...");
        runGibbsSampling();
        //runLDASampling();
    }

    private static void runGibbsSampling() throws Exception {
        String asm_folder = samplingFolder + "asm/";
        IOUtils.createFolder(asm_folder);

        AuthorShiftSampler asm = new AuthorShiftSampler();
        asm.configure(samplingFolder, words, authors,
                K, author_vocab.size(), word_vocab.size(),
                alpha, beta, gamma,
                burnIn, maxIter, sampleLag);
        asm.sample();
        asm.outputLogLikelihoods(asm_folder + "loglikelihood.txt");
        asm.outputShiftAssignments(asm_folder + "shift_asgn.txt");
        asm.outputTopicAssignments(asm_folder + "topic_asgn.txt");

        asm.outputPhi(asm_folder + "phi.txt");
        asm.outputPi(asm_folder + "pi.txt");
        asm.outputTheta(asm_folder + "theta.txt");
    }

    protected static void loadData(String path) throws Exception {
        // show vocab
        show_ids = new ArrayList<String>();
        String pre_show = new String();
        BufferedReader reader = IOUtils.getBufferedReader(path + ".shows");
        String line;
        while ((line = reader.readLine()) != null) {
            if (!line.equals("") && !line.equals(pre_show)) {
                show_ids.add(line);
                pre_show = line;
            }
        }
        reader.close();
        System.out.println("# shows: " + show_ids.size());

        // word vocab
        word_vocab = new ArrayList<String>();
        reader = IOUtils.getBufferedReader(path + ".voc");
        while ((line = reader.readLine()) != null) {
            word_vocab.add(line);
        }
        reader.close();
        System.out.println("# words: " + word_vocab.size());

        // author vocab
        author_vocab = new ArrayList<String>();
        reader = IOUtils.getBufferedReader(path + ".whois");
        while ((line = reader.readLine()) != null) {
            author_vocab.add(line);
        }
        reader.close();
        System.out.println("# authors: " + author_vocab.size());

        // show words
        reader = IOUtils.getBufferedReader(path + ".words");
        int num_shows = Integer.parseInt(reader.readLine());
        int num_turns = Integer.parseInt(reader.readLine());
        words = new int[num_turns][];

        int turn_count = 0;
        while ((line = reader.readLine()) != null) {
            if (line.equals("")) {
                words[turn_count] = new int[0];
                turn_count++;
                continue;
            }

            String[] sline = line.split("\t");
            int num_turn_words = Integer.parseInt(sline[0]);
            int[] turn = new int[num_turn_words];
            if (num_turn_words > 0) {
                String[] turn_words = sline[1].split(" ");
                for (int i = 0; i < num_turn_words; i++) {
                    turn[i] = Integer.parseInt(turn_words[i]);
                }
            }
            words[turn_count] = turn;
            turn_count++;
        }
        reader.close();
        System.out.println("# turns: " + words.length);

        reader = IOUtils.getBufferedReader(path + ".authors");
        authors = new int[num_turns];
        turn_count = 0;
        while ((line = reader.readLine()) != null) {
            authors[turn_count] = Integer.parseInt(line);
            turn_count++;
        }
        reader.close();
        System.out.println("# turns: " + authors.length);
    }

    public static int getNumShows() {
        return show_ids.size();
    }

    public static int getNumTurns() {
        return words.length;
    }

    public static int getNumActualTurns() {
        return getNumTurns() - getNumShows(); // exclude separators
    }
}
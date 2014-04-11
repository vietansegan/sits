package segmentation;

import java.io.BufferedWriter;
import java.util.ArrayList;
import java.util.Collections;
import util.IOUtils;
import util.RankingItem;

/**
 * Model to compute the segmentation scores/probabilities from the result of a
 * sampler
 *
 * @author vietan
 */
public abstract class AbstractSegmentationModel {

    public static final String SegmentationScoreFile = "segmentation_scores.txt";
    protected String folder;
    protected String modelName;
    protected int[][] showWords; // show's words
    protected int[] showSpeakers; // show's speakers
    protected double[] segmentScores;
    protected int[] predictedSegmentation;
    protected SegmentationMeasure performance;

    public AbstractSegmentationModel(String folder) {
        this.folder = folder;
    }

    public AbstractSegmentationModel(String folder, int[][] show_words, int[] show_speakers) {
        this.folder = folder;
        this.showWords = show_words;
        this.showSpeakers = show_speakers;
    }

    public String getModelFolder() {
        return getModelName() + "/";
    }

    public String getModelPath() {
        return this.folder + this.getModelFolder();
    }

    public int getNumTurns() {
        return this.showSpeakers.length;
    }

    public void setSegmentationScores(double[] s) {
        this.segmentScores = s;
    }

    public double[] getSegmentationScores() {
        return this.segmentScores;
    }

    public SegmentationMeasure getPerformance() {
        return this.performance;
    }

    public void evaluate(int[] groudtruthSegmentation) throws Exception {
        this.performance = new SegmentationMeasure(groudtruthSegmentation, predictedSegmentation);
    }

    public double getWindowDiff(int k) {
        return this.performance.getWindowDiff(k);
    }

    public double getPk(int k) {
        return this.performance.getPk(k);
    }

    public int[] getPredictedSegmentation() {
        return this.predictedSegmentation;
    }

    public int getDamerauLevenshteinDistance() {
        return this.performance.getDamerauLevenshteinDistance();
    }

    public int getLevenshteinDistance() {
        return this.performance.getLevenshteinDistance();
    }

    public void getPredictedSegmentation(double threshold) {
        predictedSegmentation = new int[this.segmentScores.length];
        for (int i = 0; i < predictedSegmentation.length; i++) {
            if (segmentScores[i] > threshold) {
                predictedSegmentation[i] = 1;
            } else {
                predictedSegmentation[i] = 0;
            }
        }
    }

    public void getPredictedSegmentation(int numSegments) {
        ArrayList<RankingItem<Integer>> rankedTurns = new ArrayList<RankingItem<Integer>>();
        for (int t = 0; t < segmentScores.length; t++) {
            rankedTurns.add(new RankingItem<Integer>(t, this.segmentScores[t]));
        }
        Collections.sort(rankedTurns);

        predictedSegmentation = new int[this.segmentScores.length];
        for (int i = 0; i < numSegments; i++) {
            int predShift = rankedTurns.get(i).getObject();
            predictedSegmentation[predShift] = 1;
        }
    }

    public void setModelName(String name) {
        this.modelName = name;
        IOUtils.createFolder(this.getModelPath());
    }

    public String getModelName() {
        return this.modelName;
    }

    /**
     * To compute the segmentScores
     */
    public abstract void segment() throws Exception;

    public void outputSegmetationScores() throws Exception {
        BufferedWriter writer = IOUtils.getBufferedWriter(this.getModelPath() + SegmentationScoreFile);
        for (int i = 0; i < this.segmentScores.length; i++) {
            writer.write(i + "\t" + this.segmentScores[i] + "\n");
        }
        writer.close();
    }
}

package segmentation;

/**
 *
 * @author vietan
 */
public class SegmentationMeasure {

    private static final int HAS_SHIFT = 1;
    private int[] referenceSegmentation;
    private int[] hypothesizedSegmentation;
    private double[] hypothesizedScores;

    public SegmentationMeasure(int[] ref, int[] hyp) {
        this.referenceSegmentation = ref;
        this.hypothesizedSegmentation = hyp;
    }

    public int[] getReferenceSegmentation() {
        return this.referenceSegmentation;
    }

    public int[] getHypothesizedSegmentation() {
        return this.hypothesizedSegmentation;
    }

    /**
     * Compute the Beeferman's Probability of Segmentation Error
     *
     * @param k The window size
     *
     * Beeferman D, Berger A and Lafferty J D, Statistical Models for Text
     * Segmentation, Machine Learning 34 (1-3), 177-210 (1999)
     */
    public double getPk(int k) {
        int num = 0;
        for (int i = 0; i < this.hypothesizedSegmentation.length - k; i++) {
            int dH = this.indicatorFunction(hypothesizedSegmentation, i, i + k);
            int dR = this.indicatorFunction(referenceSegmentation, i, i + k);
            num += (dH + dR) % 2;
        }
        return (double) num / (this.hypothesizedSegmentation.length - k);
    }

    public double getProbMiss(int k) {
        double num = 0;
        double den = 0;
        for (int i = 0; i < this.hypothesizedSegmentation.length - k; i++) {
            num += indicatorFunction(hypothesizedSegmentation, i, i + k)
                    * (1 - indicatorFunction(referenceSegmentation, i, i + k));
            den += (1 - indicatorFunction(referenceSegmentation, i, i + k));
        }
        return num / den;
    }

    public double getProbFalseAlarm(int k) {
        double num = 0;
        double den = 0;
        for (int i = 0; i < this.hypothesizedSegmentation.length - k; i++) {
            num += (1 - indicatorFunction(hypothesizedSegmentation, i, i + k))
                    * indicatorFunction(referenceSegmentation, i, i + k);
            den += indicatorFunction(referenceSegmentation, i, i + k);
        }
        return num / den;
    }

    private int indicatorFunction(int[] segmentaion, int start, int end) {
        return getNumBoundaries(segmentaion, start, end) > 0 ? 1 : 0;
    }

    public double getWindowDiff(int k) {
        return getWDMiss(k) + getWDFalseAlarm(k);
    }

    public double getWDMiss(int k) {
        int num = 0;
        for (int i = 0; i < this.referenceSegmentation.length - k; i++) {
            if (getNumBoundaries(hypothesizedSegmentation, i, i + k)
                    < getNumBoundaries(referenceSegmentation, i, i + k)) {
                num++;
            }
        }
        return (double) num / (this.referenceSegmentation.length - k);
    }

    public double getWDFalseAlarm(int k) {
        int num = 0;
        for (int i = 0; i < this.referenceSegmentation.length - k; i++) {
            if (getNumBoundaries(hypothesizedSegmentation, i, i + k)
                    > getNumBoundaries(referenceSegmentation, i, i + k)) {
                num++;
            }
        }
        return (double) num / (this.referenceSegmentation.length - k);
    }

    private int getNumBoundaries(int[] segmentation, int start, int end) {
        int num_boundaries = 0;
        for (int i = start + 1; i <= end; i++) {
            if (segmentation[i] == HAS_SHIFT) {
                num_boundaries++;
            }
        }
        return num_boundaries;
    }

    public int getLevenshteinDistance() {
        return computeLevenshteinDistance(this.hypothesizedSegmentation, this.referenceSegmentation);
    }

    private static int minimum(int a, int b, int c) {
        return Math.min(Math.min(a, b), c);
    }

    private static int computeLevenshteinDistance(int[] str1, int[] str2) {
        int[][] distance = new int[str1.length + 1][str2.length + 1];

        for (int i = 0; i <= str1.length; i++) {
            distance[i][0] = i;
        }
        for (int j = 0; j <= str2.length; j++) {
            distance[0][j] = j;
        }

        for (int i = 1; i <= str1.length; i++) {
            for (int j = 1; j <= str2.length; j++) {
                distance[i][j] = minimum(
                        distance[i - 1][j] + 1,
                        distance[i][j - 1] + 1,
                        distance[i - 1][j - 1]
                        + ((str1[i - 1] == str2[j - 1]) ? 0 : 1));
            }
        }

        return distance[str1.length][str2.length];
    }

    public int getDamerauLevenshteinDistance() {
        return computeDamerauLevenshteinDistance(hypothesizedSegmentation, referenceSegmentation);
    }

    private static int computeDamerauLevenshteinDistance(int[] str1, int[] str2) {
        int[][] distance = new int[str1.length + 1][str2.length + 1];

        for (int i = 0; i <= str1.length; i++) {
            distance[i][0] = i;
        }
        for (int j = 0; j <= str2.length; j++) {
            distance[0][j] = j;
        }

        for (int i = 1; i <= str1.length; i++) {
            for (int j = 1; j <= str2.length; j++) {
                int cost = ((str1[i - 1] == str2[j - 1]) ? 0 : 1);
                distance[i][j] = minimum(
                        distance[i - 1][j] + 1, // deletion
                        distance[i][j - 1] + 1, // insertion
                        distance[i - 1][j - 1] + cost);  // substitution

                if (i > 1 && j > 1 && str1[i - 1] == str2[j - 2] && str1[i - 2] == str2[j - 1]) {
                    distance[i][j] = Math.min(
                            distance[i][j],
                            distance[i - 2][j - 2] + cost); // transposition
                }
            }
        }

        return distance[str1.length][str2.length];
    }
}

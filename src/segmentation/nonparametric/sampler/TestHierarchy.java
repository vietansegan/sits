package segmentation.nonparametric.sampler;

import segmentation.TopicSegmentation.InternalAssumption;
import util.sampling.ResizableMultinomials;

/**
 *
 * @author vietan
 */
public class TestHierarchy {

    public static void main(String[] args) {
        ResizableMultinomials topic_word = new ResizableMultinomials(3, 1);

        double[] params = {1, 1, 1};
        int C = 3;
        int[][] l = new int[C][];
        int[][] s = new int[C][];
        int[][][] z = new int[C][][];
        int[][][] w = new int[C][][];

        for (int c = 0; c < C; c++) {
            int Tc = 3;
            l[c] = new int[Tc];
            s[c] = new int[Tc];
            z[c] = new int[Tc][];
            w[c] = new int[Tc][];
            int seg_count = 0;
            for (int t = 0; t < Tc; t++) {
                if (t == 0 || t == 1) {
                    l[c][t] = 1;
                } else {
                    l[c][t] = 0;
                }

                int Nct = 3;
                z[c][t] = new int[Nct];
                w[c][t] = new int[Nct];
                for (int n = 0; n < Nct; n++) {
                    z[c][t][n] = n;
                    w[c][t][n] = n;
                    topic_word.increment(z[c][t][n], w[c][t][n]);
                }
                if (l[c][t] == 1) {
                    seg_count++;
                }
                s[c][t] = seg_count - 1;
            }
        }

        Hierarchy tree = new Hierarchy(params, InternalAssumption.MINIMAL);
        Restaurant[][] turnPseudoRests = new Restaurant[z.length][];
        for (int i = 0; i < z.length; i++) {
            turnPseudoRests[i] = new Restaurant[z[i].length];
            for (int j = 0; j < z[i].length; j++) {
                turnPseudoRests[i][j] = new Restaurant(j, null, -1, false, null);
            }
        }
        createHierarchy(tree, l, s, z, turnPseudoRests);
        System.out.println("Original tree\n" + tree.toString());
        System.out.println("Topic-word\n" + topic_word.toString());
        // ---------------  Done initialize tree  ------------------------------

        Restaurant testRest = turnPseudoRests[0][0].getParent();
        int wordIndex = 0;
        for (Table table : testRest.getActiveTables()) {
            System.out.println("Table " + table.getIndex()
                    + ": prob = " + testRest.getProbAssignTable(table.getIndex(), wordIndex, topic_word));
        }
        System.out.println("Table " + Table.EMPTY_TABLE_INDEX
                + ": prob = " + testRest.getProbAssignTable(Table.EMPTY_TABLE_INDEX, wordIndex, topic_word));
    }

    private static void merge(Hierarchy hier, MergeSplitRestaurant preRest, MergeSplitRestaurant posRest) {
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
                    hier.removeCustomer(oldRest, pseudoWordTable);
                    hier.addCustomer(mergeRest, pseudoWordTable);
                }
                pseudoTurnRest.setParent(mergeRest);
                mergeRest.addTurnRestaurant(pseudoTurnRest.getIndex(), pseudoTurnRest);
            }
            parent.removeChild(oldRest);
        }

        System.out.println("After adding");
        System.out.println(hier.toString());
    }

    private static void split(Hierarchy hier, MergeSplitRestaurant oriRest, int t) {
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
                hier.removeCustomer(oriRest, pseudoWordTable);
                hier.addCustomer(newRests[ii], pseudoWordTable);
            }
            oriRest.getTurnRestaurant(turn_index).setParent(newRests[ii]);
        }

        parent.removeChild(oriRest);
    }

    private static void createHierarchy(Hierarchy hier, int[][] l, int[][] s, int[][][] z,
            Restaurant[][] turnPseudoRests) {
        // construct the tree structure
        hier.root = new Restaurant(0, null, hier.hyperparameters[0], false, hier.pathAssumption);
        for (int c = 0; c < l.length; c++) {
            Restaurant conv_rest = new Restaurant(c, hier.root, hier.hyperparameters[1], false, hier.pathAssumption);
            hier.root.addChild(conv_rest);

            MergeSplitRestaurant cur_segm_rest = null;
            for (int t = 0; t < l[c].length; t++) {
                if (l[c][t] == NonparametricAuthorShiftPathAssumptionSampler.HAS_SHIFT) {
                    cur_segm_rest = new MergeSplitRestaurant(s[c][t], conv_rest, hier.hyperparameters[2], true, hier.pathAssumption);
                    conv_rest.addChild(cur_segm_rest);
                }

                // create pseudo restaurant
                turnPseudoRests[c][t].setParent(cur_segm_rest);
                cur_segm_rest.addTurnRestaurant(t, turnPseudoRests[c][t]);
            }
        }

        // from z compute counts
        for (int c = 0; c < z.length; c++) {
            for (int t = 0; t < z[c].length; t++) {
                for (int n = 0; n < z[c][t].length; n++) {
                    int globalIndex = z[c][t][n];

                    // create pseudo table
                    Table wordPseudoTable = turnPseudoRests[c][t].getNewTable();
                    wordPseudoTable.setGlobalIndex(globalIndex);
                    turnPseudoRests[c][t].addCustomerToTable(wordPseudoTable);

                    // add to the hierarchy
                    hier.addCustomer(turnPseudoRests[c][t].getParent(), wordPseudoTable);
                }
            }
        }
    }
}

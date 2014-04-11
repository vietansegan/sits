package segmentation.nonparametric.sampler;

import segmentation.TopicSegmentation.InternalAssumption;

/**
 *
 * @author vietan
 */
public class Hierarchy {

    protected Restaurant root;
    protected InternalAssumption pathAssumption;
    protected double[] hyperparameters; // array of hyperparameters, each for 1 level

    public Hierarchy(double[] params, InternalAssumption path) {
        this.hyperparameters = params;
        this.pathAssumption = path;
        this.root = new Restaurant(0, null, hyperparameters[0], false, pathAssumption);
    }

    public int getNumTopics() {
        return this.root.getNumTables();
    }

    public void setHyperparameters(double[] params) {
        this.hyperparameters = params;
        this.updateHyperparameter(root, 0, this.hyperparameters);
    }

    private void updateHyperparameter(Restaurant rest, int depth, double[] params) {
        rest.setHyperparameter(params[depth]);
        if (!rest.isTerminal()) {
            for (Restaurant child : rest.getChildren()) {
                updateHyperparameter(child, depth + 1, params);
            }
        }
    }

    public double getScore(Restaurant restaurant, int globalIndex, double score) {
        if (restaurant.isTop()) {
            double val = (restaurant.getCountByGlobalIndex(globalIndex)
                    + restaurant.getHyperparameter() * score)
                    / (restaurant.getNumTables() + restaurant.getHyperparameter());
            return val;
        } else {
            double val = (restaurant.getCountByGlobalIndex(globalIndex)
                    + restaurant.getHyperparameter() * getScore(restaurant.getParent(), globalIndex, score))
                    / (restaurant.getNumTables() + restaurant.getHyperparameter());
            return val;
        }
    }

    /**
     * Recursively add a customer to a path of the hierarchy
     *
     * @param restaurant The restaurant at the bottom of the path
     * @param customer The customer to be added
     */
    public void addCustomer(Restaurant restaurant, Table customer) {
        Table table = restaurant.sampleTable(customer);
        int preNumCusts = table.getNumCustomers();
        restaurant.addCustomerToTable(table);
        customer.setDish(table);

        if (restaurant.getParent() != null) {
            if ((this.pathAssumption == InternalAssumption.MINIMAL && preNumCusts == 0)
                    || (this.pathAssumption == InternalAssumption.MAXIMAL)) {
                addCustomer(restaurant.getParent(), table);
            }
        }
    }

    /**
     * Recursively remove a customer from a path of the hierarchy
     *
     * @param restaurant The restaurant at the bottom of the path
     * @param customer The customer to be removed
     */
    public void removeCustomer(Restaurant restaurant, Table customer) {
        Table curTable = customer.getDish();
        if (curTable == null) {
            throw new RuntimeException("Null dish");
        }
        curTable = restaurant.removeCustomerFromTable(curTable);

        if (restaurant.getParent() != null && curTable.isEmpty()) {
            removeCustomer(restaurant.getParent(), curTable);
        }
    }

    /**
     * Compute the log prior of the hierarchy
     */
    public double getLogPrior() {
        double val = 0;
        val += root.getLogPrior();
        for (Restaurant conv_rest : root.getChildren()) {
            val += conv_rest.getLogPrior();
            for (Restaurant segm_rest : conv_rest.getChildren()) {
                val += segm_rest.getLogPrior();
            }
        }
        return val;
    }

    /**
     * Compute the log prior of the hierarchy given the parameters
     */
    public double getLogPrior(double[] params) {
        double val = 0;
        val += root.getLogPrior(params[0]);
        for (Restaurant conv_rest : root.getChildren()) {
            val += conv_rest.getLogPrior(params[1]);
            for (Restaurant segm_rest : conv_rest.getChildren()) {
                val += segm_rest.getLogPrior(params[2]);
            }
        }
        return val;
    }

    public String getState() {
        int totalConvTables = 0;
        int numConvRests = root.getNumChildren();
        int totalSegmTables = 0;
        int numSegmRests = 0;
        for (Restaurant conv_node : root.getChildren()) {
            totalConvTables += conv_node.getNumTables();
            numSegmRests += conv_node.getNumChildren();
            for (Restaurant segm_node : conv_node.getChildren()) {
                totalSegmTables += segm_node.getNumTables();
            }
        }
        return "Level 1: " + root.getNumTables()
                + ", Level 2: " + numConvRests + "," + totalConvTables
                + ", Level 3: " + numSegmRests + "," + totalSegmTables;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("Level 0: ").append(root.toString()).append("\n");
        for (Restaurant conv_node : root.getChildren()) {
            str.append("   Level 1: ").append(conv_node.toString()).append("\n");
            for (Restaurant segm_node : conv_node.getChildren()) {
                str.append("      Level 2: ").append(((MergeSplitRestaurant) segm_node).toString()).append("\n");
            }
        }
        return str.toString();
    }

    public void validate() {
        // validate all restaurants
        root.validate();
        for (Restaurant conv_rest : root.getChildren()) {
            conv_rest.validate();
            for (Restaurant segm_rest : conv_rest.getChildren()) {
                segm_rest.validate();
            }
        }

        // validate
        assertTableCustomerCount(root);
        for (Restaurant conv_rest : root.getChildren()) {
            assertTableCustomerCount(conv_rest);
            for (Restaurant segm_rest : conv_rest.getChildren()) {
                assertTableCustomerCount(segm_rest);
            }
        }
    }

    private void assertTableCustomerCount(Restaurant rest) {
        int totalNumTableChildren = 0;
        if (!rest.isTerminal()) {
            for (Restaurant child : rest.getChildren()) {
                totalNumTableChildren += child.getNumTables();
            }
        } else {
            MergeSplitRestaurant msRest = (MergeSplitRestaurant) rest;
            for (Restaurant turnRest : msRest.getTurnRestaurants()) {
                totalNumTableChildren += turnRest.getNumTables();
            }
        }
        if (rest.getTotalNumCustomers() != totalNumTableChildren) {
            throw new RuntimeException("# customers = " + rest.getTotalNumCustomers() + " != # tables = " + totalNumTableChildren);
        }
    }
}

class SplitRestaurantProposal {
    // input

    MergeSplitRestaurant oriRest;
    int splitPoint;
    // outputs
    MergeSplitRestaurant[] splitRests;
    Restaurant parent;

    public SplitRestaurantProposal(MergeSplitRestaurant r, int t) {
        this.oriRest = r;
        this.splitPoint = t;
    }

    void split() {
        if (!oriRest.containsTurn(splitPoint)) {
            throw new RuntimeException("Split point out of bound");
        }

        InternalAssumption pathAssumption = oriRest.getPathAssumption();

        // initialize
        Restaurant curParent = oriRest.getParent();
        splitRests = new MergeSplitRestaurant[2];
        for (int i = 0; i < splitRests.length; i++) {
            splitRests[i] = new MergeSplitRestaurant(
                    MergeSplitRestaurant.UNASSIGNED_REST_INDEX,
                    curParent,
                    oriRest.getHyperparameter(),
                    oriRest.isTerminal(),
                    oriRest.getPathAssumption());
        }

        // split turns
        for (int turn_index : oriRest.getTurnRestaurantIndices()) {
            int ii = turn_index < splitPoint ? 0 : 1;
            splitRests[ii].addTurnRestaurant(turn_index, oriRest.getTurnRestaurant(turn_index));
        }

        // split tables
        for (int i = 0; i < splitRests.length; i++) {
            MergeSplitRestaurant rest = splitRests[i];
            for (Restaurant turnRestaurant : rest.getTurnRestaurants()) {
                for (Table wordTable : turnRestaurant.getActiveTables()) {
                    Table oriSegmTable = wordTable.getDish();
                    Table table = rest.getTable(oriSegmTable.getIndex());
                    if (table == null) {
                        table = new Table(oriSegmTable.getIndex(), rest);
                        rest.addNewTable(table);
                    }
                    table.setDish(oriSegmTable.getDish());
                    table.setGlobalIndex(oriSegmTable.getGlobalIndex());
                    rest.addCustomerToTable(table);
                }
            }

            // add inactive tables to the list of each split restaurant
            for (int inactiveMergeTable : oriRest.getInactiveTableIndices()) {
                rest.addInactiveTableIndex(inactiveMergeTable);
            }
            for (Table activeMergeTable : oriRest.getActiveTables()) {
                if (!rest.containsTable(activeMergeTable)) {
                    rest.addInactiveTableIndex(activeMergeTable.getIndex());
                }
            }
        }

        // create pseudor parent restaurant
        if (pathAssumption == InternalAssumption.MAXIMAL) {
            parent = curParent;
        } else {
            assert (oriRest.getPathAssumption() == InternalAssumption.MINIMAL);
            // create a new pseudo parent restaurant to compute the log-prior
            // add tables from the cur parent
            parent = new Restaurant(Restaurant.UNASSIGNED_REST_INDEX,
                    curParent.getParent(),
                    curParent.getHyperparameter(),
                    curParent.isTerminal(),
                    curParent.getPathAssumption());
            for (Table curConvTable : curParent.getActiveTables()) {
                Table newConvTable = new Table(curConvTable.getIndex(), parent);
                newConvTable.setGlobalIndex(curConvTable.getGlobalIndex());
                parent.addNewTable(newConvTable); // create conv tables
            }
            // copy customers to tables
            for (Restaurant segmRest : curParent.getChildren()) {
                if (((MergeSplitRestaurant) segmRest).equals(oriRest)) {
                    continue;
                }
                for (Table segmTable : segmRest.getActiveTables()) {
                    Table newConvTable = parent.getTable(segmTable.getDish().getIndex());
                    parent.addCustomerToTable(newConvTable);
                }
            }
            // copy customers from split segm restaurants
            for (MergeSplitRestaurant rest : splitRests) {
                for (Table table : rest.getActiveTables()) {
                    parent.addCustomerToTable(parent.getTable(table.getDish().getIndex()));
                }
            }
        }
    }

    public double getNewParentLogPrior() {
        return parent.getLogPrior();
    }

    public double getCurParentLogPrior() {
        return oriRest.getParent().getLogPrior();
    }

    public double getPreSplitRestaurantLogPrior() {
        return splitRests[0].getLogPrior();
    }

    public double getPosSplitRestaurantLogPrior() {
        return splitRests[1].getLogPrior();
    }
}

class MergeRestaurantProposal {
    // intputs

    MergeSplitRestaurant preRest;
    MergeSplitRestaurant posRest;
    // outputs
    MergeSplitRestaurant mergedRest;
    Restaurant parent;

    public MergeRestaurantProposal(MergeSplitRestaurant pre, MergeSplitRestaurant pos) {
        this.preRest = pre;
        this.posRest = pos;
    }

    void propose() {
        Restaurant curParent = preRest.getParent();
        if (!curParent.equals(posRest.getParent())) {
            throw new RuntimeException("Merging restaurants from different parents");
        }

        InternalAssumption pathAssumption = preRest.getPathAssumption();

        // initialize the merged restaurant
        mergedRest = new MergeSplitRestaurant(Restaurant.UNASSIGNED_REST_INDEX,
                curParent,
                preRest.getHyperparameter(),
                preRest.isTerminal(),
                preRest.getPathAssumption());

        // copy turn restaurants
        for (int t : preRest.getTurnRestaurantIndices()) {
            mergedRest.addTurnRestaurant(t, preRest.getTurnRestaurant(t));
        }
        for (int t : posRest.getTurnRestaurantIndices()) {
            mergedRest.addTurnRestaurant(t, posRest.getTurnRestaurant(t));
        }

        // copy tables from the pre restaurant
        for (Table preTable : preRest.getActiveTables()) {
            Table newtable = new Table(preTable.getIndex(), mergedRest);
            mergedRest.addNewTable(newtable);
            newtable.setDish(preTable.getDish()); // set upstream link
            newtable.setGlobalIndex(preTable.getGlobalIndex());
            mergedRest.addCustomersToTable(newtable, preTable.getNumCustomers());
        }
        for (int inactivePreSegTableIndex : preRest.getInactiveTableIndices()) {
            mergedRest.addInactiveTableIndex(inactivePreSegTableIndex);
        }

        // copy pos tables
        for (Table posTable : posRest.getActiveTables()) {
            Table mergeTable = mergedRest.sampleTable(posTable);
            if (pathAssumption == InternalAssumption.MAXIMAL
                    || (pathAssumption == InternalAssumption.MINIMAL
                    && mergeTable.isEmpty())) {
                mergeTable.setDish(posTable.getDish());
                mergeTable.setGlobalIndex(posTable.getGlobalIndex());
            }
            mergedRest.addCustomersToTable(mergeTable, posTable.getNumCustomers());
        }

        // get pseudo new parent
        if (pathAssumption == InternalAssumption.MAXIMAL) {
            parent = curParent;
        } else {
            parent = new Restaurant(Restaurant.UNASSIGNED_REST_INDEX,
                    curParent.getParent(),
                    curParent.getHyperparameter(),
                    curParent.isTerminal(),
                    curParent.getPathAssumption());
            // create tables
            for (Table curConvTable : curParent.getActiveTables()) {
                Table newConvTable = new Table(curConvTable.getIndex(), parent);
                newConvTable.setGlobalIndex(curConvTable.getGlobalIndex());
                parent.addNewTable(newConvTable); // create conv tables
            }

            // copy customers (from neither pre nor pos restaurant) to tables
            for (Restaurant segmRest : curParent.getChildren()) {
                if (((MergeSplitRestaurant) segmRest).equals(preRest) || ((MergeSplitRestaurant) segmRest).equals(posRest)) {
                    continue;
                }
                for (Table segmTable : segmRest.getActiveTables()) {
                    Table newConvTable = parent.getTable(segmTable.getDish().getIndex());
                    parent.addCustomerToTable(newConvTable);
                }
            }
            // copy customers from merged restaurant
            for (Table table : mergedRest.getActiveTables()) {
                parent.addCustomerToTable(parent.getTable(table.getDish().getIndex()));
            }
        }
    }

    public double getNewParentLogPrior() {
        return parent.getLogPrior();
    }

    public double getCurParentLogPrior() {
        return preRest.getParent().getLogPrior();
    }

    public double getMergedRestaurantLogPrior() {
        return this.mergedRest.getLogPrior();
    }
}
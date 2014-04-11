package segmentation.nonparametric.sampler;

import java.util.*;
import segmentation.TopicSegmentation.InternalAssumption;
import util.sampling.ResizableMultinomials;

/**
 *
 * @author vietan
 */
public class Restaurant {

    public static final int UNASSIGNED_REST_INDEX = -1;
    protected int index;
    protected double hyperparameter;
    protected final boolean terminal;
    protected final InternalAssumption pathAssumption;
    protected Restaurant parent;
    protected ArrayList<Restaurant> children;
    // map from the globalIndex to the number of customer
    protected Hashtable<Integer, Integer> globalIndexCustomerCounts;
    /**
     * Store all the active tables in the restaurant. The key of the Hashtable
     * is the table index
     */
    protected Hashtable<Integer, Table> activeTables;
    /**
     * Store all inactive tables which were created but are currently empty
     */
    protected SortedSet<Integer> inactiveTables;
    protected int totalNumCustomers;

    public Restaurant(int index, Restaurant parent, double hyperparam, boolean t, InternalAssumption pathAssumption) {
        this.index = index;
        this.parent = parent;
        this.hyperparameter = hyperparam;
        this.terminal = t;
        this.pathAssumption = pathAssumption;

        if (!terminal) {
            this.children = new ArrayList<Restaurant>();
        }
        this.globalIndexCustomerCounts = new Hashtable<Integer, Integer>();

        this.activeTables = new Hashtable<Integer, Table>();
        this.inactiveTables = new TreeSet<Integer>();

        this.totalNumCustomers = 0;
    }

    public int getIndex() {
        return this.index;
    }

    public void setIndex(int i) {
        this.index = i;
    }

    /**
     * Return the count of all customers sitting at tables having globalIndex
     *
     * @param globalIndex The global index (or topic)
     * @return Number of customers
     */
    public int getCountByGlobalIndex(int globalIndex) {
        Integer count = this.globalIndexCustomerCounts.get(globalIndex);
        if (count == null) {
            count = 0;
        }
        if (this.pathAssumption == InternalAssumption.MAXIMAL
                || (this.pathAssumption == InternalAssumption.MINIMAL && this.isTerminal())) {
            return count; // number of customers equal number of tables
        } else if (this.pathAssumption == InternalAssumption.MINIMAL) {
            return count == 0 ? 0 : 1; // if exists, there's only 1 table
        } else {
            assert (false);
        }
        return -1;
    }

    /**
     * Change the count according to a global index
     *
     * @param globalIndex The global index (topic)
     * @param diff The amount to change
     */
    private void changeCountByGlobalIndex(int globalIndex, int diff) {
        Integer count = globalIndexCustomerCounts.get(globalIndex);
        if (count == null) {
            globalIndexCustomerCounts.put(globalIndex, diff);
        } else {
            int newCount = count + diff;
            if (newCount < 0) {
                throw new RuntimeException("Negative count");
            } else if (newCount == 0) {
                globalIndexCustomerCounts.remove(globalIndex);
            } else {
                globalIndexCustomerCounts.put(globalIndex, newCount);
            }
        }
    }

    /**
     * Get a list of tables having a given global index
     *
     * @param globalIndex The global index
     * @return A list of tables
     */
    private ArrayList<Table> getTables(int globalIndex) {
        ArrayList<Table> tables = new ArrayList<Table>();
        for (Table table : this.activeTables.values()) {
            if (table.getGlobalIndex() == globalIndex) {
                tables.add(table);
            }
        }
        return tables;
    }

    /**
     * Get a table given its index
     */
    public Table getTable(int tableIndex) {
        return this.activeTables.get(tableIndex);
    }

    public void addCustomersToTable(Table table, int numCusts) {
        if (!this.containsTable(table)) {
            throw new RuntimeException("The restaurant does not contain this table");
        }

        this.totalNumCustomers += numCusts;
        table.changeNumCustomers(numCusts);
        changeCountByGlobalIndex(table.getGlobalIndex(), numCusts);
    }

    /**
     * Add a customer to a table
     *
     * @param table The table to to which a customer is added
     */
    public void addCustomerToTable(Table table) {
        if (!this.containsTable(table)) {
            throw new RuntimeException("Exception while adding customer. "
                    + "The restaurant does not contain this table");
        }

        this.totalNumCustomers++;
        table.incrementNumCustomers();
        changeCountByGlobalIndex(table.getGlobalIndex(), 1);
    }

    /**
     * Remove a customer from a table
     *
     * @param table The table from which a customer is removed
     * @return The table after removal
     */
    public Table removeCustomerFromTable(Table table) {
        if (!this.containsTable(table)) {
            throw new RuntimeException("Exception while removing customer. "
                    + "The restaurant does not contain this table");
        }

        this.totalNumCustomers--;
        table.decrementNumCustomers();
        changeCountByGlobalIndex(table.getGlobalIndex(), -1);

        if (table.isEmpty()) {
            this.inactiveTables.add(table.getIndex());
            return this.activeTables.remove(table.getIndex());
        }
        return table;
    }

    /**
     * Get a new active table if a new table is sampled. If there exists some
     * inactive table the first inactive table will be returned. Otherwise, a
     * new table will be created and returned.
     *
     * @return A new active table
     */
    public Table getNewTable() {
        int newTableIndex;
        if (this.inactiveTables.isEmpty()) {
            newTableIndex = this.activeTables.size();
        } else {
            newTableIndex = this.inactiveTables.first();
            this.inactiveTables.remove(newTableIndex);
        }
        Table newTable = new Table(newTableIndex, this);
        this.activeTables.put(newTableIndex, newTable);
        return newTable;
    }

    /**
     * Add a new table to this restaurant. If the inactive table list contains
     * the id of this table, remove it.
     *
     * @param table The table to be added to the restaurant
     */
    public void addNewTable(Table table) {
        if (this.inactiveTables.contains(table.getIndex())) {
            this.inactiveTables.remove(table.getIndex());
        }
        this.activeTables.put(table.getIndex(), table);
    }

    public void addChild(Restaurant child) {
        this.children.add(child);
    }

    public boolean removeChild(Restaurant child) {
        return this.children.remove(child);
    }

    public void setParent(Restaurant p) {
        this.parent = p;
    }

    public Restaurant getParent() {
        return this.parent;
    }

    public InternalAssumption getPathAssumption() {
        return this.pathAssumption;
    }

    public double getHyperparameter() {
        return this.hyperparameter;
    }

    public void setHyperparameter(double param) {
        this.hyperparameter = param;
    }

    public SortedSet<Integer> getInactiveTableIndices() {
        return this.inactiveTables;
    }

    public void addInactiveTableIndex(int inactiveTableIndex) {
        this.inactiveTables.add(inactiveTableIndex);
    }

    public Collection<Table> getActiveTables() {
        return this.activeTables.values();
    }

    public Set<Integer> getActiveTableIndices() {
        return this.activeTables.keySet();
    }

    public int getTotalNumCustomers() {
        return this.totalNumCustomers;
    }

    public int getNumTables() {
        return this.activeTables.size();
    }

    public int getNumChildren() {
        if (isTerminal()) {
            return 0;
        }
        return this.children.size();
    }

    public boolean isTerminal() {
        return this.terminal;
    }

    public boolean isTop() {
        return this.parent == null;
    }

    public boolean containsTable(Table table) {
        return this.containsTable(table.getIndex());
    }

    public boolean containsTable(int tableIndex) {
        return this.activeTables.containsKey(tableIndex);
    }

    public ArrayList<Restaurant> getChildren() {
        return this.children;
    }

    /**
     * Return the seating assignment for a customer
     *
     * @param customer The customer to be assigned
     * @return The table that the customer will sit
     */
    public Table sampleTable(Table customer) {
        int globalIndex = customer.getGlobalIndex();
        Table table = null;

        if (this.pathAssumption == InternalAssumption.MINIMAL) {
            if (!this.globalIndexCustomerCounts.containsKey(globalIndex)) {
                table = this.getNewTable();
                table.setGlobalIndex(globalIndex);
            } else {
                ArrayList<Table> tables = getTables(globalIndex);
                if (tables.size() != 1) {
                    throw new RuntimeException("There are " + tables.size()
                            + " tables having globalIndex = " + globalIndex);
                }
                table = tables.get(0);
            }
        } else if (this.pathAssumption == InternalAssumption.MAXIMAL) {
            table = this.getNewTable();
            table.setGlobalIndex(globalIndex);
        } else {
            assert this.pathAssumption == InternalAssumption.NORMAL;


        }
        // in general case without any assumption on the internal path,
        // the table will be sampled according to the CRP

        return table;
    }

    /**
     * Compute the probability of assigning a given word to a table of this
     * restaurant
     *
     * @param tableIndex The table index
     * @param wordIndex The index (in the vocab) of the given word
     * @param topicWord The topic-word multinomials
     */
    protected double getProbAssignTable(int tableIndex, int wordIndex, ResizableMultinomials topicWord) {
        double prob = 0;
        if (this.containsTable(tableIndex)) {
            Table table = this.getTable(tableIndex);
            prob = (table.getNumCustomers() * topicWord.getLikelihood(table.getGlobalIndex(), wordIndex))
                    / (this.totalNumCustomers + this.hyperparameter);
        } else {
            double newTableProb = 0;
            if (this.parent == null) {
                newTableProb = 1.0 / topicWord.getDimension();
            } else {
                for (Table parentTable : this.parent.getActiveTables()) {
                    newTableProb += this.parent.getProbAssignTable(parentTable.getIndex(), wordIndex, topicWord);
                }
                newTableProb += this.parent.getProbAssignTable(Table.EMPTY_TABLE_INDEX, wordIndex, topicWord);
            }
            prob = (this.hyperparameter * newTableProb)
                    / (this.totalNumCustomers + this.hyperparameter);
        }
        return prob;
    }

    /**
     * Get the prior distribution of this restaurant This formula can be
     * rewritten to use the Stirling's approximation of log gamma
     */
    public double getLogPrior() {
        int J = this.getNumTables();
        double samplingLogValue = J * Math.log(this.hyperparameter);
        for (Table table : this.activeTables.values()) {
            for (int n = 1; n < table.getNumCustomers(); n++) {
                samplingLogValue += Math.log(n);
            }
        }
        for (int x = 1; x <= this.totalNumCustomers; x++) {
            samplingLogValue -= Math.log(x - 1 + this.hyperparameter);
        }
        return samplingLogValue;
    }

    /**
     * Get the prior distribution given the hyperparameter
     */
    public double getLogPrior(double testHyperparameter) {
        int J = this.getNumTables();
        double samplingLogValue = J * Math.log(testHyperparameter);
        for (Table table : this.activeTables.values()) {
            for (int n = 1; n < table.getNumCustomers(); n++) {
                samplingLogValue += Math.log(n);
            }
        }
        for (int x = 1; x <= this.totalNumCustomers; x++) {
            samplingLogValue -= Math.log(x - 1 + testHyperparameter);
        }
        return samplingLogValue;
    }

    @Override
    public int hashCode() {
        return Integer.valueOf(index).hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if ((obj == null) || (this.getClass() != obj.getClass())) {
            return false;
        }
        Restaurant r = (Restaurant) (obj);

        if (this.parent == null) {
            return this.index == r.index;
        } else {
            return this.index == r.index && this.parent.equals(r.parent);
        }
    }

    public void validate() {
        // number of cusomters must match
        int totalCustomerCount = 0;
        for (Table table : this.activeTables.values()) {
            totalCustomerCount += table.getNumCustomers();
        }
        if (totalCustomerCount != this.totalNumCustomers) {
            throw new RuntimeException("Mismatch # customers: " + totalCustomerCount + " vs " + this.totalNumCustomers);
        }

        int totalCustomerCountFromGlobaIndex = 0;
        for (int c : this.globalIndexCustomerCounts.values()) {
            totalCustomerCountFromGlobaIndex += c;
        }
        if (totalCustomerCountFromGlobaIndex != this.totalNumCustomers) {
            throw new RuntimeException("Mismatch # customers: " + totalCustomerCountFromGlobaIndex + " vs " + this.totalNumCustomers);
        }

        // number of global indices
        Set<Integer> globalIndices = new HashSet<Integer>();
        for (Table table : activeTables.values()) {
            globalIndices.add(table.getGlobalIndex());
        }
        if (globalIndices.size() != globalIndexCustomerCounts.size()) {
            throw new RuntimeException("Mismatch # global indices: " + globalIndices.size() + " vs " + globalIndexCustomerCounts.size());
        }

        if (this.pathAssumption == InternalAssumption.MINIMAL) {
            if (globalIndices.size() != this.getNumTables()) {
                throw new RuntimeException("MINIMAL: # global indices = " + globalIndices.size() + " != # tables = " + this.getNumTables());
            }
        } else if (this.pathAssumption == InternalAssumption.MAXIMAL) {
            if (this.getNumTables() != this.totalNumCustomers) {
                throw new RuntimeException("MAXIMAL: # tables = " + this.getNumTables() + " != # customers = " + this.totalNumCustomers);
            }
        }

        // assert global indices
        for (Table table : this.getActiveTables()) {
            if (table.getGlobalIndex() == Table.UNASSIGNED_GLOBAL_INDEX) {
                throw new RuntimeException("Unassigned global index for table " + table.getIndex());
            } else if (this.parent != null && table.getGlobalIndex() != table.getDish().getGlobalIndex()) {
                throw new RuntimeException("Mismatch global indices: " + table.getGlobalIndex() + " vs " + table.getDish().getGlobalIndex());
            }
        }

        // check unassigned index
        if (this.index == UNASSIGNED_REST_INDEX) {
            throw new RuntimeException("Unassigned index restaurant");
        }
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("Restaurant: ").append(index).append(" (").append(hyperparameter).append(") ").append(": ");
        // tables
        str.append("# tables = ").append(this.getNumTables());
        for (Table table : getActiveTables()) {
            str.append(" ").append(table.getIndex()).append(",").
                    append(table.getGlobalIndex()).append("(#").append(table.getNumCustomers()).append(")");
        }
        str.append("\t");

        // global indices
        str.append("# global indices = ").append(this.globalIndexCustomerCounts.size());
        for (int globalIndex : this.globalIndexCustomerCounts.keySet()) {
            str.append(" ").append(globalIndex).append("(").append(this.globalIndexCustomerCounts.get(globalIndex)).append(")");
        }
        str.append(". # children = ").append(getNumChildren());
        return str.toString();
    }
}

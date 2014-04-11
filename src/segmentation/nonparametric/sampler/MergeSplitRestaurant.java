package segmentation.nonparametric.sampler;

import java.util.Collection;
import java.util.Hashtable;
import java.util.Set;
import segmentation.TopicSegmentation.InternalAssumption;

/**
 *
 * @author vietan
 */
public class MergeSplitRestaurant extends Restaurant {

    private Hashtable<Integer, Restaurant> turnRestaurants;

    public MergeSplitRestaurant(int index, Restaurant parent, double hyperparam,
            boolean t, InternalAssumption pathAssumption) {
        super(index, parent, hyperparam, t, pathAssumption);
        this.index = index;
        this.turnRestaurants = new Hashtable<Integer, Restaurant>();
    }

    public boolean containsTurn(int t) {
        return this.turnRestaurants.containsKey(t);
    }

    public void addTurnRestaurant(int t, Restaurant turnRest) {
        this.turnRestaurants.put(t, turnRest);
    }

    public Restaurant getTurnRestaurant(int t) {
        return this.turnRestaurants.get(t);
    }

    public Set<Integer> getTurnRestaurantIndices() {
        return this.turnRestaurants.keySet();
    }

    public Collection<Restaurant> getTurnRestaurants() {
        return this.turnRestaurants.values();
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
        MergeSplitRestaurant r = (MergeSplitRestaurant) (obj);

        if (this.getParent() == null) {
            return this.index == r.index;
        } else {
            return this.index == r.index && this.parent.equals(r.parent);
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
        str.append("\t");

        // turns
        str.append(this.turnRestaurants.keySet().toString());
        return str.toString();
    }
}

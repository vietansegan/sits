package segmentation.nonparametric.sampler;

/**
 *
 * @author vietan
 */
public class Table implements Comparable<Table> {

    public static final int UNASSIGNED_GLOBAL_INDEX = -1;
    public static final int EMPTY_TABLE_INDEX = -1;
    private final int index;
    private int numCustomers;
    private Table dish;
    private int globalIndex;
    private Restaurant restaurant;

    public Table(int index, Restaurant rest) {
        this.index = index;
        this.restaurant = rest;
        this.globalIndex = UNASSIGNED_GLOBAL_INDEX;
    }

    public int getIndex() {
        return this.index;
    }

    public boolean isEmpty() {
        return this.numCustomers == 0;
    }

    public Table getDish() {
        return dish;
    }

    public void setDish(Table dish) {
        this.dish = dish;
    }

    public int getGlobalIndex() {
        return globalIndex;
    }

    public void setGlobalIndex(int globalIndex) {
        this.globalIndex = globalIndex;
    }

    public int getNumCustomers() {
        return numCustomers;
    }

    public void incrementNumCustomers() {
        this.changeNumCustomers(1);
    }

    public void decrementNumCustomers() {
        this.changeNumCustomers(-1);
    }

    public void changeNumCustomers(int diff) {
        this.numCustomers += diff;
    }

    public Restaurant getRestaurant() {
        return restaurant;
    }

    public void setRestaurant(Restaurant restaurant) {
        this.restaurant = restaurant;
    }

    @Override
    public int compareTo(Table t) {
        return -Double.compare(this.getNumCustomers(), t.getNumCustomers());
    }

    @Override
    public int hashCode() {
        return Integer.valueOf(index + restaurant.hashCode()).hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if ((obj == null) || (this.getClass() != obj.getClass())) {
            return false;
        }
        Table t = (Table) (obj);

        return this.index == t.index && this.restaurant.equals(t.getRestaurant());
    }
}

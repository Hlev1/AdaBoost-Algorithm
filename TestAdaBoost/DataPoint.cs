using System;
namespace AdaBoost
{
    // A class to store an individual data sample in the full data set, so that
    // we can assign a weight to each sample, and update this weight as we
    // advance in the algorithm.
    public class DataPoint {

        private double weight;
        private Object[] x;
        private int y;
        private double e = Math.E;


        public DataPoint(Object[] x, int y, int n) {
            this.x = x;
            this.y = y;
            this.weight = 1.0f / n;
        }

        // After using this constructor must manually set the weight of each data point
        public DataPoint(Object[] x, int y) {
            this.x = x;
            this.y = y;
        }

        public double getWeight() {
            return weight;
        }

        public void updateWeight(int y, double alpha, int classified) {
            weight = (weight * (Math.Pow(e, (-y * alpha * classified))));
        }

        public void setWeight(double weight) {
            this.weight = weight;
        }

        public int getY() {
            return y;
        }

        public Object[] getX() {
            return x;
        }

        public String toString() {
            return "x = " + x.ToString() + " weight = " + weight;
        }

    }
}

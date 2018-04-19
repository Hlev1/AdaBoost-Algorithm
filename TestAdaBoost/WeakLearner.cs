using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace AdaBoost
{   // Implementation of a decision stump.
    public class WeakLearner {

        private double alpha;
        private int splitFeature; // The index of the split feature
        private Object splitValue;
        // Data to help me when testing
        private String[] splits = { "action_type", "combined_shot_type", "game_event_id", "game_id", "lat", "loc_x", "loc_y", "lon", "minutes_remaining", "period", "playoffs", "season", "seconds_remaining", "shot_distance", "shot_type", "shot_zone_area", "shot_zone_basic", "shot_zone_range", "team_name", "game_date", "matchup", "opponent", "shot_id" };

        /// <summary>
        /// Constructor finds the splitValue which 
        /// </summary>
        /// <param name="splitFeature">Split feature.</param>
        /// <param name="trainData">Train data.</param>
        public WeakLearner(int splitFeature, DataPoint[] trainData) {
            List<List<Object>> possibleSplits = new List<List<Object>>();
            // This is an integer which indexes the item in the array of data which this stump will split on.
            // It is the value which the data is compared with when deciding which path down the tree the data belongs to (whether to output 1 or -1).
            this.splitFeature = splitFeature;

            foreach (DataPoint dataP in trainData) {
                Object value = dataP.getX()[splitFeature];
                bool foundValue = false;
                foreach (List<Object> list in possibleSplits) {
                    if (list[0].Equals(value)) {
                        foundValue = true;
                        list[1] = Convert.ToInt32(list[1].ToString()) + dataP.getY();
                    }
                }
                if (!foundValue) {
                    possibleSplits.Add(new List<object>() { value, dataP.getY() });
                }
            }

            // Then set the splitValue to the feature with the lowest entropy
            possibleSplits = possibleSplits.OrderByDescending(lst => lst[1]).ToList();
            splitValue = possibleSplits[0][0];
        }
        /// <summary>
        /// Calculates the weighted sum error of the data when tested on this single weak learner
        /// </summary>
        /// <returns>The weighted sum error</returns>
        /// <param name="trainData">The data to train</param>
        public double runTest(DataPoint[] trainData) {
            double sumError = 0.0f;
            int individualError;
            foreach (DataPoint testCase in trainData) {
                individualError = this.classify(testCase.getX()) == testCase.getY() ? 0 : 1;
                sumError += (testCase.getWeight() * individualError);
            }

            return sumError;
        }
        /// <summary>
        /// Set the alpha of the weaklearner given the weighted sum error of the training set
        /// </summary>
        /// <param name="Et">Et.</param>
        public void setAlpha(double Et) {
            alpha = (float)((0.5) * Math.Log((1 - Et) / Et));
        }
        /// <summary>
        /// This is the method which is used when testing on new data.
        /// </summary>
        /// <returns>The classification</returns>
        /// <param name="xs">the data which is being classified, this represents only one datapoint</param>
        public int classify(Object[] xs) {
            if (xs[splitFeature].Equals(splitValue))
                return 1;
            else
                return -1;
        }

        public double getAlpha() {
            return alpha;
        }

        public int getSplitFeature() {
            return splitFeature;
        }
        /// <summary>
        /// To help when testing
        /// </summary>
        /// <returns>The string</returns>
        public String toString() {
            return "wl: " + splits[splitFeature] + " val: " + splitValue;
        }

    }
}

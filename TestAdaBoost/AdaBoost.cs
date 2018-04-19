using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
namespace AdaBoost {
    public class AdaBoost {

        // T is the hyperparameter of the algorithm which will determine how many iterations the algorithm completed.
        private static int T = 15;
        DataPoint[] trainData; // The test data that is being used.
        public WeakLearner[] finalLearners = new WeakLearner[T];
        private bool testDataAdded = false;

        /// <summary>
        /// Method to be run during each iteration of the algorithm. This method creates all possible weak learners, and then chooses the one which
        /// reduces the total weighted error on the set by the most.
        /// </summary>
        /// <param name="algorithmIt"> The current iteration which the algorithm is in </param>
        private void initLearners(int algorithmIt) {
            Debug.Assert(testDataAdded == true); // Method shouldnt be called before the test data has been added.
            // An arrayList to hold the WeakLearner's created at each iteration
            ArrayList tempLearners = new ArrayList();
            int featuresToSplit = trainData[0].getX().Length;
            // This loop will produce multiple weak learners for this iteration of the algorithm. Which we will then use to find the weak learner which minimizes the
            // total weighted error on the set.
            double minSumError = float.MaxValue;
            WeakLearner l;
            for (int i = 0; i < featuresToSplit; i++) {

                tempLearners.Add(l = new WeakLearner(i, trainData));
                double err = l.runTest(trainData);
                if (err < minSumError) {
                    minSumError = err;
                    finalLearners[algorithmIt] = l;
                }
            }

            // Then set the alpha value for the newly generated weak learner.
            finalLearners[algorithmIt].setAlpha(minSumError);
            Console.WriteLine("WeakLearner selected ( " + finalLearners[algorithmIt].toString() + " )");

            // After we have chosen the weak learner which reduces the weighted sum error by the most, we need to update the weights of each data point.
            double sumWeights = 0.0f; // This is our normalisation value so we can normalise the weights after we have finished updating them/
            foreach (DataPoint dataP in trainData) {
                int y = dataP.getY();
                Object[] x = dataP.getX();
                // Classify the data input using the weak learner. Then check to see if this classification is correct/incorrect and adjust the weights accordingly.
                int classified = finalLearners[algorithmIt].classify(x);
                dataP.updateWeight(y, finalLearners[algorithmIt].getAlpha(), classified);
                sumWeights += dataP.getWeight();

            }
            double weightTot = 0.0f;
            // Normalise the weights for all data points by dividing by the total of the summed weights.
            foreach (DataPoint dataP in trainData) {
                dataP.setWeight(dataP.getWeight() / sumWeights);
                weightTot += dataP.getWeight();
            }
            Console.WriteLine("The total of all the weights of the datapoints is = " + weightTot);
        }


        /// <summary>
        /// Method used for the final classification of data after the system has been learned on the test data.
        /// </summary>
        /// <returns>The classification of the input data, either 1 or -1, since we are doing binary classification. </returns>
        /// <param name="data"> The data which is being classified. </param>
        public int classify(Object[] data) {
            Debug.Assert(finalLearners.Length != 0);
            double classification = 0.0f;
            foreach (WeakLearner learner in finalLearners) {
                classification += (learner.classify(data) * (learner.getAlpha())); // Get the value from the weak learner and apply the weight to the output.
            }

            int classified = (classification >= 0.0f) ? 1 : -1;
            return classified;
        }

        /// <summary>
        /// Sets the test data to be used to learn the system.
        /// </summary>
        /// <param name="entry"> The test data which is being used to learn the system. </param>
        public void setTrainData(DataPoint[] entry) {
            Debug.Assert(entry != null);
            this.trainData = entry;
            testDataAdded = true;

        }

        public void learn() {
            Debug.Assert(testDataAdded == true);
            for (int i = 0; i < T; i++) {
                initLearners(i);
            }
            foreach (WeakLearner wl in finalLearners) {
                Console.WriteLine(wl.toString());
            }
        }

    }
}

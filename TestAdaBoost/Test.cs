using System;
using System.Collections;
using System.IO;
using AdaBoost;

namespace AdaBoost
{
    class Test {

        private ArrayList trainingData = new ArrayList();
        private DataPoint[] trainingDataArray; 
        private ArrayList testData = new ArrayList();
        private Object[][] testDataArray;
        private int sizeOfTrainArray = 0;
        private int sizeOfTestArray = 0;

        private static String inputFile = "/Users/harrylevick/Documents/GitHub/AdaBoost-Algorithm/usedFiles/data.csv";
        private static String outputFile = "/Users/harrylevick/Documents/GitHub/AdaBoost-Algorithm/usedFiles/output.csv";
        private static String markFile = "/Users/harrylevick/Documents/GitHub/AdaBoost-Algorithm/usedFiles/ActualOutcomes.csv";

        static void Main(string[] args) {
            AdaBoost ada = new AdaBoost();
            Test testAda = new Test();
            testAda.parseDataFromFile(inputFile);
            testAda.setWeights(testAda.trainingDataArray.Length, testAda.trainingDataArray);
            ada.setTrainData(testAda.trainingDataArray);
            ada.learn();

            using (var w = new StreamWriter(outputFile)) {

                var firstline = string.Format("{0},{1}", "shot_id", "shot_made_flag");
                w.WriteLine(firstline);
                w.Flush();

                foreach (Object[] entity in testAda.testDataArray) {

                    int shtMade = ada.classify(entity) == 1 ? 1 : 0;
                    var line = string.Format("{0},{1}", entity[21], shtMade);
                    w.WriteLine(line);
                    w.Flush();
                }
                
            }
            Console.WriteLine(testAda.generatePassPercentage(outputFile, markFile));


        }

        private double generatePassPercentage(String outPath, String markPath) {
            int correctClassifications = 0;
            int incorrectClassifications = 0;
            using (StreamReader sr1 = new StreamReader(outPath)) {
                using (StreamReader sr2 = new StreamReader(markPath)) {
                    String currentLine1 = sr1.ReadLine();
                    String currentLine2 = sr2.ReadLine();
                    while ((currentLine1 = sr1.ReadLine()) != null && (currentLine2 = sr2.ReadLine()) != null) {
                        var values1 = currentLine1.Split(',');
                        var values2 = currentLine2.Split(',');
                        if (values1[1].Equals(values2[1])) {
                            correctClassifications += 1;
                        } else incorrectClassifications += 1;
                    }
                }
            }
            int totalClassifications = correctClassifications + incorrectClassifications;
            return ((double) correctClassifications / ((double) totalClassifications));
        }

        /// <summary>
        /// A method to take the data in the input csv file, and turn this into useable data, 
        /// in the form that our AdaBoost object can use.
        /// </summary>
        /// <param name="path"> The path of the input file </param>
        public void parseDataFromFile(String path) {
            using (StreamReader sr = new StreamReader(path)) {
                String currentLine = sr.ReadLine();
                while ((currentLine = sr.ReadLine()) != null) {
                    Object[] eachEntity = new Object[22];
                    var values = currentLine.Split(',');
                    Console.WriteLine("");

                    if (!values[14].Equals("")) { // Check the data is training data (that the entry at index 14 - the y value - is given)
                        int position = 0;
                        for (int i = 0; i < values.Length; i++) {
                            if (i != 14) {
                                eachEntity[position] = values[i];
                                position++;
                            }
                        }
                        int outcome = (values[14].Equals("1") ? 1 : -1);
                        trainingData.Add(new DataPoint(eachEntity, outcome));

                        sizeOfTrainArray++;

                    } else {
                        int position = 0;
                        for (int i = 0; i < values.Length; i++) {
                            if (i != 14) {
                                eachEntity[position] = values[i];
                                position++;
                            }
                        }
                        testData.Add(eachEntity);
                        sizeOfTestArray++;
                    }
                }
                int p = 0;
                trainingDataArray = new DataPoint[sizeOfTrainArray];
                foreach (DataPoint entity in trainingData) {
                    trainingDataArray[p] = entity;
                    p++;
                }
                int k = 0;
                testDataArray = new Object[sizeOfTestArray][];
                foreach (Object[] entity in testData) {
                    testDataArray[k] = entity;
                    k++;
                }
            }
        }

        /// <summary>
        /// Since, when we parse the data from the input file, we dont know exactly how many data sets there are, we cannot initialise each individual weight of each 
        /// data point. Insead we have to wait until we know the total number of data points that we are using, then we set the weight of each data point.
        /// </summary>
        /// <param name="n"> The number of data points </param>
        /// <param name="trainArray"> The array which holds all of the DataPoint's to be used.</param>
        public void setWeights(int n, DataPoint[] trainArray) {
            foreach (DataPoint dataP in trainArray) {
                dataP.setWeight(1.0f / n);
            }
        }
    }
}

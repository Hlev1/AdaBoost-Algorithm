# AdaBoost-Challenge

I have implemented the AdaBoost Boosting algorithm using a series of decision stumps (decision trees of height h = 1).
### A brief explanation of how my implementation works:
In this implementation I have used 3 classes (AdaBoost, WeakLearner and DataPoint).
##### AdaBoost 
This is the main class from which the structure of the algorithm can be seen. The class contains a variable, T, which determines how many iterations the algorithm will compute, and hence how many WeakLearners the algorithm will create. At each iteration of the algorithm, a weak learner is created for each possible feature in the data set. I will later explain the data set used in my testing. Hence if there are 5 different features for each data entry in the set, the algorithm will create 5 temporary weak learners. Then it will select the WeakLearner which reduces the weighted sum error by the most. And discard the other 4 WeakLearner's. The alpha for this WeakLearner is then assigned, using the setAlpha() method, which takes in as a parameter the weighted sum error produced by the WeakLearner. Next the weights for each individual DataPoint is adjusted, according to how the newly created WL classifies this DataPoint. The weights for all these DataPoint's are then normalised so that they sum to 1.
This process is repeated for the number of WeakLearners that are to be created. Then when we want to classify new data, we use each WL to create a summed classification value, weighted with the WL's Alpha. Then we take the sign of this summed classification to be the final classification.

##### WeakLearner
This class represents a decision tree of height 1 (decision stump). The constructor takes in the whole training set and finds the value of the feature-to-split-on which reduces the error the most. I do this using the entropy reduction rule. Here, the total entropy for each split is found, hence we then choose the split which reduces the entropy by the most.

##### DataPoint
I decided to create a separate object for each individual data entry because this allows me to easily assign a weight to each individual entry. Hence, in my testing, I initially convert the .csv file into an array of type DataPoint so that my algorithm can work with the data.

### Test Data Set
The test data that I have used is about the NBA player Kobe Bryant. I found this data online at Kaggle.com (https://www.kaggle.com/c/kobe-bryant-shot-selection). This contains information about each individual field goal (shot) that Bryant ever took in his entire career. Containing information such as opponent, position on court, type of shot, year, minutes remaining in the game etc. The majority of entries also contains a value {-1, +1} to represent whether Bryant made or missed this shot. The role of the classifier is then to predict whether Bryant made the shots given in the test set (the test set is included in the training set, but just doesn't have a value for shot_made_flag, so my program extracts this and used this separately from the training data).
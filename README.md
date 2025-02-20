# Module 21 Report - Neural Network Model
# Deep Learning Challenge

## Overview:
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received access to a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

* EIN and NAME — Identification columns
* APPLICATION_TYPE — Alphabet Soup application type
* AFFILIATION — Affiliated sector of industry
* CLASSIFICATION — Government organization classification
* USE_CASE — Use case for funding
* ORGANIZATION — Organization type
* STATUS — Active status
* INCOME_AMT — Income classification
* SPECIAL_CONSIDERATIONS — Special considerations for application
* ASK_AMT — Funding amount requested
* IS_SUCCESSFUL — Was the money used effectively

## Instructions:
### Preprocess the Data
1) From the provided cloud URL, read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:

* What variable(s) are the target(s) for your model?
* What variable(s) are the feature(s) for your model?

2) Drop the EIN and NAME columns.
3) Determine the number of unique values for each column.
4) For columns that have more than 10 unique values, determine the number of data points for each unique value.
5) Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other, and then check if the replacement was successful.
6) Use pd.get_dummies() to encode categorical variables.
7) Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
8) Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

### Compile, Train and Evaluate the Model
1) Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
2) Create the first hidden layer and choose an appropriate activation function.
3) If necessary, add a second hidden layer with an appropriate activation function.
4) Create an output layer with an appropriate activation function.
5) Check the structure of the model.
6) Compile and train the model.
7) Create a callback that saves the model's weights every five epochs
* Performed on Original Run Only

8) Evaluate the model using the test data to determine the loss and accuracy.
9) Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

## Results:
# Original Run
Evaluating the Model:

![Original](https://github.com/mlbybee/deep-learning-challenge/blob/main/Resources/Original_definingmodel.png)
![First_Run](https://github.com/mlbybee/deep-learning-challenge/blob/main/Resources/First_Run.png)

Were you able to achieve the target model performance? NO

## Optimize the Model
Using TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
* Dropping more or fewer columns.
* Creating more bins for rare occurrences in columns.
* Increasing or decreasing the number of values for each bin.
* Add more neurons to a hidden layer.
* Add more hidden layers.
* Use different activation functions for the hidden layers.
* Add or reduce the number of epochs to the training regimen.

# Optimization 1
Data Preprocessing:
1) Removed unnecessary columns ('EIN', 'Name', 'Special Considerations', 'Ask Amt')
2) Adjusted cutoff value for classifications to 1883 from 777
3) Reduced the number of neurons
    - Layer 1 = 20
    - Layer 2 = 10
* Reduce the risk of overfitting

Evaluating the Model:

![Opt1](https://github.com/mlbybee/deep-learning-challenge/blob/main/Resources/Optimization1_definingmodel.png)
![Second_Run](https://github.com/mlbybee/deep-learning-challenge/blob/main/Resources/Second_Run.png)

Were you able to achieve the target model performance? NO

# Optimization 2
Data Preprocessing:
1) Removed unnecessary column ('Name', 'EIN', 'Special Considerations', 'Ask Amt', 'Status')
2) Reduced cutoff value for application type to 725
3) Went back to original number of neurons
    - Layer 1 = 80
    - Layer 2 = 30
4) Changed the first layer activation to ('tanh')
    - The tanh function introduces non-linearity into the model, allowing the network to learn complex patterns in the data. 

Evaluating the Model:

![Opt2](https://github.com/mlbybee/deep-learning-challenge/blob/main/Resources/Optimization2_definingmodel.png)
![Third_Run](https://github.com/mlbybee/deep-learning-challenge/blob/main/Resources/Third_Run.png)

Were you able to achieve the target model performance? NO

# Optimization 3
Data Preprocessing:
1) Removed unnecessary columns ('Special Considerations', 'EIN', 'Ask Amt')
* Kept 'Name' column' - Wanted to test keeping one of the identification columns
3) Number of neurons
    - Layer 1 = 80
    - Layer 2 = 30
4) Reverted back the first layer activation being ('ReLu')
* Since ReLu involves simple thresholding at zero, it allows for faster computations compared to other activation functions like sigmoid or tanh, which require more complex calculations.

Evaluating the Model:

![Opt3](https://github.com/mlbybee/deep-learning-challenge/blob/main/Resources/Optimization3_definingmodel.png)
![Fourth_Run](https://github.com/mlbybee/deep-learning-challenge/blob/main/Resources/Fourth_Run.png)

Were you able to achieve the target model performance? YES

# Summary
The final optimized neural network trained model from the keras tuner method achieved 78% prediction accuracy with a 48% loss. Keeping the 'Name' column was crucial in achieving and going beyond the target. This shows the importance of the shape of your datasets before you preprocess it.

Additional Recommendations for Optimization: 

* Experimenting with different batch sizes and learning rates.
* Adjusting dropout rates to prevent overfitting.
* Exploring different weight initializers and loss functions.
* Testing alternative optimizers such as Adam, RMSprop, or Nadam to find the best fit for the dataset.
  
## Support Received:
* Stack Overflow
* Xpert Learning Assistant Chat+
* Class Lecture Materials

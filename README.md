We first clone and install the required packages to make sure of reproducibility of the code

We first clone and install the required packages to make sure of reproducibility of the code

We import the necessary packges that we will use throught this notebook

We import the necessary packges that we will use throught this notebook

# Data Loading

We load the data from github using wget into colab then into our notebook 

# Data Loading

We load the data from github using wget into colab then into our notebook 

We do a few data exploratory analysis to know the type of data and find if there any NANs and what reltions we can come up with

We do a few data exploratory analysis to know the type of data and find if there any NANs and what reltions we can come up with

We find if there was any missing values

We find if there was any missing values

We drop the z columns since it would not be any benefical to us and find the new shape 

We drop the z columns since it would not be any benefical to us and find the new shape 

We find if the name of the lables and find if the counts of each of them to dtermine if it was a balanced problem or not 

We find if the name of the lables and find if the counts of each of them to dtermine if it was a balanced problem or not 

We discovered that it is a bit imbalanced dataset, but we can deal with it as it is

We discovered that it is a bit imbalanced dataset, but we can deal with it as it is

# Data Visualization

Now we do a bit of data visualization to discover our data and how it looks like

# Data Visualization

Now we do a bit of data visualization to discover our data and how it looks like

First we visualize a single sample from the palm label to see how it looks like and create a connection dictinary to draw the hand skeleton

First we visualize a single sample from the palm label to see how it looks like and create a connection dictinary to draw the hand skeleton

Now we generalize the graph to be able to visualize all the different labels

Now we generalize the graph to be able to visualize all the different labels

Now we try to visualize and find if there was any correlation between the set of points

We do it by calculating the euclidean distances as a part of feature engineering two combine two points and the default point is the palm with index 1

Now we try to visualize and find if there was any correlation between the set of points

We do it by calculating the euclidean distances as a part of feature engineering two combine two points and the default point is the palm with index 1

We also try to find the correlation between all the point and the tip of the middle index finger

We also try to find the correlation between all the point and the tip of the middle index finger

From both correlations, we can see that some points have a high correlations above the 0.98, which can be utilized to help in the training of the model

From both correlations, we can see that some points have a high correlations above the 0.98, which can be utilized to help in the training of the model

# Data Preprocessing

We do now a small task of data preprocessing where we first normalize the coordinates by image resolution, after, we recenter the data along the palm coordinates (x1,y1) and then we scale it by calculating the euclidean distance to the middle index finger tip (x13,y13)

Also, here we divide the dataset into a train set and a test set, with a seed of 42 to ensure the reproducibility.

# Data Preprocessing

We do now a small task of data preprocessing where we first normalize the coordinates by image resolution, after, we recenter the data along the palm coordinates (x1,y1) and then we scale it by calculating the euclidean distance to the middle index finger tip (x13,y13)

Also, here we divide the dataset into a train set and a test set, with a seed of 42 to ensure the reproducibility.

We first do a label encoding to the features to be able to pass them as the labels for the machine learning model

We first do a label encoding to the features to be able to pass them as the labels for the machine learning model

To verify that our preprocessing was correct, we will revisualize the graph to make sure that nothing has changed and it is the same

To verify that our preprocessing was correct, we will revisualize the graph to make sure that nothing has changed and it is the same

We will also redraw the correlation matrix to know how the correlation was changed between the features and whether we can come up with useful relations

We will also redraw the correlation matrix to know how the correlation was changed between the features and whether we can come up with useful relations

As it is shown, some features have a correlation of around 0.99 and other are around 0.98. So, we will take them into consideration.

We will train 4 differants variants of the dataset, which are:
1- raw dataset, which will be trained without any preprocessing
2- preprocessed dataset, which will be trained after doing preprocessing only
3- preprocessed palm dataset, which will be trained after doing preprocessing and then dropping the heighly correlated columns according to the palm (x0,y0) 
4- preprocessed index dataset, which will be trained after doing preprocessing and then dropping the heighly correlated columns according to the middle index finger tip (x13,y13)


To ensure unification between those different datasets, we will use the same indicies that willbe generated by train_test_split to have those different datasets

As it is shown, some features have a correlation of around 0.99 and other are around 0.98. So, we will take them into consideration.

We will train 4 differants variants of the dataset, which are:
1- raw dataset, which will be trained without any preprocessing
2- preprocessed dataset, which will be trained after doing preprocessing only
3- preprocessed palm dataset, which will be trained after doing preprocessing and then dropping the heighly correlated columns according to the palm (x0,y0) 
4- preprocessed index dataset, which will be trained after doing preprocessing and then dropping the heighly correlated columns according to the middle index finger tip (x13,y13)


To ensure unification between those different datasets, we will use the same indicies that willbe generated by train_test_split to have those different datasets

To make sure that there is no data leakage, we will first calculate the correlation matrix to the train set, and from it, we will drop the same columns for both the train and test sets and then we will draw the correlation matrix for the training set.

To make sure that there is no data leakage, we will first calculate the correlation matrix to the train set, and from it, we will drop the same columns for both the train and test sets and then we will draw the correlation matrix for the training set.

As shown, some parameters have been dropped, so in the palm variant, there is 10 columns dropped, and the index variant  columns dropped which will reduce the curse of dimensionality

As shown, some parameters have been dropped, so in the palm variant, there is 10 columns dropped, and the index variant  columns dropped which will reduce the curse of dimensionality

# Model Training and Evaluation

We will train 5 different classifiers which are:

1- Support Vector Machine (SVM)

2- Decision Tree

3- Random forest

4- Logistic Regression

5- XGBoost

Those classifiers are trained in a function called tune_model_with_performance, this function will take the train and test sets, along with a string called variant_name to help us identify which type of dataset was entered for the model, and the model_type that is default to SVM, to help identify which model to train, and finally a bool variable called use_GPU and it was in case for XGBoost, and a seed variable.

After specifying the model_type, we perform a random search to search for the best hyperparameter the model could have, after that we refine it a bit by extracting the best 3 hyperparameters and generating a boundary between them, and we feed this boundry to grid search to try and come up with the best possible combination of hyperparameters.

Finally, we will train a classifier with the best possible combination and generate our model,and visualizing the confusion matrix of our model.

After training the classifiers, we will evaluate their performance according to different criterias, mainly the accuracy, along with precision, recall, and F1-score. Also we will look at there training time, inference time, memory usage and CPU usage. Which will help us to evaluate the different models.

Also, we will use MLflow to make the expirements reproducible, and to be able to log the different paramters in case we needed to reproduce the expirement.

# Model Training and Evaluation

We will train 5 different classifiers which are:

1- Support Vector Machine (SVM)

2- Decision Tree

3- Random forest

4- Logistic Regression

5- XGBoost

Those classifiers are trained in a function called tune_model_with_performance, this function will take the train and test sets, along with a string called variant_name to help us identify which type of dataset was entered for the model, and the model_type that is default to SVM, to help identify which model to train, and finally a bool variable called use_GPU and it was in case for XGBoost, and a seed variable.

After specifying the model_type, we perform a random search to search for the best hyperparameter the model could have, after that we refine it a bit by extracting the best 3 hyperparameters and generating a boundary between them, and we feed this boundry to grid search to try and come up with the best possible combination of hyperparameters.

Finally, we will train a classifier with the best possible combination and generate our model,and visualizing the confusion matrix of our model.

After training the classifiers, we will evaluate their performance according to different criterias, mainly the accuracy, along with precision, recall, and F1-score. Also we will look at there training time, inference time, memory usage and CPU usage. Which will help us to evaluate the different models.

Also, we will use MLflow to make the expirements reproducible, and to be able to log the different paramters in case we needed to reproduce the expirement.

Those are the helper functions that will be used with our main function, they have a purpose of calculating the memory and the CPU usages, along with the inference time.

The refine_bounds function has a purpose to helping in reducing the grid of searching for the grid search

Also we will save the generated model, along with the output of the function in a directory called 'outputs/' to make it easier to retrieve a certain model and/or its performance metrics in case we need to do further analysis in the future like we retrieved the model for the MediaPipe phase, we save the model in .pkl format and the performance metrics, best hyperparameters and accuracy in .pkl, .json and .txt format, and this in case different people from different domains are going to work on it.

Those are the helper functions that will be used with our main function, they have a purpose of calculating the memory and the CPU usages, along with the inference time.

The refine_bounds function has a purpose to helping in reducing the grid of searching for the grid search

Also we will save the generated model, along with the output of the function in a directory called 'outputs/' to make it easier to retrieve a certain model and/or its performance metrics in case we need to do further analysis in the future like we retrieved the model for the MediaPipe phase, we save the model in .pkl format and the performance metrics, best hyperparameters and accuracy in .pkl, .json and .txt format, and this in case different people from different domains are going to work on it.

### SVM
We genetrate the first model of SVM classifiers

### SVM
We genetrate the first model of SVM classifiers

### DecisionTree
We genetrate the second model of Decision Tree classifiers, we built here only a single tree to make the decision

### DecisionTree
We genetrate the second model of Decision Tree classifiers, we built here only a single tree to make the decision

### RandomForest
We genetrate the third model of Random Forest classifiers

### RandomForest
We genetrate the third model of Random Forest classifiers

### LogisticRegression
We genetrate the forth model of Logistic Regression classifiers

### LogisticRegression
We genetrate the forth model of Logistic Regression classifiers

### XGBoost
We genetrate the fifth model of XGBoost classifiers

### XGBoost
We genetrate the fifth model of XGBoost classifiers

# Conclusion

We finish our project by doing a quick comaprison of the different metrics and what can we deduce from them

Those are our functions that we are going to use throught this conclusion

# Conclusion

We finish our project by doing a quick comaprison of the different metrics and what can we deduce from them

Those are our functions that we are going to use throught this conclusion

First, we visualize all the different perfromance metrics in a dataframe, sorted by the accuracy column in descending order

First, we visualize all the different perfromance metrics in a dataframe, sorted by the accuracy column in descending order

### Accuracy
We first visualisze the accuracy of the different models in different graphs to analyse it.

### Accuracy
We first visualisze the accuracy of the different models in different graphs to analyse it.

From these graphs, the best model using accuracy as a criteria is XGBoost with preprocessed dataset, however, we have to note that the accuracy of the SVM model with preprocessed dataset is quite similar with XGBoost with a small difference in accuracy

Overall, when we take the aveage of all different dataset variants, the SVM model is the best, and comes after it directly the XGBoost, while if we take the aveage of all different models for each dataset variants, the preprocessed variant achived the heighest accuracy, and comes after it the preprocessed palm variant, and shortly the third preprocessed index variant, and the raw data achived the worst accuracy. Also this shows that we can replace the whole preprocessed dataset with one that had dropped various columns like when we did with the preprocessed palm variant, to reduce the curse of dimensionality and speed up the training process

From these graphs, the best model using accuracy as a criteria is XGBoost with preprocessed dataset, however, we have to note that the accuracy of the SVM model with preprocessed dataset is quite similar with XGBoost with a small difference in accuracy

Overall, when we take the aveage of all different dataset variants, the SVM model is the best, and comes after it directly the XGBoost, while if we take the aveage of all different models for each dataset variants, the preprocessed variant achived the heighest accuracy, and comes after it the preprocessed palm variant, and shortly the third preprocessed index variant, and the raw data achived the worst accuracy. Also this shows that we can replace the whole preprocessed dataset with one that had dropped various columns like when we did with the preprocessed palm variant, to reduce the curse of dimensionality and speed up the training process

### F1-score
We also visualisze the F1-score of the different models because the dataset is a bit imbalanced to know about the performanceof the models.

### F1-score
We also visualisze the F1-score of the different models because the dataset is a bit imbalanced to know about the performanceof the models.

It seems that there is no significant difference between using the accuracy score or the f1-score, because it has yielded the same observations.

It seems that there is no significant difference between using the accuracy score or the f1-score, because it has yielded the same observations.

### Inference Time	
We now see how long does it usually take a mdel to infer a new data point in order to make prediction

### Inference Time	
We now see how long does it usually take a mdel to infer a new data point in order to make prediction

From the graphs, the SVM models has the biggest inference time, and it can reach to near 2 seconds in order to classify a data point, which makes it unsuitable for any application that requires live feedback. While XGBoost is intermediate value taking around 87 milliseconds, and finally the Logistic Regression and the Decision Tree are almost instantaneous which makes them suitable to any application that requires a moderate accuracy but must have a fast decision

While, the classification by vaiant, they have almost similar time for all the models, except at the SVM, where the raw dataset has the lowest inference time in comparison to the other variants, and at random forest it has the highest one 

From the graphs, the SVM models has the biggest inference time, and it can reach to near 2 seconds in order to classify a data point, which makes it unsuitable for any application that requires live feedback. While XGBoost is intermediate value taking around 87 milliseconds, and finally the Logistic Regression and the Decision Tree are almost instantaneous which makes them suitable to any application that requires a moderate accuracy but must have a fast decision

While, the classification by vaiant, they have almost similar time for all the models, except at the SVM, where the raw dataset has the lowest inference time in comparison to the other variants, and at random forest it has the highest one 

### CPU usage	
We now see how much CPU usage does it usually take a mdel in order to infer a new data point in order to make prediction

### CPU usage	
We now see how much CPU usage does it usually take a mdel in order to infer a new data point in order to make prediction

Here the XGBoost has one of the highest CPU usage, which make it consumes a lot of CPU, while the SVM has the lowest CPU usage, which makes it a great candidate for the application where the CPU is limited like embedded systems.Also in most variants of random forest it has a low CPU usage, same goes for logstic regression.

For the different variants of the datasets, it seems that the raw is the one that consumes less than the other types, whule the palm has the highest avergae CPU usage accross the 4 variants

Here the XGBoost has one of the highest CPU usage, which make it consumes a lot of CPU, while the SVM has the lowest CPU usage, which makes it a great candidate for the application where the CPU is limited like embedded systems.Also in most variants of random forest it has a low CPU usage, same goes for logstic regression.

For the different variants of the datasets, it seems that the raw is the one that consumes less than the other types, whule the palm has the highest avergae CPU usage accross the 4 variants

## Conclusion
After analyzing the results, we selected XGBoost (preprocessed) as the best model to use with our MediaPipe gesture prediction system, primarily due to its highest accuracy and F1-score. While it comes with a higher CPU usage (20.3%), it significantly outperforms other models in inference speed, achieving predictions in around 87 milliseconds, making it ideal for real-time classification.

Although SVM (preprocessed) showed similar performance in terms of accuracy and F1-score, its inference time exceeded 2 seconds, making it less suitable for live predictions. However, if CPU efficiency were the main priority, SVM (preprocessed) would have been a better choice, consuming only 1.3% CPU compared to XGBoost's usage.

For embedded systems, we recommend RandomForest (preprocessed). It offers a strong balance between performance and efficiency, with a moderate CPU usage (0.6%), inference time around 207 milliseconds, and high accuracy (0.97). It serves as a middle ground between the performance of XGBoost and the efficiency of SVM.

In a web-based application, we suggest using XGBoost (palm), which provides near-identical accuracy to XGBoost (preprocessed) but with reduced CPU usage (17.8%) and a slightly faster inference time 80 milliseconds, making it more suitable for scalable, cloud-hosted environments.

Finally, for quick testing or exploratory data analysis, DecisionTree (preprocessed) is recommended. It trains in seconds and delivers an accuracy of around 0.94, making it ideal for demonstration or prototyping purposes.

## Conclusion
After analyzing the results, we selected XGBoost (preprocessed) as the best model to use with our MediaPipe gesture prediction system, primarily due to its highest accuracy and F1-score. While it comes with a higher CPU usage (20.3%), it significantly outperforms other models in inference speed, achieving predictions in around 87 milliseconds, making it ideal for real-time classification.

Although SVM (preprocessed) showed similar performance in terms of accuracy and F1-score, its inference time exceeded 2 seconds, making it less suitable for live predictions. However, if CPU efficiency were the main priority, SVM (preprocessed) would have been a better choice, consuming only 1.3% CPU compared to XGBoost's usage.

For embedded systems, we recommend RandomForest (preprocessed). It offers a strong balance between performance and efficiency, with a moderate CPU usage (0.6%), inference time around 207 milliseconds, and high accuracy (0.97). It serves as a middle ground between the performance of XGBoost and the efficiency of SVM.

In a web-based application, we suggest using XGBoost (palm), which provides near-identical accuracy to XGBoost (preprocessed) but with reduced CPU usage (17.8%) and a slightly faster inference time 80 milliseconds, making it more suitable for scalable, cloud-hosted environments.

Finally, for quick testing or exploratory data analysis, DecisionTree (preprocessed) is recommended. It trains in seconds and delivers an accuracy of around 0.94, making it ideal for demonstration or prototyping purposes.

# MediaPipe

We merge the XGBoost model by the MediaPipe, to have our final model, we use opencv to load the video, and then we deal with it using Mediapipe

# MediaPipe

We merge the XGBoost model by the MediaPipe, to have our final model, we use opencv to load the video, and then we deal with it using Mediapipe


# Human Activity Recognition Predictions
Matt Narrell  
May 23, 2015  


# Synopsis
This discussion is a prediction of how well participants performed exercise activities measured by personal collectors such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit*.  From the [Human Activity Recognition study](http://groupware.les.inf.puc-rio.br/har) [1]

> "Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E)."
  
We will use machine learning algorithms (specifically random forest) to predict the "class" of activity given the accumulated data.

# Getting and Cleaning the Data
The following function will download the training and test data sets from the indicated URLs, if necessary.  There are several expressions of NA that are considered when loading the raw data into data frames.  Next, the first seven columns are discarded from the data frames as they will not contribute to the ML models.  Only the activity measurements are of importance.  Lastly, since the machine learning algorithms are sensitive to missing data, those variables that are exclusively missing are discarded from the data frame.

```r
load.data <- function(src_location) {
  # Create data directory if necessary
  if (!file.exists("./data")) { dir.create("./data") }
  
  # Fetch data from source location if necessary
  filename <- tail(unlist(strsplit(src_location, "/")), n = 1)
  filePath <- paste("./data", filename, sep = "/")
  if (!file.exists(filePath)) {
    download.file(src_location, filePath, method = "curl")
  }
  
  # Read the CSV data with the given NA values, and omit the first seven variables 
  # (they will not contribute to the models)
  dataFrame <- read.csv(filePath, na.strings = c("NA", "", "#DIV/0!"))
  dataFrame <- dataFrame[, -(1:7)]
  
  # Drop the variables that are all NA values
  na.sums <- sapply(dataFrame, function(x) { sum(is.na(x)) })
  dataFrame <- dataFrame[, which(na.sums == 0)]
  dataFrame
} 

trainingData <- load.data("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
testingData <- load.data("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
```

# Cross Validation Set
Here we create a cross validation data set from 25% of the training data to estimate the out of sample error before we predict against the testing data.  Ideally we'd do this multiple times and average the error estimates but this leads to very long execution times.

```r
library(caret)
set.seed(13131313)
training.indexes <- createDataPartition(trainingData$classe, p = 0.75, list = FALSE)
training <- trainingData[training.indexes,]
cross.validation <- trainingData[-training.indexes,]
```

# Modeling
## Model fitting
The random forest ML algorithm was chosen for its high degree of accuracy, out of the box.  Due to the high potential of overfitting, we must cross validate to remedy this.  This leads to very long execution times.

```r
# http://topepo.github.io/caret/training.html
train.control <- trainControl(method="cv", number = 4, savePredictions = TRUE, allowParallel = TRUE)
system.time(random.forest <- train(classe ~ ., method = "rf", data = training, trControl = train.control))
```

Characteristics of our final model

```r
random.forest$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.6%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4179    3    2    0    1 0.001433692
## B   17 2827    4    0    0 0.007373596
## C    0   17 2542    8    0 0.009738995
## D    0    0   23 2388    1 0.009950249
## E    0    1    4    8 2693 0.004804139
```
As shown above, the estimated error rate of this model is **0.6%**, which is extremely accurate.

## Model evaluation
Below we make predictions on the training set and cross validation set with our newly fitted random forest model.  The accuracy on the training set is extremely high as expected and we'll compare that with the cross validation set to estimate the out of sample error rate.

```r
training.predictions <- predict(random.forest, training)
confusionMatrix(training.predictions, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
validation.predictions <- predict(random.forest, cross.validation)
confusionMatrix(validation.predictions, cross.validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1391   10    0    0    0
##          B    4  934    3    2    0
##          C    0    5  845    7    1
##          D    0    0    7  795    3
##          E    0    0    0    0  897
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9914          
##                  95% CI : (0.9884, 0.9938)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9892          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9971   0.9842   0.9883   0.9888   0.9956
## Specificity            0.9972   0.9977   0.9968   0.9976   1.0000
## Pos Pred Value         0.9929   0.9905   0.9848   0.9876   1.0000
## Neg Pred Value         0.9989   0.9962   0.9975   0.9978   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2836   0.1905   0.1723   0.1621   0.1829
## Detection Prevalence   0.2857   0.1923   0.1750   0.1642   0.1829
## Balanced Accuracy      0.9971   0.9910   0.9925   0.9932   0.9978
```
The cross validation confusion matrix shows the model accuracy to be **99%**; quite suitable for predicting our test set outcomes, and validates the estimated error rate from above.
The *roll_belt* variable is by far the most important predictor of how well a subject performed his or her exercises.

```r
varImp(random.forest)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                      Overall
## roll_belt             100.00
## pitch_forearm          58.12
## yaw_belt               55.25
## pitch_belt             45.00
## magnet_dumbbell_z      44.65
## magnet_dumbbell_y      43.05
## roll_forearm           39.45
## accel_dumbbell_y       22.48
## magnet_dumbbell_x      18.64
## roll_dumbbell          18.35
## accel_forearm_x        16.15
## magnet_belt_z          15.38
## accel_dumbbell_z       15.30
## accel_belt_z           13.86
## magnet_belt_y          13.29
## magnet_forearm_z       12.14
## total_accel_dumbbell   11.94
## gyros_belt_z           11.09
## yaw_arm                10.72
## magnet_belt_x          10.46
```

# Test Set Predictions

```r
testing.predictions <- predict(random.forest, testingData)
testing.predictions
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

# Submission

```r
dir <- "./submission"
if (!file.exists(dir)) { dir.create(dir) }

pml_write_files = function(x) {
  n = length(x)
  for (i in 1:n) {
    filename = paste0(dir, "/", "problem_id_", i, ".txt")
    write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
  }
}

pml_write_files(as.vector(testing.predictions))
```

# References
1. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. [Qualitative Activity Recognition of Weight Lifting Exercises](http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201). Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.


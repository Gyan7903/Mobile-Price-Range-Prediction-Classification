# Mobile-Price-Range-Prediction-Classification

## <u>**Project Summary** <u>
### **This data science project aimed to predict mobile phone price ranges based on their specifications using machine learning algorithms. The dataset consisted of various features related to mobile phones such as battery capacity, RAM, internal memory, camera quality, and other hardware specifications. The dataset was split into training and testing sets and several machine learning algorithms such as Logistic Regression, Decision Tree, Random Forest, and Support Vector Machines (SVM) were applied to the training set. The Random Forest algorithm was chosen as the final model due to its high accuracy and F1-score. The project concluded that predictive modeling can be an effective approach for mobile price range detection and highlighted the importance of data preprocessing and feature engineering for improving model accuracy.**

## <u>**Problem Statement**<u>
### **In the competitive mobile phone market companies want to understand sales data of mobile phones and factors which drive the prices.The objective is to find out some relation between features of a mobile phone(eg:- RAM, Internal Memory, etc) and its selling price. In this problem, we do not have to predict the actual price but a price range indicating how high the price is.**

## **<u>Data Description<u>**
* **Battery_power** - Total energy a battery can store in one time measured in mAh
* **Blue** - Has bluetooth or not
* ***Clock_speed*** - speed at which microprocessor executes instructions
* ***Dual_sim*** - Has dual sim support or not
* ***Fc*** - Front Camera mega pixels
* ***Four_g*** - Has 4G or not
* ***Int_memory*** - Internal Memory in Gigabytes
* ***M_dep*** - Mobile Depth in cm
* ***Mobile_wt*** - Weight of mobile phone
* ***N_cores*** - Number of cores of processor
* ***Pc*** - Primary Camera mega pixels
* ***Px_height*** - Pixel Resolution Height
* ***Px_width*** - Pixel Resolution Width
* ***Ram*** - Random Access Memory in Mega Bytes
* ***Sc_h*** - Screen Height of mobile in cm
* ***Sc_w*** - Screen Width of mobile in cm
* ***Talk_time*** - longest time that a single battery charge will last when you are
* ***Three_g*** - Has 3G or not
* ***Touch_screen*** - Has touch screen or not
* ***Wifi*** - Has wifi or not
* ***Price_range*** - This is the target variable with value of
* 0(low cost),
* 1(medium cost),
* 2(high cost) and
* 3(very high cost).
* Thus our target variable has 4 categories so basically it is a Multiclass classification problem.

* # Predictive Modeling:
Algorithms used for predictive modeling:
* 1) Decision Tree
* 2) Random Forest classifier
* 3) Gradient Boosting Classifier
* 4) K-nearest Neighbour classifier
* 5) XG Boost Classifier
* 6) Support Vector Machine(SVM)

# <u>***Final Observations From Above Predictive Modeling</u>***

## **1) <u>Observations of Decision Tree Classifier:</u>**
### <u>Before Hyperparameter Tuning:</u>


*   training accuarcy = 100%
*   test accuarcy = 84%

Model is overfitted the data and does not generalised well. So we tuned the hyperparameters.

### <u>After Hyperparameter tuning:</u>


*   Training accuarcy= 98%
*   Test accuarcy = 91%

However this will not be good model for us. RAM,battery power,px_height and width came out to be the most important featrures. This model classified the class 0 and class 3 very nicely as we can see the AUC is almost 0.96 for both classes,whereas for class 1 and class 2 it is 0.88.

## **2) <u>Observations of Random Forest:**
### Before Hyperparameter Tuning:</u>

* training accuarcy = 100%
* test accuarcy = 88%

Model is overfitted the data and does not generalised well. So we tuned the hyperparameters.

### <u>After Hyperparameter tuning:</u>

* Training accuarcy= 100%
* Test accuarcy = 90%

We have slightly improved the model and overfitting is reduced slightly. From roc curve its clear that model has poorly performed to classify class 1 and class 2

## **3) <u>Observations of Gradient Boost Classifiers:**
### Before Hyperparameter tunning:</u>

* Train accuracy score= 100%.
* Test accuracy score= 89%

Model did not generalised well and overfitted the training data. so we tuned hyperparameters of model.

### <u>After Hyperparameter Tuning:</u>

* Train accuracy score= 100%
* Test accuarcy score=90%

Thus we slightly improved the model performance.However the model is not best. From ROC curve it's clear that model was good to classify the class 0 and class 3.From the classification report its clear that recall for class 0 and class 3 is also good which is 96% and 90% respectively.

## **4) <u>Observations of K Nearest Neighbors:**
### Before hyperparameters tuning:</u>

* Train Accuracy:75 %
* Test Accuarcy:59 %

Clearly Model has performed very worst. We did hyperparameter tuning

### <u>After Hyperparameter Tuning:</u>

* Train Accuarcy: 77%
* Test Accuarcy: 70%

Surely we improved the model perfromance and reduced overfitting but however this is not good model for us.

## **5) <u>Observations of XGBoost Classifier:**
### Before Hyperparameter Tuning:</u>

* Train Accuarcy = 98%
* Test Accuarcy = 90%

### <u>After Hyperparameter Tuning:</u>

* Train Accuarcy = 100%
* Test Accuarcy = 92%

we have improved the model performance by Hyperparamter tuning. Test accuracy is increased to 92%.But still the difference of accuracy score between train and test is more than 5%.We can say model is very slightly overfitted From AUC-ROC curve its clear that model has almost correctly predicted the class 0 and class 3.

## **6) <u>Observations of SVM:**
### Before Hyperparameter Tuning:</u>

* Train Accuarcy = 98.5%
* Test Accuarcy = 89%

### <u>After Hyperparameter Tuning:</u>

* Train Accuarcy = 98.3%
* Test Accuarcy = 97%

SVM performed very well as compared to other alogorithms. In terms of feature importance RAM,Battery power,px_height and px_weight are the imporatant features. f1 score for individual classes is also very good. Area under curve for each class prediction is also almost 1.

# <u>***Conclusions:***</u>
* We Started with Data understanding, data wrangling, basic EDA where we found the relationships, trends between price range and other independent variables.
* We selected the best features for predictive modeling by using K best feature selection method using Chi square statistic.
* Implemented various classification algorithms, out of which the SVM(Support vector machine) algorithm gave the best performance after hyper-parameter tuning with 98.3% train accuracy and 97 % test accuracy.
* XG boost is the second best good model which gave good performance after hyper-parameter tuning with 100% train accuracy and 92.25% test accuracy score.
* KNN gave very worst model performance.
* We checked for the feature importance's of each model. RAM, Battery Power, Px_height and px_width contributed the most while predicting the price range.

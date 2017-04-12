# Fraud Detection


The dataset comes from e-commerce company and contains two tables. The goal is to predict fraud transaction based on user profile and browsing acitivities.

The first table has 2 million rows, and contains information like userid (unique for each row),signup date, purchase date, purchase amount, deviceid (code of characters),ip address, user gender, user age, and label for fraud. So this is a supervised learning task. The second table is a lookup table for ipaddress (upper bound,lower bound) and countries.

### Feature engineering
To create new features, I used ipaddress to create country column. The ip address is not a continuous spectrum for each country, so I have to look up for the country for each ip address. There are hundreds of countries in the table, I only keep the top 50 countries that have most fraud activities. Dummy variables are created for each country.

A lot of work has to do with signup and purchase time. It is straightforward to get the time difference between purchase time and signup time. Intuitively, this would be an important feature. Then, I tried to extract the week of the year, day of the week and the hour of the day for signup and purchase,respectively. I plotted them as is, and found the counts of activities are almost uniformly distributed for all hours, no matter fraud or not. That does not sound right because why should someone signup during the mid of night. So I thought maybe the timestamp is the server time stamp that does not account for user's local time. So I tried to get timezone information from ip address to adjust for GMT offset. Long story short, that does not seem to help. For millons of ip address, it is not easy to query every ip address exactly. I used average GMT offset for every country. 

I also group the ip address and device id respectively to count the times the ip address and the device id were used.

For logistic regression, I also did feature scaling, and tried to create 2nd order polynomial features for numerical columns and using correlation matrix to filter the highly correlated features. 

### Classification Algorithm
I tried scikit learn logistic regression with l1,l2 regulization, random forest, gradient descent with hinge loss (linear SVM), adaboost, and gradient boost. Since two classes are not balanced (fraud:no = 1:9), I also tried to over sample using SMOTE. 

Random forest performs very well without much tuning, and set a benchmark of 0.99 auc-roc,evaluated with held out test set . With SMOTE oversampling, 20 trees are sufficient to achieve the benchmark. The second one is gradient boost, 0.94 aur-roc. Though it takes much longer to train the model. The linear model logistic regression (with and without polynomial features) and gradient descend with hinge loss perform the worst.

Since there are a lot of categorical features as a result of creating dummy variables for day of week and countries etc., tree base algorithm suits well for this kind of task. Given the 2 millions row of data, random forest also performs relatively faster than the boosting algorithm.







---
title: 'Titanic: Machine Learning from Disaster -- a Kaggle competition entry'
author: "Sizhan Shi"
date: "November 5, 2014"
output: 
    html_document:
        theme: journal
        highlight: kate

---

## Background  
This is an entry for the kaggle machine learning competition [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic-gettingStarted)

#### Introduction from kaggle's website:  
> The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.  

> One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.  

> In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.  

> This Kaggle "Getting Started" Competition provides an ideal starting place for people who may not have a lot of experience in data science and machine learning. The data is highly structured, and we provide tutorials of increasing complexity for using Excel, Python, pandas in Python, and a Random Forest in Python (see links in the sidebar). We also have links to tutorials using R instead. Please use the forums freely and as much as you like. There is no such thing as a stupid question; we guarantee someone else will be wondering the same thing!  

#### Method:  
1. Load the data  
2. Perform exploratory analysis  
3. Propose hypothesis  
4. Create model reflective of the hypothesis  
5. Submit model prediction result  
6. Repeat steps 3 through 5 until reaching the limit of accuracy improvement  

#### Reference:  
http://trevorstephens.com/post/72920580937/titanic-getting-started-with-r-part-2-the

## Summary  






## Detailed analysis  

#### Load the data

```r
train <- read.csv("train.csv")
test <- read.csv("test.csv")
head(train)
```

```
##   PassengerId Survived Pclass
## 1           1        0      3
## 2           2        1      1
## 3           3        1      3
## 4           4        1      1
## 5           5        0      3
## 6           6        0      3
##                                                  Name    Sex Age SibSp
## 1                             Braund, Mr. Owen Harris   male  22     1
## 2 Cumings, Mrs. John Bradley (Florence Briggs Thayer) female  38     1
## 3                              Heikkinen, Miss. Laina female  26     0
## 4        Futrelle, Mrs. Jacques Heath (Lily May Peel) female  35     1
## 5                            Allen, Mr. William Henry   male  35     0
## 6                                    Moran, Mr. James   male  NA     0
##   Parch           Ticket    Fare Cabin Embarked
## 1     0        A/5 21171  7.2500              S
## 2     0         PC 17599 71.2833   C85        C
## 3     0 STON/O2. 3101282  7.9250              S
## 4     0           113803 53.1000  C123        S
## 5     0           373450  8.0500              S
## 6     0           330877  8.4583              Q
```

```r
## how many observations 
nrow(train)
```

```
## [1] 891
```

```r
## how many features
ncol(train)
```

```
## [1] 12
```

```r
## how many survived and their percentage
nrow(train[train$Survived==1,]) / nrow(train)
```

```
## [1] 0.3838384
```

About 38.4% of passengers survived. Let's examine survival rate in relationship to passengers' demographic attributes. First, start with age and survival rate.  

```r
library(ggplot2)
ggplot(train, aes(as.factor(Survived),Age)) + geom_boxplot() + theme_bw()
```

```
## Warning: Removed 177 rows containing non-finite values (stat_boxplot).
```

![plot of chunk unnamed-chunk-2](figure/unnamed-chunk-2-1.png) 

From a galance, there doens't seem to have a strong difference in age between survivors and victims. Let's look at sex next.  

### Gender Model:  


```r
table(train$Sex)
```

```
## 
## female   male 
##    314    577
```

```r
table(train$Sex, train$Survived)  
```

```
##         
##            0   1
##   female  81 233
##   male   468 109
```

```r
##  seems like females are more likely to sruvive, let's look at the proportions:
prop.table(table(train$Sex, train$Survived), 1)
```

```
##         
##                  0         1
##   female 0.2579618 0.7420382
##   male   0.8110919 0.1889081
```
74% of females survived, versus only 19% for males.  

#### Round 1 submission:  
Assume all females survived, and all males died.  

#### Round 1 submission result:  
This model had an accuracy of 0.76555.  

### Gender, Class, Fare, and Children Model:  


Now let's get back to the ages.  

```r
summary(train$Age)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
##    0.42   20.12   28.00   29.70   38.00   80.00     177
```

```r
## there are several NAs. Let's clean them up by replacing them with the average age of all passengers  
yesAge <- train[!is.na(train$Age),]
train[is.na(train$Age),"Age"] <- mean(yesAge$Age)
```
Here, the assumption is that women and children had priority access to lifeboats and life vests. The question is how young should a child be to be considered eligible for that treatment? Let's use a for loop, starting from the age 18, and compare the difference of the survival rates between boys and girls, if we found an age group with almost equal survival rate between both genders, we have found the threadshold.  

**Update:** This finding was later confirmed by my reserach on [Wikipedia](http://en.wikipedia.org/wiki/RMS_Titanic#Survivors_and_victims). According to the table, there were 109 children in total, since we only have a portion of the data (train), we can calculate the ratio of observation data in the training set versus the complete dataset.  


```r
rate <- nrow(train) / (nrow(train) + nrow(test))
```

We have about 68% of all observations, meaning the number of children in our training dataset should follow the same ratio.  


```r
rate * 109
```

```
## [1] 74.19328
```

We should have about 74 children in our training dataset. Let's verify that with our 15 thredshold.  


```r
nrow(train[train$Age < 15,])
```

```
## [1] 78
```
78, very close to the number 74. 15 is the correct threshold. According to the table, 100% of children stayed in second class survived. We will apply this learning later.  

**Update over**  


```r
for (i in 2:18) {
    ageTemp <- subset(train, Age < i)
    prop.table(table(ageTemp$Sex, ageTemp$Survived),1)
    diff <- prop.table(table(ageTemp$Sex, ageTemp$Survived),1)[1,2]-prop.table(table(ageTemp$Sex, ageTemp$Survived),1)[2,2]
    print(paste("Age", i, "survival diff: ", diff))
}
```

```
## [1] "Age 2 survival diff:  0.2"
## [1] "Age 3 survival diff:  -0.0428571428571429"
## [1] "Age 4 survival diff:  -0.138888888888889"
## [1] "Age 5 survival diff:  0.0537084398976982"
## [1] "Age 6 survival diff:  0.109730848861284"
## [1] "Age 7 survival diff:  0.072463768115942"
## [1] "Age 8 survival diff:  0.134615384615385"
## [1] "Age 9 survival diff:  0.123626373626374"
## [1] "Age 10 survival diff:  0.0395833333333333"
## [1] "Age 11 survival diff:  0.0371456500488758"
## [1] "Age 12 survival diff:  0.0381944444444444"
## [1] "Age 13 survival diff:  0.0261824324324325"
## [1] "Age 14 survival diff:  0.0500794912559619"
## [1] "Age 15 survival diff:  0.076923076923077"
## [1] "Age 16 survival diff:  0.126162790697674"
## [1] "Age 17 survival diff:  0.242096838735494"
## [1] "Age 18 survival diff:  0.29435736677116"
```
It seems like age 15 is the threashold -- younger than 15, regardless boy or girl, the person was considered a child and given priority; older than 15, the person was considered an adult, a woman or a man, and was treated accordingly. 


```r
age15 <- subset(train, Age < 15)
table(age15$Sex, age15$Survived)
```

```
##         
##           0  1
##   female 15 24
##   male   18 21
```

```r
prop.table(table(age15$Sex, age15$Survived), 1)
```

```
##         
##                  0         1
##   female 0.3846154 0.6153846
##   male   0.4615385 0.5384615
```
Children younger than 15 had a survival rate higher than 54%, much better than the average of 38%. Let's assign this level to the passengers.  


```r
train$Child <- 0
train[train$Age < 15,]$Child <- 1
```

#### Learnings so far:  
1. Females had higher survival rate than men;  
2. Children under the age of 15 were treated equally regardless of gender;  

Now take another look at the categorization of fare and class. First, find the most common fare price.  


```r
for (i in 10:100) {
    propotion <- length(train[train$Fare<i,]$Fare) / nrow(train)
    msg <- paste("Fare lower than ", i, " account for ", round(propotion, digits=2))
    print(msg)
}
```

```
## [1] "Fare lower than  10  account for  0.38"
## [1] "Fare lower than  11  account for  0.41"
## [1] "Fare lower than  12  account for  0.42"
## [1] "Fare lower than  13  account for  0.43"
## [1] "Fare lower than  14  account for  0.49"
## [1] "Fare lower than  15  account for  0.51"
## [1] "Fare lower than  16  account for  0.54"
## [1] "Fare lower than  17  account for  0.56"
## [1] "Fare lower than  18  account for  0.56"
## [1] "Fare lower than  19  account for  0.57"
## [1] "Fare lower than  20  account for  0.58"
## [1] "Fare lower than  21  account for  0.59"
## [1] "Fare lower than  22  account for  0.6"
## [1] "Fare lower than  23  account for  0.6"
## [1] "Fare lower than  24  account for  0.61"
## [1] "Fare lower than  25  account for  0.63"
## [1] "Fare lower than  26  account for  0.63"
## [1] "Fare lower than  27  account for  0.7"
## [1] "Fare lower than  28  account for  0.72"
## [1] "Fare lower than  29  account for  0.72"
## [1] "Fare lower than  30  account for  0.73"
## [1] "Fare lower than  31  account for  0.75"
## [1] "Fare lower than  32  account for  0.76"
## [1] "Fare lower than  33  account for  0.77"
## [1] "Fare lower than  34  account for  0.77"
## [1] "Fare lower than  35  account for  0.78"
## [1] "Fare lower than  36  account for  0.78"
## [1] "Fare lower than  37  account for  0.78"
## [1] "Fare lower than  38  account for  0.79"
## [1] "Fare lower than  39  account for  0.79"
## [1] "Fare lower than  40  account for  0.8"
## [1] "Fare lower than  41  account for  0.8"
## [1] "Fare lower than  42  account for  0.81"
## [1] "Fare lower than  43  account for  0.81"
## [1] "Fare lower than  44  account for  0.81"
## [1] "Fare lower than  45  account for  0.81"
## [1] "Fare lower than  46  account for  0.81"
## [1] "Fare lower than  47  account for  0.81"
## [1] "Fare lower than  48  account for  0.82"
## [1] "Fare lower than  49  account for  0.82"
## [1] "Fare lower than  50  account for  0.82"
## [1] "Fare lower than  51  account for  0.82"
## [1] "Fare lower than  52  account for  0.82"
## [1] "Fare lower than  53  account for  0.84"
## [1] "Fare lower than  54  account for  0.84"
## [1] "Fare lower than  55  account for  0.84"
## [1] "Fare lower than  56  account for  0.85"
## [1] "Fare lower than  57  account for  0.86"
## [1] "Fare lower than  58  account for  0.86"
## [1] "Fare lower than  59  account for  0.86"
## [1] "Fare lower than  60  account for  0.86"
## [1] "Fare lower than  61  account for  0.86"
## [1] "Fare lower than  62  account for  0.87"
## [1] "Fare lower than  63  account for  0.87"
## [1] "Fare lower than  64  account for  0.87"
## [1] "Fare lower than  65  account for  0.87"
## [1] "Fare lower than  66  account for  0.87"
## [1] "Fare lower than  67  account for  0.87"
## [1] "Fare lower than  68  account for  0.87"
## [1] "Fare lower than  69  account for  0.87"
## [1] "Fare lower than  70  account for  0.88"
## [1] "Fare lower than  71  account for  0.88"
## [1] "Fare lower than  72  account for  0.89"
## [1] "Fare lower than  73  account for  0.89"
## [1] "Fare lower than  74  account for  0.89"
## [1] "Fare lower than  75  account for  0.89"
## [1] "Fare lower than  76  account for  0.89"
## [1] "Fare lower than  77  account for  0.9"
## [1] "Fare lower than  78  account for  0.9"
## [1] "Fare lower than  79  account for  0.91"
## [1] "Fare lower than  80  account for  0.91"
## [1] "Fare lower than  81  account for  0.92"
## [1] "Fare lower than  82  account for  0.92"
## [1] "Fare lower than  83  account for  0.92"
## [1] "Fare lower than  84  account for  0.93"
## [1] "Fare lower than  85  account for  0.93"
## [1] "Fare lower than  86  account for  0.93"
## [1] "Fare lower than  87  account for  0.93"
## [1] "Fare lower than  88  account for  0.93"
## [1] "Fare lower than  89  account for  0.93"
## [1] "Fare lower than  90  account for  0.93"
## [1] "Fare lower than  91  account for  0.94"
## [1] "Fare lower than  92  account for  0.94"
## [1] "Fare lower than  93  account for  0.94"
## [1] "Fare lower than  94  account for  0.94"
## [1] "Fare lower than  95  account for  0.94"
## [1] "Fare lower than  96  account for  0.94"
## [1] "Fare lower than  97  account for  0.94"
## [1] "Fare lower than  98  account for  0.94"
## [1] "Fare lower than  99  account for  0.94"
## [1] "Fare lower than  100  account for  0.94"
```

We can categorize fare into one of the following buckets:  
1. < 10, about 38% passengers;  
2. >=10 and <15, about 13% passengers;  
3. >=15 and <24, about 10% passengers;  
4. >=24 and <28, about 11% passengers;  
5. >=28 and <52, about 10% passengers;  
6. >=52 and <78, about 8% passengers;  
7. >=78, the rest of the 10% passengers;  

Now let's assign a new class to the train dataset.  


```r
train$Fare2 <- "78+"
train[train$Fare<78 & train$Fare>=52,]$Fare2 <- "52-78"
train[train$Fare<52 & train$Fare>=28,]$Fare2 <- "28-52"
train[train$Fare<28 & train$Fare>=24,]$Fare2 <- "24-28"
train[train$Fare<24 & train$Fare>=15,]$Fare2 <- "15-24"
train[train$Fare<15 & train$Fare>=10,]$Fare2 <- "10-15"
train[train$Fare<10,]$Fare2 <- "<10"
```

Let's look at the distribution of passengers across gender, class, and fare level:  


```r
survSumFareClassSex <- aggregate(Survived ~ Fare2 + Pclass + Sex + Child, data=train, length)
colnames(survSumFareClassSex)[5] <- "Total.Size"
```

Now let's look at their survival rate.  


```r
survRate <- function(x) {
    round(sum(x) / length(x),digits=2)
    
}
survRateFareClassSex <- aggregate(Survived ~ Fare2 + Pclass + Sex + Child, data=train, survRate)
colnames(survRateFareClassSex)[5] <- "Survival.Rate"
survRateFareClassSexChildFull <- merge(survRateFareClassSex, survSumFareClassSex)
survRateFareClassSexChildFull[order(-survRateFareClassSexChildFull$Survival.Rate, -survRateFareClassSexChildFull$Total.Size),]
```

```
##    Fare2 Pclass    Sex Child Survival.Rate Total.Size
## 42 52-78      1 female     0          1.00         24
## 34 28-52      2 female     0          1.00          6
## 22 24-28      1 female     0          1.00          5
## 25 24-28      2 female     1          1.00          4
## 35 28-52      2 female     1          1.00          4
## 37 28-52      2   male     1          1.00          4
## 13 10-15      3   male     1          1.00          3
## 51   78+      1   male     1          1.00          3
## 15 15-24      2 female     1          1.00          2
## 17 15-24      2   male     1          1.00          2
## 27 24-28      2   male     1          1.00          2
## 44 52-78      2 female     0          1.00          2
## 6    <10      3   male     1          1.00          1
## 9  10-15      2   male     1          1.00          1
## 48   78+      1 female     0          0.98         54
## 7  10-15      2 female     0          0.90         30
## 14 15-24      2 female     0          0.90         10
## 24 24-28      2 female     0          0.89         18
## 32 28-52      1 female     0          0.89          9
## 11 10-15      3 female     1          0.71          7
## 19 15-24      3 female     1          0.67          9
## 21 15-24      3   male     1          0.67          6
## 18 15-24      3 female     0          0.64         25
## 3    <10      3 female     0          0.60         62
## 4    <10      3 female     1          0.50          2
## 49   78+      1 female     1          0.50          2
## 47 52-78      3   male     0          0.45         11
## 23 24-28      1   male     0          0.42         26
## 43 52-78      1   male     0          0.42         24
## 33 28-52      1   male     0          0.34         35
## 50   78+      1   male     0          0.32         28
## 10 10-15      3 female     0          0.31         13
## 20 15-24      3   male     0          0.17         30
## 28 24-28      3 female     0          0.17          6
## 39 28-52      3 female     1          0.17          6
## 8  10-15      2   male     0          0.12         57
## 38 28-52      3 female     0          0.12          8
## 5    <10      3   male     0          0.11        259
## 41 28-52      3   male     1          0.07         15
## 26 24-28      2   male     0          0.05         19
## 12 10-15      3   male     0          0.00         10
## 30 24-28      3   male     0          0.00          7
## 1    <10      1   male     0          0.00          6
## 2    <10      2   male     0          0.00          6
## 16 15-24      2   male     0          0.00          6
## 36 28-52      2   male     0          0.00          6
## 45 52-78      2   male     0          0.00          5
## 29 24-28      3 female     1          0.00          3
## 40 28-52      3   male     0          0.00          3
## 46 52-78      3 female     0          0.00          3
## 31 24-28      3   male     1          0.00          2
```

```r
## find the females that had low survival rate 
subset(survRateFareClassSexChildFull, Sex=="female" & Survival.Rate < 0.5)
```

```
##    Fare2 Pclass    Sex Child Survival.Rate Total.Size
## 10 10-15      3 female     0          0.31         13
## 28 24-28      3 female     0          0.17          6
## 29 24-28      3 female     1          0.00          3
## 38 28-52      3 female     0          0.12          8
## 39 28-52      3 female     1          0.17          6
## 46 52-78      3 female     0          0.00          3
```

#### Round 2 submission:  
Seesm like females in the 3rd class with a fare price 24-28, 28-52, or 25-78 mostly died. Let's use this information to modify the gender model for round 2 submission.  


```r
test <- read.csv("test.csv")
## create a column "Survived"
test$Survived <- 0
test[test$Survived == 0 & test$Sex == "female",]$Survived <- 1
test[test$Sex == "female" & test$Fare >= 24 & test$Fare < 78 & test$Pclass==3,]$Survived <- 0

result2 <- subset(test, select=c(PassengerId, Survived))
write.csv(result2, "result2.csv", row.names=FALSE)
```

#### Round 2 submission result 

My score improved. Now we have 0.77990 accuracy, beat the benchmark *Basic Random Forests Model* provied by Kaggle (0.77512).  

Based on this model, let's investigate male survivors; currently all males were assumed dead, but maybe there were characteristics indicative of their survival.  


```r
survRateFareClassSexChildMale <- subset(survRateFareClassSexChildFull, Sex=="male")
survRateFareClassSexChildMale[order(-survRateFareClassSexChildMale$Survival.Rate),]
```

```
##    Fare2 Pclass  Sex Child Survival.Rate Total.Size
## 6    <10      3 male     1          1.00          1
## 9  10-15      2 male     1          1.00          1
## 13 10-15      3 male     1          1.00          3
## 17 15-24      2 male     1          1.00          2
## 27 24-28      2 male     1          1.00          2
## 37 28-52      2 male     1          1.00          4
## 51   78+      1 male     1          1.00          3
## 21 15-24      3 male     1          0.67          6
## 47 52-78      3 male     0          0.45         11
## 23 24-28      1 male     0          0.42         26
## 43 52-78      1 male     0          0.42         24
## 33 28-52      1 male     0          0.34         35
## 50   78+      1 male     0          0.32         28
## 20 15-24      3 male     0          0.17         30
## 8  10-15      2 male     0          0.12         57
## 5    <10      3 male     0          0.11        259
## 41 28-52      3 male     1          0.07         15
## 26 24-28      2 male     0          0.05         19
## 1    <10      1 male     0          0.00          6
## 2    <10      2 male     0          0.00          6
## 12 10-15      3 male     0          0.00         10
## 16 15-24      2 male     0          0.00          6
## 30 24-28      3 male     0          0.00          7
## 31 24-28      3 male     1          0.00          2
## 36 28-52      2 male     0          0.00          6
## 40 28-52      3 male     0          0.00          3
## 45 52-78      2 male     0          0.00          5
```

It seems like male children, if stayed in a first or second class cabin, all survived.  


```r
subset(survRateFareClassSexChildMale, (Pclass==2 | Pclass==1) & Child==1)
```

```
##    Fare2 Pclass  Sex Child Survival.Rate Total.Size
## 9  10-15      2 male     1             1          1
## 17 15-24      2 male     1             1          2
## 27 24-28      2 male     1             1          2
## 37 28-52      2 male     1             1          4
## 51   78+      1 male     1             1          3
```

#### Round 3 submission:  
Assume male children stayed in first or second class survived. This assumption also is reflective of our learning from Wikipedia, that all children from second class survived, regardless of gender.  



```r
test <- read.csv("test.csv")
## create a column "Survived"
test$Survived <- 0
test[test$Survived == 0 & test$Sex == "female",]$Survived <- 1
test[test$Sex == "female" & test$Fare >= 24 & test$Fare < 78 & test$Pclass==3,]$Survived <- 0
test[test$Sex == "male" & (test$Pclass == 1 | test$Pclass == 2) & test$Age < 15 & !is.na(test$Age),]$Survived <- 1

result3 <- subset(test, select=c(PassengerId, Survived))
write.csv(result3, "result3.csv", row.names=FALSE)
```

#### Round 3 submission result:  
Score improved by 0.00478, now accuracy at 0.78469, also beat the *Gender, Price, and Class Based Model* provided by Kaggle (0.77990).  

### Decision Tree Model:  


```r
##install.packages("rpart")
library(rpart)
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class")
##install.packages('rattle')
##install.packages('rpart.plot')
##install.packages('RColorBrewer')
library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(fit)
```

![plot of chunk unnamed-chunk-19](figure/unnamed-chunk-19-1.png) 

```r
Prediction <- predict(fit, test, type = "class")
result4 <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(result4, "result4.csv", row.names=F)
```

### Round 4 submission result:  
Score did not improve, accuracy from rpart decision tree model is at 0.77033.  

Now let's change some settings within the rpart function via rpart.control.  


```r
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train,
             method="class", control=rpart.control(minisplit=5, cp=0))
fancyRpartPlot(fit)
```

![plot of chunk unnamed-chunk-20](figure/unnamed-chunk-20-1.png) 

```r
Prediction <- predict(fit, test, type = "class")
result5 <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(result5, "result5.csv", row.names=F)
```

### Round 5 submission result:  
Score did not improve; rather, the accuracy dropped to 0.73684, worse than the simple gender model.  

## Feature Engineering  

First, create a combined data frame with both training and test data.  

```r
test <- read.csv("test.csv")
train <- read.csv("train.csv")
test$Survived <- NA
combi <- rbind(train, test)
```

Make each passenger's title a factor.  


```r
##install.packages("stringr")
library(stringr)
combi$Name <- as.character(combi$Name)

cleanTitle <- function(x) {
    nameSplit <- str_trim(strsplit(x, ",")[[1]][2])
    nameSplitClean <- strsplit(nameSplit, "\\.")[[1]][1]
    nameSplitClean
}

combi$Title <- sapply(combi$Name, cleanTitle)
```

Summarize the titles:  
+ Capt: honorific addressed to someone served in military with Captain ranking
+ Col: someone served in military with Colonel ranking (higher than Captain)  
+ Don: Spanish honorific equivelant to "Mister"  
+ Dona: Spanish honorific equivelant to "Miss"  
+ Dr: Someone with a doctorate-level degree (likely wealthy)    
+ Jonkheer: Dutch honorific of nobility, equivelant to "Master"  
+ Lady: a woman of nobility  
+ Major: someone served in military, higher than Captain but lower than Colonel  
+ Master: a young unmarried boy  
+ Miss: a young unmarried woman  
+ Mlle: french honorific equivelant to "Miss"  
+ Mme: french honorific equivelant to "Mrs"  
+ Ms: equivelant to Mrs  
+ Rev: a Chiristian clergy or minister  
+ Sir: a knight  
+ Countess:  a woman of nobility  
 
 Combine duplicate titles:  
 

```r
combi[combi$Title %in% c("Capt", "Col", "Major", "Sir", "Dr"),]$Title <- "Sir"
combi[combi$Title %in% c("Don", "Mr"),]$Title <- "Mr"
combi[combi$Title %in% c("Dona", "Miss", "Mlle"),]$Title <- "Miss"
combi[combi$Title %in% c("Mrs", "Mme", "Ms"),]$Title <- "Mrs"
combi[combi$Title %in% c("Master", "Jonkheer"),]$Title <- "Master"
combi[combi$Title %in% c("Lady", "the Countess"),]$Title <- "Lady"
```

Now let's add a factor indicating the family size the passenger was traveling with, also identifying their family name.   


```r
combi$Fam.Size <- combi$Parch + combi$SibSp + 1
## create a new factor that is the traveler's last name.
combi$Last.Name <- as.character(combi$Name)

getLast.Name <- function(x) {
    Last.Name <- str_trim(strsplit(x, ",")[[1]][1])
    Last.Name
}

combi$Last.Name <- sapply(combi$Name, getLast.Name)
```

Create a new factor called "Fam.Id"" combining family name and family size.  


```r
combi$Fam.Id <- paste0(combi$Fam.Size, combi$Last.Name)
Fam.Id <- data.frame(table(combi$Fam.Id))
Fam.Id <- Fam.Id[order(-Fam.Id[,2]),]
head(Fam.Id)
```

```
##           Var1 Freq
## 1       11Sage   11
## 926 7Andersson    9
## 928   8Goodwin    8
## 927   7Asplund    7
## 921   6Fortune    6
## 922    6Panula    6
```

*Distractive side note: Number one on the list, Sage family, traveled with 11 family members. An article on [Encyclopedia-Titanica](http://www.encyclopedia-titanica.org/titanic-victim/john-george-sage.html) says that John George Sage, 44, was reloacting his entire family (including his wife and nine kids) to Jacksonville, FL when they boarded Titanic. The whole family were lost in the accident. Their daughter Stella (20) was able to board a lifeboat, but gave it up when her family were unable to join.*  

Any Fam.Id with frequency lower than 3 should be assigned a generic factor to prevent overfitting.  


```r
Fam.IdSmall <- subset(Fam.Id, Freq < 3)
combi[combi$Fam.Id %in% Fam.IdSmall$Var1,]$Fam.Id <- "Small"
combi$Fam.Id <- as.factor(combi$Fam.Id)
```

Before we split train and test data sets, let's do two things: at instances where Age is NA, fill in the average Age; and add an additional Child factor we have used before.  


```r
## add average age to NAs
combiYesAge <- combi[!is.na(combi$Age),]
combi[is.na(combi$Age),"Age"] <- mean(combiYesAge$Age)
## add Child factor
combi$Child <- 0
combi[combi$Age < 15,]$Child <- 1
```

Now let's separate test and training data sets.   


```r
train <- combi[1:891,]
test <- combi[892:nrow(combi),]
```

#### Round 6 submission.  
Do another rpart fit with new factors.  


```r
fit <- rpart(Survived ~ Pclass + Sex + Age + Child + SibSp + Parch + Fare + Embarked
             + Title + Fam.Size + Fam.Id, 
             data=train, method="class",
             control=rpart.control(minisplit=3))
fancyRpartPlot(fit)
```

![plot of chunk unnamed-chunk-29](figure/unnamed-chunk-29-1.png) 

```r
Prediction <- predict(fit, test, type = "class")
result5 <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(result5, "result5.csv", row.names=F)
```

#### Round 6 submission result.  
Score improved by 0.01914, accuracy at 0.80383.  

### Random Forests  

First, before we jump into Random Forests model creation, let's use rpart decision tree to predict the age, instead of using the average age.  


```r
## create combi data frame from scratch
###############################
test <- read.csv("test.csv")
train <- read.csv("train.csv")
test$Survived <- NA
combi <- rbind(train, test)

combi$Name <- as.character(combi$Name)
library(stringr)
cleanTitle <- function(x) {
    nameSplit <- str_trim(strsplit(x, ",")[[1]][2])
    nameSplitClean <- strsplit(nameSplit, "\\.")[[1]][1]
    nameSplitClean
}

combi$Title <- sapply(combi$Name, cleanTitle)

combi[combi$Title %in% c("Capt", "Col", "Major", "Sir", "Dr"),]$Title <- "Sir"
combi[combi$Title %in% c("Don", "Mr"),]$Title <- "Mr"
combi[combi$Title %in% c("Dona", "Miss", "Mlle"),]$Title <- "Miss"
combi[combi$Title %in% c("Mrs", "Mme", "Ms"),]$Title <- "Mrs"
combi[combi$Title %in% c("Master", "Jonkheer"),]$Title <- "Master"
combi[combi$Title %in% c("Lady", "the Countess"),]$Title <- "Lady"

combi$Fam.Size <- combi$Parch + combi$SibSp + 1
## create a new factor that is the traveler's last name.
combi$Last.Name <- as.character(combi$Name)

getLast.Name <- function(x) {
    Last.Name <- str_trim(strsplit(x, ",")[[1]][1])
    Last.Name
}

combi$Last.Name <- sapply(combi$Name, getLast.Name)

combi$Fam.Id <- paste0(combi$Fam.Size, combi$Last.Name)
Fam.Id <- data.frame(table(combi$Fam.Id))
Fam.Id <- Fam.Id[order(-Fam.Id[,2]),]

Fam.IdSmall <- subset(Fam.Id, Freq < 3)
combi[combi$Fam.Id %in% Fam.IdSmall$Var1,]$Fam.Id <- "Small"
combi$Fam.Id <- as.factor(combi$Fam.Id)

################################
## use rpart to predict age 
hasAge <- combi[!is.na(combi$Age),]
noAge <- combi[is.na(combi$Age),]
library(rpart)
ageFit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked
             + Title, 
             data=hasAge, method="anova")
combi[is.na(combi$Age),]$Age <- predict(ageFit, noAge)
## add Child factor
combi$Child <- 0
combi[combi$Age < 15,]$Child <- 1
```

Make sure there is no NAs in any factors.  


```r
summary(combi)
```

```
##   PassengerId      Survived          Pclass          Name          
##  Min.   :   1   Min.   :0.0000   Min.   :1.000   Length:1309       
##  1st Qu.: 328   1st Qu.:0.0000   1st Qu.:2.000   Class :character  
##  Median : 655   Median :0.0000   Median :3.000   Mode  :character  
##  Mean   : 655   Mean   :0.3838   Mean   :2.295                     
##  3rd Qu.: 982   3rd Qu.:1.0000   3rd Qu.:3.000                     
##  Max.   :1309   Max.   :1.0000   Max.   :3.000                     
##                 NA's   :418                                        
##      Sex           Age            SibSp            Parch      
##  female:466   Min.   : 0.17   Min.   :0.0000   Min.   :0.000  
##  male  :843   1st Qu.:22.00   1st Qu.:0.0000   1st Qu.:0.000  
##               Median :28.86   Median :0.0000   Median :0.000  
##               Mean   :29.70   Mean   :0.4989   Mean   :0.385  
##               3rd Qu.:36.50   3rd Qu.:1.0000   3rd Qu.:0.000  
##               Max.   :80.00   Max.   :8.0000   Max.   :9.000  
##                                                               
##       Ticket          Fare                     Cabin      Embarked
##  CA. 2343:  11   Min.   :  0.000                  :1014    :  2   
##  1601    :   8   1st Qu.:  7.896   C23 C25 C27    :   6   C:270   
##  CA 2144 :   8   Median : 14.454   B57 B59 B63 B66:   5   Q:123   
##  3101295 :   7   Mean   : 33.295   G6             :   5   S:914   
##  347077  :   7   3rd Qu.: 31.275   B96 B98        :   4           
##  347082  :   7   Max.   :512.329   C22 C26        :   4           
##  (Other) :1261   NA's   :1         (Other)        : 271           
##     Title              Fam.Size       Last.Name                Fam.Id    
##  Length:1309        Min.   : 1.000   Length:1309        Small     :1017  
##  Class :character   1st Qu.: 1.000   Class :character   11Sage    :  11  
##  Mode  :character   Median : 1.000   Mode  :character   7Andersson:   9  
##                     Mean   : 1.884                      8Goodwin  :   8  
##                     3rd Qu.: 2.000                      7Asplund  :   7  
##                     Max.   :11.000                      6Fortune  :   6  
##                                                         (Other)   : 251  
##      Child        
##  Min.   :0.00000  
##  1st Qu.:0.00000  
##  Median :0.00000  
##  Mean   :0.09244  
##  3rd Qu.:0.00000  
##  Max.   :1.00000  
## 
```

```r
## there are two missing values in Embarked. Let's examine this factor.
combi[order(combi$Ticket),c("Ticket", "Embarked")][40:63,]
```

```
##      Ticket Embarked
## 1110 113503        C
## 1299 113503        C
## 167  113505        S
## 357  113505        S
## 55   113509        C
## 918  113509        C
## 352  113510        S
## 253  113514        S
## 62   113572         
## 830  113572         
## 391  113760        S
## 436  113760        S
## 764  113760        S
## 803  113760        S
## 186  113767        S
## 749  113773        S
## 1074 113773        S
## 152  113776        S
## 337  113776        S
## 298  113781        S
## 306  113781        S
## 499  113781        S
## 709  113781        S
## 1033 113781        S
```

```r
## based on the ticket number, it's almost confirmed that these two passengers embarked from S.
combi[combi$Embarked == "",]$Embarked <- "S"
## There is one passenger with Fare missing. Let's examine that.
combi[is.na(combi$Fare),]
```

```
##      PassengerId Survived Pclass               Name  Sex  Age SibSp Parch
## 1044        1044       NA      3 Storey, Mr. Thomas male 60.5     0     0
##      Ticket Fare Cabin Embarked Title Fam.Size Last.Name Fam.Id Child
## 1044   3701   NA              S    Mr        1    Storey  Small     0
```

```r
##  examine ticket prices similar to his ticket number
ticket <- data.frame(combi)
ticket$Ticket <- as.integer(as.character(combi$Ticket))
```

```
## Warning: NAs introduced by coercion
```

```r
subset(ticket, Ticket > 2800 & Ticket < 4200)
```

```
##      PassengerId Survived Pclass
## 54            54        1      2
## 114          114        0      3
## 177          177        0      3
## 230          230        0      3
## 403          403        0      3
## 410          410        0      3
## 478          478        0      3
## 484          484        1      3
## 486          486        0      3
## 504          504        0      3
## 544          544        1      2
## 547          547        1      2
## 585          585        0      3
## 678          678        1      3
## 811          811        0      3
## 1000        1000       NA      3
## 1024        1024       NA      3
## 1044        1044       NA      3
## 1135        1135       NA      3
## 1169        1169       NA      2
##                                                    Name    Sex       Age
## 54   Faunthorpe, Mrs. Lizzie (Elizabeth Anne Wilkinson) female 29.000000
## 114                             Jussila, Miss. Katriina female 20.000000
## 177                       Lefebre, Master. Henry Forbes   male  7.123786
## 230                             Lefebre, Miss. Mathilde female  7.123786
## 403                            Jussila, Miss. Mari Aina female 21.000000
## 410                                  Lefebre, Miss. Ida female  7.123786
## 478                           Braund, Mr. Lewis Richard   male 29.000000
## 484                              Turkula, Mrs. (Hedwig) female 63.000000
## 486                              Lefebre, Miss. Jeannie female  7.123786
## 504                      Laitinen, Miss. Kristina Sofia female 37.000000
## 544                                   Beane, Mr. Edward   male 32.000000
## 547                   Beane, Mrs. Edward (Ethel Clarke) female 19.000000
## 585                                 Paulner, Mr. Uscher   male 28.862881
## 678                             Turja, Miss. Anna Sofia female 18.000000
## 811                              Alexander, Mr. William   male 26.000000
## 1000                   Willer, Mr. Aaron (Abi Weller")"   male 28.862881
## 1024                      Lefebre, Mrs. Frank (Frances) female 28.862881
## 1044                                 Storey, Mr. Thomas   male 60.500000
## 1135                                 Hyman, Mr. Abraham   male 28.862881
## 1169                              Faunthorpe, Mr. Harry   male 40.000000
##      SibSp Parch Ticket    Fare Cabin Embarked  Title Fam.Size  Last.Name
## 54       1     0   2926 26.0000              S    Mrs        2 Faunthorpe
## 114      1     0   4136  9.8250              S   Miss        2    Jussila
## 177      3     1   4133 25.4667              S Master        5    Lefebre
## 230      3     1   4133 25.4667              S   Miss        5    Lefebre
## 403      1     0   4137  9.8250              S   Miss        2    Jussila
## 410      3     1   4133 25.4667              S   Miss        5    Lefebre
## 478      1     0   3460  7.0458              S     Mr        2     Braund
## 484      0     0   4134  9.5875              S    Mrs        1    Turkula
## 486      3     1   4133 25.4667              S   Miss        5    Lefebre
## 504      0     0   4135  9.5875              S   Miss        1   Laitinen
## 544      1     0   2908 26.0000              S     Mr        2      Beane
## 547      1     0   2908 26.0000              S    Mrs        2      Beane
## 585      0     0   3411  8.7125              C     Mr        1    Paulner
## 678      0     0   4138  9.8417              S   Miss        1      Turja
## 811      0     0   3474  7.8875              S     Mr        1  Alexander
## 1000     0     0   3410  8.7125              S     Mr        1     Willer
## 1024     0     4   4133 25.4667              S    Mrs        5    Lefebre
## 1044     0     0   3701      NA              S     Mr        1     Storey
## 1135     0     0   3470  7.8875              S     Mr        1      Hyman
## 1169     1     0   2926 26.0000              S     Mr        2 Faunthorpe
##        Fam.Id Child
## 54      Small     0
## 114     Small     0
## 177  5Lefebre     1
## 230  5Lefebre     1
## 403     Small     0
## 410  5Lefebre     1
## 478     Small     0
## 484     Small     0
## 486  5Lefebre     1
## 504     Small     0
## 544     Small     0
## 547     Small     0
## 585     Small     0
## 678     Small     0
## 811     Small     0
## 1000    Small     0
## 1024 5Lefebre     0
## 1044    Small     0
## 1135    Small     0
## 1169    Small     0
```

```r
## it's reasonable to assume that his ticket price should be around $8, but let's use the median of 3rd class fare
combi$Fare[which(is.na(combi$Fare))] <- median(combi[combi$Pclass ==3 & !is.na(combi$Fare),]$Fare)
```

Because Fam.Id has too many levels, instead of using family id, let's use two factors only: small family or big family.  


```r
combi$Fam.Id2 <- as.character(combi$Fam.Id)
combi$Fam.Id2[combi$Fam.Id2 != "Small"] <- "Big"
combi$Fam.Id2 <- factor(combi$Fam.Id2)
```

Now it's time to create the Random Forests.  


```r
##install.packages("randomForest")
library(randomForest)
combi$Title <- as.factor(combi$Title)
combi$Last.Name <- as.factor(combi$Last.Name)
train <- combi[1:891,]
test <- combi[892:nrow(combi),]
test$Survived <- as.integer(0)
set.seed(12)
treeFit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + Child + SibSp
                    + Parch + Fare + Embarked + Title + Fam.Size 
                    + Fam.Id2, data=train, importance=TRUE, ntree=2000)
varImpPlot(treeFit)
```

![plot of chunk unnamed-chunk-33](figure/unnamed-chunk-33-1.png) 

#### Round 7 submission  


```r
Prediction <- predict(treeFit, test)
result7 <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(result7, file="result7.csv", row.names=FALSE)
```

#### Round 7 submission result  
Did not improve; accuracy dropped to 0.77512.  


#### Round 8 submission  

Use conditional Random Forest.  


```r
##install.packages("party")
library(party)
set.seed(12)
treeFit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch
                   + Fare + Embarked + Title + Fam.Size + Fam.Id,
                   data=train, controls=cforest_unbiased(ntree=2000, mtry=3))

Prediction <- predict(treeFit, test, OOB=TRUE, type="response")
result8 <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(result8, file="result8.csv", row.names=FALSE)
```

#### Round 8 submission result  
Score did not improve from the best entry (0.80383).  

### Cross-Validation  

We will use some cross-validation techniques to examine our model performance before submission. Models will be applied on training datasets, and accuracy will be measured.  

First, with cross-validation, let's examine the average accuracy of our last rpart model, the one produced 0.80383 accuracy.  


```r
##install.packagess("ROCR")
library(ROCR)
##install.packages("gbm")
library(gbm)

k=10
n = floor(nrow(train)/k)
errvect1 <- c()
resultRpart <- as.numeric()

for(i in 1:10) {
    s1 = (i - 1) * n + 1
    s2 = i * n
    subset = s1:s2
    cv.train = train[-subset,]
    cv.test = train[subset,]
    
    fit <- rpart(Survived ~ Pclass + Sex + Age + Child + SibSp + Parch + Fare + Embarked
             + Title + Fam.Size + Fam.Id, 
             data=cv.train, method="class",
             control=rpart.control(minisplit=3))
    
    fit.pr <- predict(fit, cv.test, type="prob")[,2]
    fit.pred <- prediction(fit.pr, cv.test[,2])
    ##fit.perf <- performance(fit.pred, "prec", "rec")
    ##plot(fit.perf, main="PR Curve for Random Forest", col=2, lwd=2)
    
    fit.perf <- performance(fit.pred, "tpr", "fpr")
    #plot(fit.perf, main="ROC Curve for Random Forest", col=2, lwd=2)
    #abline(a=0, b=1, lwd=2, lty=2, col="gray")
    auc <- performance(fit.pred, "auc")
    auc <- unlist(slot(auc, "y.values"))
    errvect1[i] <- auc
    print(paste("AUC for fold: ", i, errvect1[i]))
    resultRpart <- c(resultRpart, errvect1[i])
}
```

```
## [1] "AUC for fold:  1 0.840512820512821"
## [1] "AUC for fold:  2 0.868115942028985"
## [1] "AUC for fold:  3 0.793831168831169"
## [1] "AUC for fold:  4 0.848989898989899"
## [1] "AUC for fold:  5 0.825363825363825"
## [1] "AUC for fold:  6 0.809253246753247"
## [1] "AUC for fold:  7 0.776923076923077"
## [1] "AUC for fold:  8 0.774122807017544"
## [1] "AUC for fold:  9 0.864406779661017"
## [1] "AUC for fold:  10 0.855614973262032"
```

```r
print(paste("Avg AUC: ", mean(errvect1)))
```

```
## [1] "Avg AUC:  0.825713453934362"
```

```r
meanRpart <- mean(errvect1)
t.test(resultRpart - resultRForestNew)
```

```
## 
## 	One Sample t-test
## 
## data:  resultRpart - resultRForestNew
## t = -3.2471, df = 9, p-value = 0.01004
## alternative hypothesis: true mean is not equal to 0
## 95 percent confidence interval:
##  -0.08323171 -0.01487993
## sample estimates:
##   mean of x 
## -0.04905582
```

How about the Conditional Random Forest, the one received the same score? Let's modify the above code slightly.  


```r
k=10
n = floor(nrow(train)/k)
errvect2 <- c()

for(i in 1:10) {
    s1 = (i - 1) * n + 1
    s2 = i * n
    subset = s1:s2
    cv.train = train[-subset,]
    cv.test = train[subset,]
    
    treeFit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch
                   + Fare + Embarked + Title + Fam.Size + Fam.Id,
                   data=cv.train, controls=cforest_unbiased(ntree=2000, mtry=3))
    fit.pr <- predict(treeFit, cv.test, OOB=TRUE, type="response")
    
    ##fit.pr <- predict(fit, cv.test, type="prob")[,2]
    fit.pred <- prediction(as.integer(fit.pr), cv.test[,2])
    ##fit.perf <- performance(fit.pred, "prec", "rec")
    ##plot(fit.perf, main="PR Curve for Random Forest", col=2, lwd=2)
    
    fit.perf <- performance(fit.pred, "tpr", "fpr")
    #plot(fit.perf, main="ROC Curve for Random Forest", col=2, lwd=2)
    abline(a=0, b=1, lwd=2, lty=2, col="gray")
    auc <- performance(fit.pred, "auc")
    auc <- unlist(slot(auc, "y.values"))
    errvect2[i] <- auc
    print(paste("AUC for fold: ", i, errvect2[i]))
}
```

```
## Error in int_abline(a = a, b = b, h = h, v = v, untf = untf, ...): plot.new has not been called yet
```

```r
print(paste("Avg AUC: ", mean(errvect2)))
```

```
## Warning in mean.default(errvect2): argument is not numeric or logical:
## returning NA
```

```
## [1] "Avg AUC:  NA"
```

```r
meanCForest <- mean(errvect2)    
```

```
## Warning in mean.default(errvect2): argument is not numeric or logical:
## returning NA
```

```r
## this line is unreproducible as I don't want to rerun the calculation a
resultCForest <- c(0.808974358974359, 0.838768115942029,
                   0.761904761904762, 0.832323232323232,
                   0.834199584199584, 0.797619047619048,
                   0.793333333333333, 0.730537280701754,
                   0.816101694915254, 0.836898395721925)
```

The real world prediction result from those two models were the same, meaning the difference we see in cross-validation may be caused by a variety of factors. Let's calculate the p-value based on the null hypothesis that the accuracy of those two models should be the same.  
H0: there is no difference in prediction accuracy betwen the rpart model and cforest model we used;  
H1: there is difference in prediction accuracy between those two models.  


```r
t.test(resultRpart - resultCForest)$p.value
```

```
## [1] 0.01243219
```

```r
t.test(resultRpart - resultCForest)$conf
```

```
## [1] 0.005649653 0.035645294
## attr(,"conf.level")
## [1] 0.95
```

Interestingly, the p-value is <0.05, and both confidence intervals are above zero -- theoretically speaking, the rpart model should be better than cforest. The null hypothesis is rejected. The actual result indifference may be caused by randomness.  

What about the Random Forest we used earlier, the one produced 0.77512 accuracy -- is it a model that may look good on paper but performs poorly in practice too?  


```r
errvect3 <- c()
resultRForest <- as.numeric()

train <- train[sample(nrow(train)), ]
test <- test[sample(nrow(test)), ]

for(i in 1:10) {
    s1 = (i - 1) * n + 1
    s2 = i * n
    subset = s1:s2
    cv.train = train[-subset,]
    cv.test = train[subset,]
    
    treeFit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + Child + SibSp
                    + Parch + Fare + Embarked + Title + Fam.Size 
                    + Fam.Id2, data=cv.train, importance=TRUE, ntree=2000)
    
    fit.pr <- predict(treeFit, cv.test, type="prob")[,2]
    fit.pred <- prediction(fit.pr, cv.test[,2])
    ##fit.perf <- performance(fit.pred, "prec", "rec")
    ##plot(fit.perf, main="PR Curve for Random Forest", col=2, lwd=2)
    
    fit.perf <- performance(fit.pred, "tpr", "fpr")
    #plot(fit.perf, main="ROC Curve for Random Forest", col=2, lwd=2)
    #abline(a=0, b=1, lwd=2, lty=2, col="gray")
    auc <- performance(fit.pred, "auc")
    auc <- unlist(slot(auc, "y.values"))
    errvect3[i] <- auc
    print(paste("AUC for fold: ", i, errvect3[i]))
    resultRForest <- c(resultRForest, errvect3[i])
}
```

```
## [1] "AUC for fold:  1 0.815561224489796"
## [1] "AUC for fold:  2 0.854420731707317"
## [1] "AUC for fold:  3 0.884199134199134"
## [1] "AUC for fold:  4 0.87987012987013"
## [1] "AUC for fold:  5 0.917359667359667"
## [1] "AUC for fold:  6 0.858021390374331"
## [1] "AUC for fold:  7 0.92825361512792"
## [1] "AUC for fold:  8 0.879885057471265"
## [1] "AUC for fold:  9 0.880978865406007"
## [1] "AUC for fold:  10 0.857954545454545"
```

```r
print(paste("Avg AUC: ", mean(errvect3)))
```

```
## [1] "Avg AUC:  0.875650436146011"
```

```r
meanRForest <- mean(errvect3)   
```

Let's do a t.test. From initial look, the Random Forest model we used earlier should theoretically outperform the rpart model. But, reality is different.  


```r
t.test(resultRForest - resultRpart)$p.value
```

```
## [1] 0.02337598
```

```r
t.test(resultRForest - resultRpart)$conf
```

```
## [1] 0.008497473 0.091376491
## attr(,"conf.level")
## [1] 0.95
```

Well, yes, on paper, the Random Forest performed much better than rpart on training data, but failed on test data. 

### Feature Engineering Pt. 2 -- Cabin Letter  

Majority of the Cabin information is missing, let's use rpart to predict the rest of the passengers' Cabin initials.  


```r
combi$Cabin.Letter <- substr(combi$Cabin, 1,1)
combi$Cabin.Letter <- as.factor(as.character(combi$Cabin.Letter))
noCabin <- combi[combi$Cabin == "",]
hasCabin <- combi[combi$Cabin != "",]

cabinFit <- rpart(Cabin.Letter ~ Pclass + Sex + SibSp + Parch + Fare + Embarked
             + Title, 
             data=hasCabin)
fancyRpartPlot(cabinFit)
```

![plot of chunk unnamed-chunk-41](figure/unnamed-chunk-41-1.png) 

```r
predict(cabinFit, noCabin, type="class")
```

```
##    1    3    5    6    8    9   10   13   14   15   16   17   18   19   20 
##    F    F    F    F    F    G    F    F    F    F    D    F    D    D    F 
##   21   23   25   26   27   29   30   31   33   34   35   36   37   38   39 
##    F    F    F    F    F    F    F    C    F    F    B    C    F    F    D 
##   40   41   42   43   44   45   46   47   48   49   50   51   52   54   57 
##    F    F    F    F    F    F    F    D    F    F    D    F    F    F    F 
##   58   59   60   61   64   65   66   68   69   70   71   72   73   74   75 
##    F    F    F    F    F    C    G    F    G    F    F    F    F    D    F 
##   77   78   79   80   81   82   83   84   85   86   87   88   90   91   92 
##    F    F    F    F    F    F    F    C    F    D    F    F    F    F    F 
##   94   95   96   99  100  101  102  104  105  106  107  108  109  110  112 
##    F    F    F    F    F    F    F    F    F    F    F    F    F    F    D 
##  113  114  115  116  117  118  120  121  122  123  126  127  128  130  131 
##    F    F    D    F    F    F    F    F    F    F    F    F    F    F    F 
##  132  133  134  135  136  139  141  142  143  144  145  146  147  148  150 
##    F    D    F    D    D    F    G    F    D    F    F    F    F    F    D 
##  151  153  154  155  156  157  158  159  160  161  162  163  164  165  166 
##    F    F    G    F    B    F    F    F    F    G    D    F    F    F    F 
##  168  169  170  172  173  174  176  177  179  180  181  182  183  185  187 
##    F    E    F    F    G    F    G    F    D    F    F    D    F    F    D 
##  188  189  190  191  192  193  197  198  199  200  201  202  203  204  205 
##    C    G    F    D    D    F    F    G    F    D    F    F    F    F    F 
##  207  208  209  211  212  213  214  215  217  218  220  221  222  223  224 
##    D    D    F    F    F    F    D    F    F    F    F    F    D    F    F 
##  226  227  228  229  230  232  233  234  235  236  237  238  239  240  241 
##    F    F    F    D    F    F    D    F    F    F    F    F    F    F    D 
##  242  243  244  245  247  248  250  251  254  255  256  257  259  260  261 
##    D    F    F    F    F    G    F    F    D    F    G    B    B    F    F 
##  262  265  266  267  268  271  272  273  275  277  278  279  280  281  282 
##    F    F    F    F    F    C    F    G    F    F    F    F    F    F    F 
##  283  284  286  287  288  289  290  291  294  295  296  297  301  302  303 
##    F    F    F    F    F    D    F    D    F    F    C    F    F    F    F 
##  305  307  309  313  314  315  316  317  318  321  322  323  324  325  327 
##    F    C    F    F    F    F    F    F    D    F    F    F    F    F    F 
##  329  331  334  335  336  339  343  344  345  347  348  349  350  351  353 
##    F    F    D    B    F    F    D    D    D    D    D    G    F    F    G 
##  354  355  356  358  359  360  361  362  363  364  365  366  368  369  372 
##    D    F    F    D    F    F    F    F    G    F    D    F    F    F    F 
##  373  374  375  376  377  379  380  381  382  383  384  385  386  387  388 
##    F    C    F    B    F    F    F    C    G    F    C    F    F    F    D 
##  389  390  392  393  396  397  398  399  400  401  402  403  404  405  406 
##    F    F    F    F    F    F    F    F    D    F    F    F    D    F    F 
##  407  408  409  410  411  412  414  415  416  417  418  419  420  421  422 
##    F    G    F    F    F    F    F    F    F    F    G    D    F    F    F 
##  423  424  425  426  427  428  429  432  433  434  437  438  440  441  442 
##    F    G    F    F    F    F    F    D    F    F    F    G    F    F    F 
##  443  444  445  447  448  449  451  452  455  456  459  460  462  464  465 
##    F    D    F    G    C    G    F    F    F    F    F    F    F    D    F 
##  466  467  468  469  470  471  472  473  475  477  478  479  480  481  482 
##    F    F    C    F    G    F    F    F    F    F    F    F    G    F    F 
##  483  484  486  489  490  491  492  494  495  496  498  500  501  502  503 
##    F    F    F    F    G    F    F    A    F    D    D    F    F    F    F 
##  504  507  508  509  510  511  512  514  515  518  519  520  522  523  525 
##    F    F    C    F    F    F    F    C    F    F    F    F    F    F    F 
##  526  527  529  530  531  532  533  534  535  536  538  539  542  543  544 
##    F    F    F    G    F    F    G    F    F    F    C    D    F    F    F 
##  546  547  548  549  550  552  553  554  555  556  558  560  561  562  563 
##    E    F    D    F    F    F    F    F    F    C    C    D    F    F    D 
##  564  565  566  567  568  569  570  571  574  575  576  577  579  580  581 
##    F    F    F    F    F    F    F    F    F    F    D    D    D    F    F 
##  583  585  587  589  590  591  593  594  595  596  597  598  599  601  602 
##    F    F    D    F    F    F    F    G    F    F    F    F    F    F    F 
##  603  604  605  606  607  608  609  611  612  613  614  615  616  617  618 
##    C    F    C    D    F    C    F    F    F    D    F    F    F    G    D 
##  620  621  623  624  625  627  629  630  632  634  635  636  637  638  639 
##    F    D    G    F    D    F    F    F    F    B    F    D    F    F    F 
##  640  641  643  644  645  647  649  650  651  652  653  654  655  656  657 
##    D    F    F    F    G    F    F    F    F    F    F    F    F    F    F 
##  658  659  661  662  664  665  666  667  668  669  671  673  674  675  676 
##    G    D    B    F    F    F    F    D    F    F    F    F    D    F    F 
##  677  678  679  681  683  684  685  686  687  688  689  692  693  694  695 
##    F    F    F    F    F    F    F    F    F    F    F    G    F    F    C 
##  696  697  698  703  704  705  706  707  709  710  714  715  719  720  721 
##    D    F    F    G    F    F    F    D    C    G    F    D    D    F    F 
##  722  723  724  726  727  728  729  730  732  733  734  735  736  737  739 
##    F    D    D    F    F    F    F    F    D    F    D    D    D    F    F 
##  740  744  745  747  748  750  751  753  754  755  756  757  758  759  761 
##    F    D    F    F    D    F    F    F    F    F    G    F    F    F    D 
##  762  763  765  767  768  769  770  771  772  774  775  776  778  779  781 
##    F    F    F    A    F    F    F    F    F    F    F    F    F    F    F 
##  784  785  786  787  788  789  791  792  793  794  795  796  798  799  800 
##    F    F    F    F    F    F    F    F    F    A    F    D    F    F    F 
##  801  802  804  805  806  808  809  811  812  813  814  815  817  818  819 
##    D    F    G    F    F    F    D    F    F    F    F    F    F    F    F 
##  820  822  823  825  826  827  828  829  831  832  833  834  835  837  838 
##    F    F    B    F    F    F    F    F    D    G    F    F    F    F    F 
##  839  841  842  843  844  845  846  847  848  849  851  852  853  855  856 
##    F    F    F    A    F    F    F    F    F    F    F    F    G    F    G 
##  857  859  860  861  862  864  865  866  867  869  870  871  874  875  876 
##    C    G    F    D    F    F    D    D    D    F    G    F    F    F    F 
##  877  878  879  881  882  883  884  885  886  887  889  891  892  893  894 
##    F    F    F    F    F    F    F    F    F    D    F    F    F    F    F 
##  895  896  897  898  899  900  901  902  903  905  907  908  909  910  911 
##    F    G    F    F    F    F    F    F    E    F    F    F    F    F    F 
##  912  913  914  915  917  919  921  922  923  924  925  927  928  929  930 
##    C    G    C    B    D    F    F    F    F    F    F    F    F    F    F 
##  931  932  934  935  937  939  941  943  944  946  947  948  950  952  953 
##    F    G    F    D    F    F    G    D    F    D    F    F    D    F    D 
##  954  955  957  958  959  962  963  964  968  970  971  972  974  975  976 
##    F    F    F    F    C    F    F    F    F    D    F    G    E    F    F 
##  977  978  979  980  981  982  983  985  986  987  989  990  991  993  994 
##    D    F    F    F    F    D    F    F    C    F    F    F    F    F    F 
##  995  996  997  998  999 1000 1002 1003 1005 1007 1008 1011 1012 1013 1015 
##    F    G    F    F    F    F    D    F    F    D    F    F    D    F    F 
## 1016 1017 1018 1019 1020 1021 1022 1024 1025 1026 1027 1028 1029 1030 1031 
##    F    G    F    F    D    F    F    F    F    F    F    F    D    F    F 
## 1032 1033 1035 1036 1037 1039 1040 1041 1043 1044 1045 1046 1047 1049 1051 
##    F    C    F    C    D    F    C    F    F    F    G    F    F    F    G 
## 1052 1053 1054 1055 1056 1057 1059 1060 1061 1062 1063 1064 1065 1066 1067 
##    F    G    D    F    D    F    F    C    F    F    F    D    F    F    F 
## 1068 1072 1075 1077 1078 1079 1080 1081 1082 1083 1084 1085 1086 1087 1089 
##    F    D    F    D    F    F    F    D    F    E    G    F    F    F    F 
## 1090 1091 1092 1093 1095 1096 1097 1098 1099 1101 1102 1103 1104 1105 1106 
##    F    F    D    G    F    F    C    F    F    F    F    F    F    F    G 
## 1108 1109 1111 1112 1113 1115 1116 1117 1118 1119 1120 1121 1122 1123 1124 
##    F    C    F    D    F    F    C    G    F    F    D    D    F    C    F 
## 1125 1127 1129 1130 1132 1133 1135 1136 1138 1139 1140 1141 1142 1143 1145 
##    F    F    F    G    C    F    F    F    F    F    F    D    F    F    F 
## 1146 1147 1148 1149 1150 1151 1152 1153 1154 1155 1156 1157 1158 1159 1160 
##    F    F    F    F    D    F    D    F    F    G    D    F    B    F    F 
## 1161 1163 1165 1166 1167 1168 1169 1170 1171 1172 1173 1174 1175 1176 1177 
##    F    F    D    F    F    F    F    F    F    F    G    F    G    F    F 
## 1178 1181 1182 1183 1184 1186 1187 1188 1189 1190 1191 1192 1194 1195 1196 
##    F    F    C    F    F    F    F    F    F    C    F    F    F    F    F 
## 1199 1201 1202 1203 1204 1205 1207 1209 1210 1211 1212 1215 1216 1217 1219 
##    G    D    F    F    F    F    F    F    F    F    F    C    C    F    B 
## 1220 1221 1222 1224 1225 1226 1228 1229 1230 1231 1232 1233 1234 1236 1237 
##    F    D    F    F    G    F    D    G    F    F    F    F    F    G    F 
## 1238 1239 1240 1241 1243 1244 1245 1246 1249 1250 1251 1252 1253 1254 1255 
##    D    F    D    F    F    F    F    F    F    F    D    F    F    F    F 
## 1257 1258 1259 1260 1261 1262 1265 1267 1268 1269 1271 1272 1273 1274 1275 
##    F    D    F    B    D    F    D    B    F    F    F    F    F    D    D 
## 1276 1277 1278 1279 1280 1281 1284 1285 1286 1288 1290 1291 1293 1294 1295 
##    D    F    F    D    F    F    F    F    F    F    F    F    F    B    C 
## 1298 1300 1301 1302 1304 1305 1307 1308 1309 
##    F    F    G    F    F    F    F    F    F 
## Levels:  A B C D E F G T
```

```r
combi[combi$Cabin == "",]$Cabin.Letter <- predict(cabinFit, noCabin, type="class")
```

Now we have predicted the Cabin Letter for all passengers, time to pass that info into another Random Forest model.  


```r
train <- combi[1:891,]
test <- combi[892:nrow(combi),]

k = 10
n = floor(nrow(train)/k)
errvect5 <- c()
resultRForestCabin <- as.numeric()

for(i in 1:10) {
    s1 = (i - 1) * n + 1
    s2 = i * n
    subset = s1:s2
    cv.train = train[-subset,]
    cv.test = train[subset,]
    
    treeFit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + Child + SibSp
                    + Parch + Fare + Embarked + Title + Fam.Size 
                    + Fam.Id2 + Cabin.Letter, data=cv.train, importance=TRUE, ntree=2000)
    
    fit.pr <- predict(treeFit, cv.test, type="prob")[,2]
    fit.pred <- prediction(fit.pr, cv.test[,2])
    ##fit.perf <- performance(fit.pred, "prec", "rec")
    ##plot(fit.perf, main="PR Curve for Random Forest", col=2, lwd=2)
    
    fit.perf <- performance(fit.pred, "tpr", "fpr")
    #plot(fit.perf, main="ROC Curve for Random Forest", col=2, lwd=2)
    #abline(a=0, b=1, lwd=2, lty=2, col="gray")
    auc <- performance(fit.pred, "auc")
    auc <- unlist(slot(auc, "y.values"))
    errvect5[i] <- auc
    print(paste("AUC for fold: ", i, errvect5[i]))
    resultRForestCabin <- c(resultRForestCabin, errvect5[i])
}
```

```
## [1] "AUC for fold:  1 0.843846153846154"
## [1] "AUC for fold:  2 0.843840579710145"
## [1] "AUC for fold:  3 0.798160173160173"
## [1] "AUC for fold:  4 0.871464646464646"
## [1] "AUC for fold:  5 0.896569646569647"
## [1] "AUC for fold:  6 0.897186147186147"
## [1] "AUC for fold:  7 0.864102564102564"
## [1] "AUC for fold:  8 0.906798245614035"
## [1] "AUC for fold:  9 0.919774011299435"
## [1] "AUC for fold:  10 0.895454545454546"
```

```r
print(paste("Avg AUC: ", mean(errvect5)))
```

```
## [1] "Avg AUC:  0.873719671340749"
```

```r
t.test(resultRForestCabin - resultRForest)
```

```
## 
## 	One Sample t-test
## 
## data:  resultRForestCabin - resultRForest
## t = -0.1363, df = 9, p-value = 0.8946
## alternative hypothesis: true mean is not equal to 0
## 95 percent confidence interval:
##  -0.03398034  0.03011881
## sample estimates:
##    mean of x 
## -0.001930765
```

Did not see significant improvement from cross-validation test, but let's make a submission to see actual result.  

#### Round 9 submission  


```r
treeFit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + Child + SibSp
                    + Parch + Fare + Embarked + Title + Fam.Size 
                    + Fam.Id2 + Cabin.Letter, data=train, importance=TRUE, ntree=2000)
Prediction <- predict(treeFit, test)
result9 <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(result9, file="result9.csv", row.names=FALSE)
```

#### Round 9 submission result  

Score did not improve, at 0.77512.  

Seems like Cabin.Letter carries certain importance, let's create a new class that combines factors ralated to Cabin.Letter -- Pclass.  


```r
combi$Wealth <- paste0(combi$Cabin.Letter, combi$Pclass)
combi$Wealth <- as.factor(combi$Wealth)

train <- combi[1:891,]
test <- combi[892:nrow(combi),]

k = 10
n = floor(nrow(train)/k)
errvect5 <- c()
resultRForestCabin <- as.numeric()

## randomnize rows
train <- train[sample(nrow(train)), ]
test <- test[sample(nrow(test)), ]

for(i in 1:10) {
    s1 = (i - 1) * n + 1
    s2 = i * n
    subset = s1:s2
    cv.train = train[-subset,]
    cv.test = train[subset,]
    
    treeFit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + Child + SibSp
                    + Parch + Fare + Embarked + Title + Fam.Size + Wealth
                    + Fam.Id2 + Cabin.Letter, data=cv.train, importance=TRUE, ntree=2000)
    
    fit.pr <- predict(treeFit, cv.test, type="prob")[,2]
    fit.pred <- prediction(fit.pr, cv.test[,2])
    ##fit.perf <- performance(fit.pred, "prec", "rec")
    ##plot(fit.perf, main="PR Curve for Random Forest", col=2, lwd=2)
    
    fit.perf <- performance(fit.pred, "tpr", "fpr")
    #plot(fit.perf, main="ROC Curve for Random Forest", col=2, lwd=2)
    #abline(a=0, b=1, lwd=2, lty=2, col="gray")
    auc <- performance(fit.pred, "auc")
    auc <- unlist(slot(auc, "y.values"))
    errvect5[i] <- auc
    print(paste("AUC for fold: ", i, errvect5[i]))
    resultRForestCabin <- c(resultRForestCabin, errvect5[i])
}
```

```
## [1] "AUC for fold:  1 0.988290398126464"
## [1] "AUC for fold:  2 0.837068965517241"
## [1] "AUC for fold:  3 0.873353596757852"
## [1] "AUC for fold:  4 0.770897832817337"
## [1] "AUC for fold:  5 0.907754010695187"
## [1] "AUC for fold:  6 0.844350282485876"
## [1] "AUC for fold:  7 0.858887733887734"
## [1] "AUC for fold:  8 0.906028368794327"
## [1] "AUC for fold:  9 0.874304783092325"
## [1] "AUC for fold:  10 0.861299435028248"
```

```r
print(paste("Avg AUC: ", mean(errvect5)))
```

```
## [1] "Avg AUC:  0.872223540720259"
```

```r
t.test(resultRForestCabin - resultRForest)
```

```
## 
## 	One Sample t-test
## 
## data:  resultRForestCabin - resultRForest
## t = -0.1486, df = 9, p-value = 0.8851
## alternative hypothesis: true mean is not equal to 0
## 95 percent confidence interval:
##  -0.05558291  0.04872912
## sample estimates:
##    mean of x 
## -0.003426895
```


#### Round 10 submission  

Alright, let's try again.   


```r
treeFit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + Child + SibSp
                    + Parch + Fare + Embarked + Title + Fam.Size 
                    + Fam.Id2 + Cabin.Letter + Wealth, data=train, 
                    importance=TRUE, ntree=2000)
varImpPlot(treeFit)
```

![plot of chunk unnamed-chunk-45](figure/unnamed-chunk-45-1.png) 

```r
## glad to see Wealth, the new feature we introduced, is actually pretty important
Prediction <- predict(treeFit, test)
result10 <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(result10, file="result10.csv", row.names=FALSE)
```

#### Round 10 submission result  

Nope, dropped even lower -- 0.76555.  


#### Round 11 submission 

Remove some less important features.  


```r
treeFit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age+
                    Fare + Title + Fam.Size 
                     + Wealth, data=train, 
                    importance=TRUE, ntree=2000)
varImpPlot(treeFit)
```

![plot of chunk unnamed-chunk-46](figure/unnamed-chunk-46-1.png) 

```r
## glad to see Wealth, the new feature we introduced, is actually pretty important
Prediction <- predict(treeFit, test)
result11 <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(result11, file="result11.csv", row.names=FALSE)
```

#### Round 11 submission result  

Did not improve, at 0.78469.  

#### Round 12 submission 

Add Fam.Id with cforest.


```r
treeFit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age
                   + Fare + Title + Fam.Id + Wealth,
                   data=train, controls=cforest_unbiased(ntree=2000, mtry=3))
##varimp(treeFit)
Prediction <- predict(treeFit, test, OOB=TRUE, type="response")
result12 <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(result12, file="result12.csv", row.names=FALSE)
```

#### Round 12 submission result   

At 0.79904.  



```r
nTrees5.100 <- data.frame(ntrees=as.integer(), prob=as.numeric())

for (i in seq(5,100, by=10)) {
    ##result <- crossValidate(10, i)
    ##print(paste(i, " trees: ", result))
    tempData <- data.frame(ntrees=as.integer(), prob=as.numeric())
    tempData[1,1] <- i
    tempData[1, 2] <- crossValidate(i)
    nTrees5.100 <- rbind(nTrees5.100, tempData)
    print(nTrees5.100)
    
}
```

```
## Error in UseMethod("predict"): no applicable method for 'predict' applied to an object of class "c('double', 'numeric')"
```













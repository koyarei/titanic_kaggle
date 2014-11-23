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
##############
##combi[combi$Title %in% c("Capt", "Col", "Major", "Sir", "Dr"),]$Title <- "Sir"
##combi[combi$Title %in% c("Don", "Mr"),]$Title <- "Mr"
##combi[combi$Title %in% c("Dona", "Miss", "Mlle"),]$Title <- "Miss"
##combi[combi$Title %in% c("Mrs", "Mme", "Ms"),]$Title <- "Mrs"
##combi[combi$Title %in% c("Master", "Jonkheer"),]$Title <- "Master"
##combi[combi$Title %in% c("Lady", "the Countess"),]$Title <- "Lady"
################
combi$Fam.Size <- combi$Parch + combi$SibSp + 1
## create a new factor that is the traveler's last name.
combi$Last.Name <- as.character(combi$Name)

getLast.Name <- function(x) {
    Last.Name <- str_trim(strsplit(x, ",")[[1]][1])
    Last.Name
}

combi$Last.Name <- sapply(combi$Name, getLast.Name)

combi$Fam.Id <- paste0(combi$Fam.Size, combi$Last.Name)
##Fam.Id <- data.frame(table(combi$Fam.Id))
##Fam.Id <- Fam.Id[order(-Fam.Id[,2]),]

##Fam.IdSmall <- subset(Fam.Id, Freq < 3)
##combi[combi$Fam.Id %in% Fam.IdSmall$Var1,]$Fam.Id <- "Small"
combi$Fam.Id[combi$Fam.Size <= 2] <- 'Small'
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


## there are two missing values in Embarked. Let's examine this factor.
combi[order(combi$Ticket),c("Ticket", "Embarked")][40:63,]
## based on the ticket number, it's almost confirmed that these two passengers embarked from S.
combi[combi$Embarked == "",]$Embarked <- "S"
## There is one passenger with Fare missing. Let's examine that.
combi[is.na(combi$Fare),]
##  examine ticket prices similar to his ticket number
ticket <- data.frame(combi)
ticket$Ticket <- as.integer(as.character(combi$Ticket))
subset(ticket, Ticket > 2800 & Ticket < 4200)
## it's reasonable to assume that his ticket price should be around $8, but let's use the median of 3rd class fare
combi$Fare[which(is.na(combi$Fare))] <- median(combi[combi$Pclass ==3 & !is.na(combi$Fare),]$Fare)

combi$Fam.Id2 <- as.character(combi$Fam.Id)
combi$Fam.Id2[combi$Fam.Id2 != "Small"] <- "Big"
combi$Fam.Id2 <- factor(combi$Fam.Id2)

library(randomForest)
combi$Title <- as.factor(combi$Title)
combi$Last.Name <- as.factor(combi$Last.Name)

test$Survived <- as.integer(0)

combi$Cabin.Letter <- substr(combi$Cabin, 1,1)
combi$Cabin.Letter <- as.factor(as.character(combi$Cabin.Letter))
noCabin <- combi[combi$Cabin == "",]
hasCabin <- combi[combi$Cabin != "",]

cabinFit <- rpart(Cabin.Letter ~ Pclass + Sex + SibSp + Parch + Fare + Embarked
                  + Title, 
                  data=hasCabin)
##fancyRpartPlot(cabinFit)
##predict(cabinFit, noCabin, type="class")
combi[combi$Cabin == "",]$Cabin.Letter <- predict(cabinFit, noCabin, type="class")


combi$Wealth <- paste0(combi$Cabin.Letter, combi$Pclass)
combi$Wealth <- as.factor(combi$Wealth)


train <- combi[1:891,]
test <- combi[892:nrow(combi),]


################
treeFit <- randomForest(as.factor(Survived) ~  Age +
                            Pclass + Title 
                        + Wealth, data=train, samplesize=c(10,20),classwt=c(20,10),
                        importance=TRUE, ntree=2000)

treeFit <- rpart(Survived ~ Pclass + Sex + Age +  Fare
                 + Title + Fam.Id + Wealth, 
                 data=train, method="class",
                 control=rpart.control(minisplit=2))
Prediction <- predict(treeFit, test, type="class")



result16 <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(result16, file="result16.csv", row.names=FALSE)
### half and half

train <- train[sample(nrow(train)), ]
subset <- 1:(891/2)
cv.train <- train[subset,]
cv.test <- train[-subset,]

treeFit <- randomForest(as.factor(Survived) ~  Age + Fare + 
                            Pclass + Title 
                        + Wealth, data=cv.train, samplesize=c(10,20),classwt=c(20,10),
                        importance=TRUE, ntree=2000)
fit.pr <- predict(treeFit, cv.test, type="class")
fit.pred <- prediction(fit.pr, cv.test[,2])
pred.labels <- slot(fit.pred, "labels")



##  cross-validation: 

lowVec5 <- c()

for (j in 1:100) {
    
    
    train <- train[sample(nrow(train)), ]
    
    k <- 10
    n = floor(nrow(train)/k)
    errvect5 <- c()
    
    for(i in 1:10) {
        s1 = (i - 1) * n + 1
        s2 = i * n
        subset = s1:s2
        cv.train = train[-subset,]
        cv.test = train[subset,]
        
       # treeFit <- randomForest(as.factor(Survived) ~  Age + Fare +
        #                            Pclass + Title 
         #                       + Wealth, data=cv.train, samplesize=c(10,20),classwt=c(10,10),
          #                      importance=TRUE, ntree=500)
        
        treeFit <- rpart(Survived ~ Pclass + Sex + Age +  Fare + Embarked 
                     + Title + Fam.Id + Wealth, 
                     data=cv.train, method="class",
                     control=rpart.control(minisplit=3))
        
        fit.pr <- predict(treeFit, cv.test, type="class")
        ##fit.pred <- prediction(fit.pr, cv.test[,2])
        #print(nrow(subset(cv.test, Survived == fit.pr))/n)
        
        #fit.perf <- performance(fit.pred, "prec", "rec")
        #plot(fit.perf, main="PR Curve for Random Forest", col=2, lwd=2)
        
        
        
        #fit.perf <- performance(fit.pred, "tpr", "fpr")
        #plot(fit.perf, main="ROC Curve for Random Forest", col=2, lwd=2)
        #abline(a=0, b=1, lwd=2, lty=2, col="gray")
        
        #fit.perf <- performance(fit.pred, "err")
        #acc <- unlist(slot(fit.perf, "y.values"))
        acc <- nrow(subset(cv.test, Survived == fit.pr))/n
        
        #auc <- performance(fit.pred, "auc")
        #auc <- unlist(slot(auc, "y.values"))
        #errvect5[i] <- auc
        errvect5[i] <- acc
        #print(paste("AUC for fold: ", i, errvect5[i]))
        print(paste("Accuracy: ", i, errvect5[i]))
        #resultRForestCabin <- c(resultRForestCabin, errvect5[i])
    }
    #print(paste("Avg AUC: ", mean(errvect5)))
    print(paste("Avg ACC: ", min(errvect5)))
    lowVec5 <-c(lowVec5, min(errvect5))
}
print(paste("mean of Avg ACC: ", mean(lowVec5)))


## lowVec mean of lowest: 0.7610, samplezie(10,20), classwt=c(15,10)
##lowVec1 : samplesize(10,20), classwt=c(10,10)
##lowVec2: samplesize(10,20), classwt=c(10,15)
t.test(lowVec - lowVec1)
t.test(lowVec - lowVec2)
#rpart mean of lowest: 0.77116 Survived ~ Pclass + Sex + Age + Child + SibSp + Parch + Fare + Embarked
##+ Title + Fam.Size + Fam.Id, 

mean(lowVec3) + c(-1,1) * qnorm(.975) * sd(lowVec3)/sqrt(2000)
[1] 0.7683118 0.7701938
## take the mean of each fold accuracy instead of the lowest
mean(lowVec4) + c(-1,1) * qnorm(.975) * sd(lowVec4)/sqrt(100)
[1] 0.8313855 0.8328392


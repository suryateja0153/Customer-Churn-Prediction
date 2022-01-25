#Author: Suryateja Chalapati

#Importing Libraries
library(readxl)
library(data.table)
library(stargazer)
library(caret)
library(ROCR)

#Setting the Working Directory and Importing the Dataset
setwd("C:/Users/surya/Downloads")

tc <- read_excel("TelcoChurn.xlsx", sheet = "Data")
names(tc) <- tolower(colnames(tc))
attach(tc)

#NA Values Column-Wise & Pre-Processing
sapply(tc, function(x) sum(is.na(x)))
str(tc)

colSums(is.na(tc))
tc <- tc[complete.cases(tc), ]
str(tc)

tc$customerid <- NULL
tc$onlinebackup <- NULL
tc$deviceprotection <-NULL
tc$contract <- NULL
tc$onlinesecurity <- NULL
tc$streamingtv <- NULL
tc$paperlessbilling <- NULL
tc$paymentmethod <- NULL

#Feature Engineering
tc$churn <- ifelse(tc$churn == "Yes", 1, 0)
tc$churn <- as.factor(tc$churn)

phone_service <- tc[tc$phoneservice == "Yes",]
phone_service <- phone_service[, -c(6, 8:10)]

internet_service <- tc[tc$internetservice != "No",]
internet_service <- internet_service[, -c(6:7)]

both_services <- tc[(tc$phoneservice == "Yes" & tc$internetservice != "No"),]
both_services <- both_services[, -c(6)]

#Exploratory Analysis
table(tc$churn, tc$phoneservice)
table(tc$churn, tc$internetservice)

#Classification Models -------------------------------------------------
#Phone Service
set.seed(1024)
phone_sample = floor(0.75*nrow(phone_service))
index_ps <- sample(seq_len(nrow(phone_service)), size=phone_sample)
train_ps <- phone_service[index_ps,]
test_ps  <- phone_service[-index_ps,]

phone_logit <- glm(churn ~ gender + seniorcitizen + partner + dependents + tenure + multiplelines + 
                     monthlycharges + totalcharges, family = binomial (link = "logit"), data = train_ps)
str(test_ps)
test_xps <- test_ps[ , c(1:8)]
predlogit_phone <-predict(phone_logit, newdata=test_xps, type="response")
predlogit_phone <- ifelse(predlogit_phone>0.5, 1, 0)

#Internet Service
set.seed(1024)
internet_sample = floor(0.75*nrow(internet_service))
index_is <- sample(seq_len(nrow(internet_service)), size=internet_sample)
train_is <- internet_service[index_is,]
test_is  <- internet_service[-index_is,]

internet_logit <- glm(churn ~ gender + seniorcitizen + partner + dependents + tenure + techsupport + 
                        streamingmovies + monthlycharges + totalcharges, family = binomial (link = "logit"), data = train_is)
str(test_is)
test_xis <- test_is[ , c(1:10)]
predlogit_internet <-predict(internet_logit, newdata=test_xis, type="response")
predlogit_internet <- ifelse(predlogit_internet>0.5, 1, 0)

#Both Services
set.seed(1024)
both_sample = floor(0.75*nrow(both_services))
index_bs <- sample(seq_len(nrow(both_services)), size=both_sample)
train_bs <- both_services[index_bs,]
test_bs  <- both_services[-index_bs,]

both_logit <- glm(churn ~ gender + seniorcitizen + partner + dependents + tenure + multiplelines + 
                    techsupport + streamingmovies + monthlycharges + totalcharges, family = binomial (link = "logit"), data = train_bs)
str(test_bs)
test_xbs <- test_bs[ , c(1:11)]
predlogit_both <-predict(both_logit, newdata=test_xbs, type="response")
predlogit_both <- ifelse(predlogit_both>0.5, 1, 0)

#Stargazer
stargazer(phone_logit, internet_logit, both_logit, type='text', single.row = TRUE)

#Odds
odd_phone <- exp(phone_logit$coef)
odd_internet <- exp(internet_logit$coef)
odd_both <- exp(both_logit$coef)

e1 <- data.frame(odd_phone)
e2 <- data.frame(odd_internet)
e3 <- data.frame(odd_both)

#Probability
prob_phone <- exp(phone_logit$coef)/(1 + exp(phone_logit$coef))
prob_internet <- exp(internet_logit$coef)/(1 + exp(internet_logit$coef))
prob_both <- exp(both_logit$coef)/(1 + exp(both_logit$coef))

p1 <- data.frame(prob_phone)
p2 <- data.frame(prob_internet)
p3 <- data.frame(prob_both)

#Summary of Odds & Probability
df1 <- data.frame(e1, p1)
df2 <- data.frame(e2, p2)
df3 <- data.frame(e3, p3)

df1
df2
df3

#Model Fit & All Metrics -------------------------------------------------
#Phone Service
table(test_ps$churn, predlogit_phone)
ClassificationError <- mean(predlogit_phone != test_ps$churn)
print(paste("Accuracy = ", 1-ClassificationError))

#Confusion Matrix, Predicted Accuracy, Recall, Precision, F1-Score & AUC
cm <- confusionMatrix(as.factor(predlogit_phone), reference = test_ps$churn, mode = "everything", positive="1")
cm

#ROC
ps <- prediction(predlogit_phone, test_ps$churn)
ps_perf <- performance(ps, measure="tpr", x.measure="fpr")
plot(ps_perf)

#AUC
auc_ps <- performance(ps, measure="auc")
auc_ps <- auc_ps@y.values[[1]]
auc_ps

#Internet Service
table(test_is$churn, predlogit_internet)
ClassificationError <- mean(predlogit_internet != test_is$churn)
print(paste("Accuracy = ", 1-ClassificationError))

#Confusion Matrix, Predicted Accuracy, Recall, Precision, F1-Score & AUC
cm <- confusionMatrix(as.factor(predlogit_internet), reference = test_is$churn, mode = "everything", positive="1")
cm

#ROC
is <- prediction(predlogit_internet, test_is$churn)
is_perf <- performance(is, measure="tpr", x.measure="fpr")
plot(is_perf)

#AUC
auc_is <- performance(is, measure="auc")
auc_is <- auc_is@y.values[[1]]
auc_is

#Both Services
table(test_bs$churn, predlogit_both)
ClassificationError <- mean(predlogit_both != test_bs$churn)
print(paste("Accuracy = ", 1-ClassificationError))

#Confusion Matrix, Predicted Accuracy, Recall, Precision, F1-Score & AUC
cm <- confusionMatrix(as.factor(predlogit_both), reference = test_bs$churn, mode = "everything", positive="1")
cm

#ROC
bs <- prediction(predlogit_both, test_bs$churn)
bs_perf <- performance(bs, measure="tpr", x.measure="fpr")
plot(bs_perf)

#AUC
auc_bs <- performance(bs, measure="auc")
auc_bs <- auc_bs@y.values[[1]]
auc_bs
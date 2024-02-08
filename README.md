---
title: "Tutorial 4"
author: "Ruolei Zhu"
date: "2024-02-09"
output:
  html_document:
    df_print: paged
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## CRAN packages

```{r}
library(dplyr)       # Data manipulation (0.8.0.1)
library(fBasics)     # Summary statistics (3042.89)
library(corrplot)    # Correlations (0.84)
library(psych)       # Correlation p-values (1.8.12)
library(grf)         # Generalized random forests (0.10.2)
library(rpart)       # Classification and regression trees, or CART (4.1-13)
library(rpart.plot)  # Plotting trees (3.0.6)
library(treeClust)   # Predicting leaf position for causal trees (1.1-7)
library(car)         # linear hypothesis testing for causal tree (3.0-2)
library(remotes)    # Install packages from github (2.0.1)
library(ggplot2)     # general plotting tool (3.1.0)
library(glmnet)
```

## Non-CRAN packages

```{r}
# For causal trees (Athey and Imbens, 2016)  version 0.0
remotes::install_github('susanathey/causalTree') # Uncomment this to install the causalTree package
library(causalTree)
remotes::install_github('grf-labs/sufrep') # Uncomment this to install the sufrep package
library(sufrep)
```

## Loading the data

```{r}
df<-read.csv("coursework.csv") #Read in the data as a dataframe
```

## Cleaning the data

```{r}
df <- na.omit(df) #drop all rows with missing values values

df$postal_code <- factor(df$postal_code)
X<-model.matrix(~age + gender + socioecon + math_score + reading_score + gpa + free_meals + single_parent + postal_code - 1, df)

pretreat_names<-c('age','gender','socioecon','math_score','reading_score','gpa','free_meals','single_parent','postal_code')


train_fraction <- 0.80  # Use train_fraction % of the dataset to train our models
n <- dim(df)[1]
train_idx <- sample.int(n, replace=F, size=floor(n*train_fraction))
df_train <- df[train_idx,]
df_test <- df[-train_idx,]
X_train <- X[train_idx,]
X_test <- X[-train_idx,]
```

## Split the test and training samples into treated and untreated groups

```{Split the test and training samples into treated and untreated groups}
df_train_treated <- df_train[df_train$treat==1,]
df_test_treated <- df_test[df_test$treat==1,]
X_train_treated <- X_train[df_train$treat==1,]
X_test_treated <- X_test[df_test$treat==1,]

df_train_untreated <- df_train[df_train$treat==0,]
df_test_untreated <- df_test[df_test$treat==0,]
X_train_untreated <- X_train[df_train$treat==0,]
X_test_untreated <- X_test[df_test$treat==0,]
```

## ATE using Lasso and OLS

```{ATE using Lasso and OLS}
#first for treated individuals:
#train LASSO
y <- df_train_treated$math_outcome

cv_model <- cv.glmnet(X_train_treated, y, nfolds=5,alpha = 1)
best_lambda <- cv_model$lambda.min
best_model <- glmnet(X_train_treated, y, alpha = 1, lambda = best_lambda)

#let's look at the estimated coefficients
lasso_coef_treated<-coef(best_model)

#and let's evaluate the prediction accuracy on the validation sample
predicted_y<-predict(best_model, s = best_lambda, newx = X_test_treated)
testy<- df_test_treated$math_outcome
lasso_error_treated<-mean((predicted_y-testy)^2)

#let's train OLS and assess performance
ols_model <- lm(math_outcome ~ + age + gender + socioecon + math_score + reading_score + gpa + free_meals + single_parent, data = df_train_treated)
predicted_y<-predict(ols_model, df_test_treated[,pretreat_names])
ols_error_treated<-mean((predicted_y-testy)^2)

#predict treated potential outcomes in training data
LASSO_predicted_treat<-predict(best_model, s = best_lambda, newx = X_train)
OLS_predicted_treat<-predict(ols_model, df_test_treated[,pretreat_names])


#now for untreated individuals
#train LASSO
y <- df_train_untreated$math_outcome
cv_model <- cv.glmnet(X_train_untreated, y, nfolds=5,alpha = 1)
best_lambda <- cv_model$lambda.min
best_model <- glmnet(X_train_untreated, y, alpha = 1, lambda = best_lambda)

#let's look at the estimated coefficients
lasso_coef_untreated<-coef(best_model)

#and let's evaluate the prediction accuracy on the validation sample
predicted_y<-predict(best_model, s = best_lambda, newx = X_test_untreated)
testy<- df_test_untreated$math_outcome
lasso_error_untreated<-mean((predicted_y-testy)^2)

#let's train OLS and assess performance
ols_model <- lm(math_outcome ~ age + gender + socioecon + math_score + reading_score + gpa + free_meals + single_parent, data = df_train_untreated)
predicted_y<-predict(ols_model, df_test_untreated[,pretreat_names])
ols_error_untreated<-mean((predicted_y-testy)^2)


#predict untreated potential outcomes in training data
LASSO_predicted_untreat<-predict(best_model, s = best_lambda, newx = X_train)
OLS_predicted_untreat<-predict(ols_model, df_test_untreated[,pretreat_names])

#Treatment effect estimates from LASSO
tauLASSO_test<-LASSO_predicted_treat-LASSO_predicted_untreat
LASSOATE_test<-mean(tauLASSO_test)

#Treatment effect estimates from OLS
tauOLS_test<-OLS_predicted_treat-OLS_predicted_untreat
OLSATE_test<-mean(tauOLS_test)

```

## Calcua Forest

```{Calcua Forest}
#train the causal forest algorithm on the training data
cf <- causal_forest(
  X = as.matrix(X_train),
  Y = df_train$math_outcome,
  W = df_train$treat,
  num.trees=200)

#predict treatment effects for the training set
oob_pred <- predict(cf, estimate.variance=TRUE)
tauhat_cf_train <- oob_pred$predictions
tauhat_cf_train_se <- sqrt(oob_pred$variance.estimates)

#predict treatment effects for the test set
test_pred <- predict(cf, newdata=as.matrix(X_test), estimate.variance=TRUE)
tauhat_cf_test <- test_pred$predictions
tauhat_cf_test_se <- sqrt(test_pred$variance.estimates)

#estimate the training sample ATE
train_ATE<-mean(tauhat_cf_train)
high_math<-as.matrix((df_train$math_score)>(mean(df_train$math_score)))
train_ATE_highmath<-mean(tauhat_cf_train[high_math])
train_ATE_lowmath<-mean(tauhat_cf_train[-high_math])
train_ATE_lowsoc<-mean(tauhat_cf_train[df_train$socioecon==0])

#estimate the test sample ATE
test_ATE<-mean(tauhat_cf_test)
high_math<-as.matrix((df_test$math_score)>(mean(df_test$math_score)))
test_ATE_highmath<-mean(tauhat_cf_test[high_math])
test_ATE_lowmath<-mean(tauhat_cf_test[-high_math])
test_ATE_lowsoc<-mean(tauhat_cf_test[df_test$socioecon==0])
```

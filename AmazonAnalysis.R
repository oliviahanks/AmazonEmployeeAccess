#setwd('C:/Users/olivi/OneDrive/Documents/School2023/AmazonEmployeeAccess')

# Libraries
library(vroom)
library(tidymodels)
library(tidyverse)
library(embed)
library(discrim)


# Data
AEA_train <- vroom("./train.csv")
AEA_test <- vroom("./test.csv")

## Split data for CV
folds <- vfold_cv(AEA_train, v = 5, repeats=1)

# Columns
# ACTION - ACTION is 1 if the resource was approved, 0 if the resource was not
# RESOURCE - An ID for each resource
# MGR_ID - The EMPLOYEE ID of the manager of the current EMPLOYEE ID record; an employee may have only one manager at a time
# ROLE_ROLLUP_1 - Company role grouping category id 1 (e.g. US Engineering)
# ROLE_ROLLUP_2 - Company role grouping category id 2 (e.g. US Retail)
# ROLE_DEPTNAME - Company role department description (e.g. Retail)
# ROLE_TITLE - Company role business title description (e.g. Senior Engineering Retail Manager)
# ROLE_FAMILY_DESC - Company role family extended description (e.g. Retail Manager, Software Engineering)
# ROLE_FAMILY - Company role family description (e.g. Retail Manager)
# ROLE_CODE - Company role code; this code is unique to each role (e.g. Manager)

# Exploratory Plot 1
#DataExplorer::plot_histrograms(AEA_train) # histograms of all numerical variables
# Exploratory Plot 2
#DataExplorer::plot_correlation(AEA_train)



### Recipes #######################
###################################

my_recipe <- my_recipe_best

my_recipe_dummy <- recipe(ACTION ~ ., data=AEA_train) %>% # Set model formula and dataset
  step_mutate(ACTION = factor(ACTION), skip = TRUE) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_factor_predictors(), threshold = .01) %>%
  step_dummy(all_nominal_predictors()) #create dummy variables

my_recipe_target <- recipe(ACTION ~ ., data=AEA_train) %>% # Set model formula and dataset
  step_mutate(ACTION = factor(ACTION), skip = TRUE) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_factor_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())#target encoding

my_recipe_pcs <- recipe(ACTION ~ ., data=AEA_train) %>%
  step_mutate(ACTION = as.factor(ACTION), skip = TRUE) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold=0.001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% ### prev recipe
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=0.8) #Threshold is between 0 and 1

my_recipe_smote <- recipe(ACTION ~ ., data=AEA_train) %>%
  step_mutate(ACTION = as.factor(ACTION), skip = TRUE) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold=0.001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
  step_normalize((all_numeric_predictors())) #%>%
  step_pca(all_predictors(), threshold=0.8) %>% #Threshold is between 0 and 1
  step_smote(all_outcomes(), neighbors=2) # also step_upsample() and step_downsample()

my_recipe_best <- recipe(ACTION ~ ., data=AEA_train) %>%
  step_mutate(ACTION = as.factor(ACTION), skip = TRUE) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  #step_other(all_nominal_predictors(), threshold=0.001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
  step_normalize((all_numeric_predictors())) #%>%
  #step_pca(all_predictors(), threshold=0.8) %>% #Threshold is between 0 and 1
  #step_smote(all_outcomes(), neighbors=2) # also step_upsample() and step_downsample()
  
prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataSet
testbake <- bake(prepped_recipe, new_data=AEA_train)
# submit as predicted probabability of a 1






### Logistic Regression ###########
###################################

# model
my_mod_log <- logistic_reg() %>% #Type of model
set_engine("glm")

# workflow
logistic_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(my_mod_log) %>%
fit(data = AEA_train) # Fit the workflow

# test predictions
test_preds <- logistic_workflow %>%
  predict(new_data=AEA_test, type="prob") %>% # "class" or "prob" (see doc)
  rename(ACTION = .pred_1) %>%
  bind_cols(., AEA_test) %>%
  select(id, ACTION)

# save to csv
vroom_write(x=test_preds, file="./LogSubmission.csv", delim=",")



### Penalized Logistic Regression #
###################################

# model
my_mod_penlog <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

penlog_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_penlog)

## Grid of values to tune over
tuning_grid_penlog <- grid_regular(penalty(),
mixture(),
levels = 4) ## L^2 total tuning possibilities

## Run the CV
CV_results <- penlog_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_penlog,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- penlog_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=AEA_train)

test_preds <- final_wf %>%
  predict(new_data=AEA_test, type="prob") %>% # "class" or "prob" (see doc)
  rename(ACTION = .pred_1) %>%
  bind_cols(., AEA_test) %>%
  select(id, ACTION)

vroom_write(x=test_preds, file="./PenLogSubmission.csv", delim=",")



### RF Binary #####################
###################################

# model
my_mod_rfbin <- rand_forest(mtry = tune(),
                            min_n=tune(),
                            trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# workflow
rfbin_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod_rfbin)

## Set up grid of tuning values
RF_tuning_grid <- grid_regular(mtry(range = c(1, 5)), # num of columns used 
                               min_n(),
                               levels = 6)

## Set up K-fold CV
CV_results <- rfbin_workflow %>%
  tune_grid(resamples=folds,
            grid=RF_tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and predict
final_wf <- rfbin_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=AEA_train)

## Predict
test_preds <- final_wf %>%
  predict(new_data=AEA_test, type="prob") %>% # "class" or "prob" (see doc)
  rename(ACTION = .pred_1) %>%
  bind_cols(., AEA_test) %>%
  select(id, ACTION)

vroom_write(x=test_preds, file="./RFBinSubmission11.csv", delim=",")



### Naive Bayes ###################
###################################

## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
set_mode("classification") %>%
set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(nb_model)

## Tune smoothness and Laplace here
NB_tuning_grid <- grid_regular(Laplace(),
                               smoothness())

## Set up K-fold CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=NB_tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and predict
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=AEA_train)

## Predict
test_preds <- final_wf %>%
  predict(amazon_workflow, new_data=AEA_test, type="prob") %>% # "class" or "prob" (see doc)
  rename(ACTION = .pred_1) %>%
  bind_cols(., AEA_test) %>%
  select(id, ACTION)

vroom_write(x=test_preds, file="./RFBinSubmission.csv", delim=",")



# KNN Model########################
###################################

knn_mod <- nearest_neighbor(neighbors = tune()) %>%
  set_mode('classification') %>%
  set_engine('kknn')

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_mod)

tune_grid <- grid_regular(neighbors(), levels = 10)

## Set up K-fold CV
# use folds from above
CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tune_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and predict
final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=AEA_train)

## Predict
test_preds <- final_wf %>%
  predict(new_data=AEA_test, type="prob") %>% # "class" or "prob" (see doc)
  rename(ACTION = .pred_1) %>%
  bind_cols(., AEA_test) %>%
  select(id, ACTION)

vroom_write(x=test_preds, file="./KnnSubmission2.csv", delim=",")




### Naive Bayes with pcs ##########
###################################

my_recipe_pcs <- recipe(ACTION ~ ., data=AEA_train) %>%
  step_mutate(ACTION = as.factor(ACTION), skip = TRUE) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold=0.001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% ##prev recipe
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=0.8) #Threshold is between 0 and 1

## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
set_mode("classification") %>%
set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_wf <- workflow() %>%
add_recipe(my_recipe_pcs) %>%
add_model(nb_model)

## Tune smoothness and Laplace here
NB_tuning_grid <- grid_regular(Laplace(),
                               smoothness())

## Set up K-fold CV
# use folds from above
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=NB_tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and predict
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=AEA_train)

## Predict
test_preds <- final_wf %>%
  predict(new_data=AEA_test, type="prob") %>% # "class" or "prob" (see doc)
  rename(ACTION = .pred_1) %>%
  bind_cols(., AEA_test) %>%
  select(id, ACTION)

vroom_write(x=test_preds, file="./NBpcs8Submission.csv", delim=",")




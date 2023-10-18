setwd('C:/Users/olivi/OneDrive/Documents/School2023/AmazonEmployeeAccess')

# Libraries
library(vroom)
library(tidymodels)
library(embed)

# Data
AEA_train <- vroom("./train.csv")
AEA_test <- vroom("./test.csv")

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
DataExplorer::plot_histrograms(AEA_train) # histograms of all numerical variables
# Exploratory Plot 2
DataExplorer::plot_correlation(AEA_train)

# Recipe
my_recipe <- recipe(ACTION ~ ., data=AEA_train) %>% # Set model formula and dataset
  step_mutate(ACTION = factor(ACTION), skip = TRUE) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_factor_predictors(), threshold = .01) %>%
  step_dummy(all_nominal_predictors()) #create dummy variables
  
prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataSet
testbake <- bake(prepped_recipe, new_data=AEA_train)
# submit as predicted probabability of a one


my_mod <- logistic_reg() %>% #Type of model
set_engine("glm")

amazon_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(my_mod) %>%
fit(data = AEA_train) # Fit the workflow

amazon_predictions <- predict(amazon_workflow
                              , new_data=AEA_test
                              , type="prob") # "class" or "prob" (see doc)

test_preds <- amazon_workflow %>%
  predict(amazon_workflow, new_data=AEA_test, type="prob") %>% # "class" or "prob" (see doc)
  rename(ACTION = .pred_1) %>%
  bind_cols(., AEA_test) %>%
  select(id, ACTION)

#vroom_write(x=test_preds, file="./LogSubmission.csv", delim=",")


#################################
# Penalized Logistic Regression #
#################################
my_recipe_penlog <- recipe(ACTION ~ ., data=AEA_train) %>% # Set model formula and dataset
  step_mutate(ACTION = factor(ACTION), skip = TRUE) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_factor_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) #target encoding 

my_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
set_engine("glmnet")

amazon_workflow <- workflow() %>%
add_recipe(my_recipe_penlog) %>%
add_model(my_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
mixture(),
levels = 4) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(AEA_train, v = 10, repeats=1)

## Run the CV
CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

#####PENALIZED LOGISTIC IN R
## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=AEA_train)

## Predict
final_wf %>%
  predict(new_data = AEA_train, type="prob")

test_preds <- final_wf %>%
  predict(amazon_workflow, new_data=AEA_test, type="prob") %>% # "class" or "prob" (see doc)
  rename(ACTION = .pred_1) %>%
  bind_cols(., AEA_test) %>%
  select(id, ACTION)

vroom_write(x=test_preds, file="./PenLogSubmission.csv", delim=",")


############
# RF Binary#
############
my_mod <- rand_forest(mtry = tune(),
min_n=tune(),
trees=1000) %>%
set_engine("ranger") %>%
set_mode("classification")

## Create a workflow with model & recipe

## Set up grid of tuning values

## Set up K-fold CV

## Find best tuning parameters

## Finalize workflow and predict


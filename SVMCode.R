

# Libraries
library(vroom)
library(tidymodels)
library(tidyverse)
library(embed)
library(discrim)
library(kernlab)


AEA_train <- vroom("./train.csv")
AEA_test <- vroom("./test.csv")

folds <- vfold_cv(AEA_train, v = 5, repeats=1)

my_recipe_pcs <- recipe(ACTION ~ ., data=AEA_train) %>%
  step_mutate(ACTION = as.factor(ACTION), skip = TRUE) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold=0.001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% ##prev recipe
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=0.8) #Threshold is between 0 and 1

## SVM models
svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
set_mode("classification") %>%
set_engine("kernlab")

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
set_mode("classification") %>%
set_engine("kernlab")

svmLinear <- svm_linear(cost=tune()) %>% # set or tune
set_mode("classification") %>%
set_engine("kernlab")

## Fit or Tune Model HERE
radial_wf <- workflow() %>%
add_recipe(my_recipe_pcs) %>%
add_model(svmLinear)

## Tune cost
radial_tuning_grid <- grid_regular(cost())

## Set up K-fold CV
# use folds from above
CV_results <- radial_wf %>%
  tune_grid(resamples=folds,
            grid=radial_tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and predict
final_wf <- radial_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=AEA_train)

## Predict
test_preds <- final_wf %>%
  predict(new_data=AEA_test, type="prob") %>% # "class" or "prob" (see doc)
  rename(ACTION = .pred_1) %>%
  bind_cols(., AEA_test) %>%
  select(id, ACTION)

vroom_write(x=test_preds, file="./RadialSubmission.csv", delim=",")


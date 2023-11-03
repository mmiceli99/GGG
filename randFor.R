library(vroom)
library(tidyverse)
library(tidymodels)
library(embed)
library(doParallel)
library(themis)

#parallel::detectCores() #How many cores do I have?
cl <- makePSOCKcluster(13) # num_cores to use
registerDoParallel(cl)


train <- vroom("./train.csv")
test <- vroom("./test.csv")

train <- train %>%
  mutate(type = as.factor(type))
my_recipe <- recipe(type ~ ., data=train) %>%
  update_role(id, new_role = "id variable") %>%
  step_mutate_at(color, fn = factor) %>%# turn color into factors
  
  #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_dummy(color) # dummy variable encoding
  #step_lencode_mixed(color, outcome = vars(type)) #%>% #target encoding
#step_smote(all_outcomes(), k=2)
# also step_lencode_glm() and step_lencode_bayes()


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)


my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

GGG_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(mtry(range=c(1,10)),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
CV_results <- GGG_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best()

## Finalize the Workflow & fit it
final_wf <-
  GGG_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

## Predict
ama_predictions <- predict(final_wf, new_data=test, type='class') %>%
  rename(type=.pred_class) %>%
  mutate(id = test$id) %>%
  select(id, type)

vroom_write(x=ama_predictions, file="./RandFor.csv", delim=",")
stopCluster(cl)


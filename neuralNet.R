library(vroom)
library(tidyverse)
library(tidymodels)
library(embed)
library(doParallel)
library(themis)
library(nnet)
library(keras)
library(tensorflow)
#parallel::detectCores() #How many cores do I have?
#cl <- makePSOCKcluster(13) # num_cores to use
#registerDoParallel(cl)


train <- vroom("./train.csv")
test <- vroom("./test.csv")

train <- train %>%
  mutate(type = as.factor(type))

my_recipe <- recipe(type ~ ., data=train) %>%
  update_role(id, new_role = "id variable") %>%
  step_mutate_at(color, fn = factor) %>%# turn color into factors
  step_lencode_glm(color, outcome=vars(type)) %>%
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]
prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

nn_model <- mlp(hidden_units = tune(),
                epochs = 500) %>%
set_engine("nnet") %>% #verbose = 0 prints off less
  set_mode("classification")

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 10)),
                            levels=200)

folds <- vfold_cv(train, v = 6, repeats=1)

nn_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nn_model)

tuned_nn <- nn_workflow %>%
  tune_grid(resamples=folds,
            grid=nn_tuneGrid,
            metrics=metric_set(accuracy))

tuned_nn %>% collect_metrics() %>%
filter(.metric=="accuracy") #%>%
#ggplot(aes(x=hidden_units, y=mean)) + geom_line()



 #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- tuned_nn %>%
  select_best()

## Finalize the Workflow & fit it
final_wf <-
  nn_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

## Predict
ama_predictions <- predict(final_wf, new_data=test, type='class') %>%
  rename(type=.pred_class) %>%
  mutate(id = test$id) %>%
  select(id, type)

vroom_write(x=ama_predictions, file="./nerualNet.csv", delim=",")
#stopCluster(cl)


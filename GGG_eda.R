library(tidyverse)
library(vroom)
library(tidymodels)

train <- vroom('train.csv')
test <- vroom('test.csv')
missing <- vroom('trainWithMissingValues.csv')

boxplot(has_soul ~ type, data=train)
boxplot(bone_length ~ type, data=train)
boxplot(hair_length ~ type, data=train)
boxplot(rotting_flesh ~ type, data=train)

ggplot(data=train) +
  geom_bar(mapping = aes(x=color, fill=train$type), position = 'dodge')

ggplot(train) +
  geom_point(mapping=aes(x=hair_length, y = has_soul, col=type))

summary(missing)

#impute_mean
my_recipe <- recipe(type ~ ., data=missing) %>%
  step_impute_mean(hair_length) %>%
  step_impute_mean(rotting_flesh) %>%
  step_impute_mean(bone_length)
  
prep <- prep(my_recipe)
baked <- bake(prep, new_data = missing)

rmse_vec(train[is.na(missing)], baked[is.na(missing)])
#0.1526155

#impute_knn
my_recipe <- recipe(type ~ ., data=missing) %>%
  step_impute_bag(hair_length,  trees=1000) %>%
  step_impute_bag(rotting_flesh, trees=1000) %>%
  step_impute_bag(bone_length,  trees=1000)

prep <- prep(my_recipe)
baked <- bake(prep, new_data = missing)

rmse_vec(train[is.na(missing)], baked[is.na(missing)])
 







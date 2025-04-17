library(tidyverse)
library(lme4)
library(nnet)      # For multinomial logistic regression
library(glmnet)    # For LASSO (variable selection with collinearity)
library(caret)     # For preprocessing
library(dplyr)     # For data manipulation
library(data.table)
library(nnet)

#library(lmvar)

# load tidy features for chair in coca corpus
#h = read_csv("/Volumes/data_gabriella_chronis/workspace/acl_r_analyses/bert-base-uncased/coca/model_buchanan_layer_7.csv")
#h = read_csv("/Volumes/Macintosh HD/Users/gabriellachronis/Box Sync/src/feature_scrap/features/coca/toxic_feature_vectors_roberta_buchanan_layer7.csv")
h = read_csv("/Volumes/Macintosh HD/Users/gabriellachronis/Box Sync/src/feature_scrap/coca_human_feature_vectors_roberta_buchanan_layer7.csv")

toks = read_csv("/Volumes/data_gabriella_chronis/workspace/acl_metapragmatics/collected_tokens/coca/human.csv")

h = left_join(h, toks, by = c("token_id" = "...1"))

# we shouldnt be missing any sentences
any(is.na(h$sentence))

# there are <1000 sentences
h$token_id %>% unique() %>% length()

# there are 3981 features
h$feature %>% unique() %>% length()

# but we have no nan values
sum(is.na(h$predicted_value)) / 3981


h.zscore = h %>%
  group_by(feature) %>%
  mutate(zscore_predicted_value = scale(predicted_value))

# what are the average feature values in general?
h.avg = h.zscore %>% group_by(feature) %>%
  summarize(m = mean(zscore_predicted_value))


# what are the average feature values for each cluster?
h.sum = h.zscore %>% group_by(feature, cluster) %>%
  summarize(m = mean(zscore_predicted_value, na.rm=TRUE)) %>%
  pivot_wider(
    id_cols = c(feature),
    names_from = c(cluster),
    values_from = c(m)
  )


# what are sample sentences for each cluster?
h.sum.sents = h.zscore %>% 
  group_by(cluster) %>%
  slice_sample(n=5)


# create a one-hot encoded columns that have the value if in that cluster or NA
h.s =  pivot_wider(
  h.zscore,
  id_cols = c(token_id, feature),
  names_from = c(cluster),
  values_from = c(zscore_predicted_value)
)  %>%
  rename(
    cluster_0 = "0",
    cluster_1 = "1",
    cluster_2 = "2",
    cluster_3 = "3",
    cluster_4 = "4"
  )



# what are the most sense0-like features?
# averages all of the non-0 vectors together and then subtracts that from the average 0 vector              
h.s <- h.zscore %>%
  mutate(is_zero = cluster == 0) %>%
  group_by(feature, is_zero) %>%
  summarise(mean = mean(zscore_predicted_value, na.rm=TRUE)) %>%
  pivot_wider(
    id_cols = c(feature),
    names_from = c(is_zero),
    values_from = c(mean)
  ) %>%
  rename(
    cluster_0 = "FALSE",
    others = "TRUE",
  ) %>%
  mutate(zero.ness = cluster_0 - others)
h.sum$zero.ness <- h.s$zero.ness

h.sum %>%
  arrange(zero.ness)

# what are the most sense1-like features?
# averages all of the non-1 vectors together and then subtracts that from the average 0 vector              
h.s <- h.zscore %>%
  mutate(is_one = cluster == 1) %>%
  group_by(feature, is_one) %>%
  summarise(mean = mean(zscore_predicted_value, na.rm=TRUE)) %>%
  pivot_wider(
    id_cols = c(feature),
    names_from = c(is_one),
    values_from = c(mean)
  ) %>%
  rename(
    cluster_1 = "FALSE",
    others = "TRUE",
  ) %>%
  mutate(one.ness = cluster_1 - others)
h.sum$one.ness <- h.s$one.ness

h.sum %>%
  arrange(one.ness)

# what are the most sense2-like features?
# averages all of the non-2 vectors together and then subtracts that from the average 0 vector              
h.s <- h.zscore %>%
  mutate(is_two = cluster == 2) %>%
  group_by(feature, is_two) %>%
  summarise(mean = mean(zscore_predicted_value, na.rm=TRUE)) %>%
  pivot_wider(
    id_cols = c(feature),
    names_from = c(is_two),
    values_from = c(mean)
  ) %>%
  rename(
    cluster_2 = "FALSE",
    others = "TRUE",
  ) %>%
  mutate(two.ness = cluster_2 - others)
h.sum$two.ness <- h.s$two.ness
h.sum %>%
  arrange(two.ness)

# what are the most sense3-like features?
# averages all of the non-3 vectors together and then subtracts that from the average 0 vector              
h.s <- h.zscore %>%
  mutate(is_three = cluster == 3) %>%
  group_by(feature, is_three) %>%
  summarise(mean = mean(zscore_predicted_value, na.rm=TRUE)) %>%
  pivot_wider(
    id_cols = c(feature),
    names_from = c(is_three),
    values_from = c(mean)
  ) %>%
  rename(
    cluster_3 = "FALSE",
    others = "TRUE",
  ) %>%
  mutate(three.ness = cluster_3 - others)
h.sum$three.ness <- h.s$three.ness
h.sum %>%
  arrange(three.ness)


# what are the most sense4-like features?
# averages all of the non-4 vectors together and then subtracts that from the average 0 vector              
h.s <- h.zscore %>%
  mutate(is_four = cluster == 4) %>%
  group_by(feature, is_four) %>%
  summarise(mean = mean(zscore_predicted_value, na.rm=TRUE)) %>%
  pivot_wider(
    id_cols = c(feature),
    names_from = c(is_four),
    values_from = c(mean)
  ) %>%
  rename(
    cluster_4 = "FALSE",
    others = "TRUE",
  ) %>%
  mutate(four.ness = cluster_4 - others)
h.sum$four.ness <- h.s$four.ness
h.sum %>%
  arrange(four.ness)


###### prepare the data

h.zscore <- as.data.table(h.zscore)

# rename columns so they dont get lost when you pivot
h.zscore <- h.zscore %>% 
  rename(x_cluster = cluster) %>%
  rename(x_feature = feature) %>%
  rename(x_id = token_id) %>%
  rename(x_corpus_id = corpus_id) %>%
  rename(x_predicted_value = predicted_value) %>%
  rename(x_sentence = sentence) %>%
  rename(x_source = source) %>%
  rename(x_word = word)

# for very large dataset
h.wide <- dcast(h.zscore, x_id  + x_cluster + x_sentence ~ x_feature, value.var = "x_predicted_value")
# merge back in cluster - adds one more var
#h.wide <- merge(h.wide, unique(h.zscore[, .(x_id, x_cluster, x_sentence)]), by = "x_id", all.x = TRUE)

# Assume df is your dataset with 60 variables and a "Condition" column
# Convert condition to a factor if it's not already
# h.wide$x_source <- as.factor(h.wide$x_source)
h.wide$x_cluster <- as.factor(h.wide$x_cluster)

#Get column names starting with "x_"
cols_x <- h.wide %>% select(starts_with("x_")) %>% colnames()
print(cols_x)

non_numeric_cols <- names(h.wide)[sapply(h.wide, function(x) !is.numeric(x))]
print(non_numeric_cols)

# Compute correlation matrix
#cor_matrix <- cor(h.wide %>% select(-c(x_cluster, x_source)))

# Find and remove highly correlated variables (threshold = 0.9)
#high_corr <- findCorrelation(cor_matrix, cutoff = 0.9)
#df_reduced <- h.wide %>% select(-all_of(names(h.wide)[high_corr]))

###################################
###### which features are most distinctive for each cluster?
# let's do a multinomial regression analysis predicting the cluster based on feature value

just_feats <- h.wide %>% 
  select(-c(x_cluster, x_id, x_sentence))
cols_x <- just_feats %>% select(starts_with("x_")) %>% colnames()
                                
# Remove predictors with zero variance
# df_cleaned <- just_feats[, sapply(just_feats, function(x) length(unique(x)) > 1)]

#### Prepare data for LASSO with train and test
#X <- as.matrix(just_feats)  # Independent variables
#y <- h.wide$x_cluster  # Dependent variable

set.seed(42)  # Set seed for reproducibility

# Split data (80% for training, 20% for testing)
train_index <- sample(1:nrow(h.wide), size = 0.8 * nrow(h.wide))

# Create training and test datasets
train_data <- h.wide[train_index, ]
test_data <- h.wide[-train_index, ]

# Split into predictors (X) and target (y) for both training and test sets
X_train <- as.matrix(train_data %>% select(-c(x_cluster, x_id, x_sentence)))  # Train predictors
y_train <- train_data$x_cluster  # Train target

X_test <- as.matrix(test_data %>% select(-c(x_cluster, x_id, x_sentence)))  # Test predictors
y_test <- test_data$x_cluster  # Test target

# Perform LASSO multinomial regression
lasso_model <- cv.glmnet(X_train, 
                         y_train, 
                         family="multinomial", # multiple categories for outcome var
                         alpha=1, # alpha=1 is LASSO
                         type.multinomial = "grouped" # as per the posts here and here
                          # https://stackoverflow.com/questions/73519676/glmnet-for-feature-selection-when-the-number-of-classes-is-more-than-two-family
                          # https://stats.stackexchange.com/questions/423054/how-to-interpret-coefficients-of-a-multinomial-elastic-net-glmnet-regression
                         )  

# Get best lambda (penalty parameter)
best_lambda <- lasso_model$lambda.min



######### Extract and store selected variables and their coefficients
lasso_coefs <- coef(lasso_model, s = best_lambda)

# ger the row and column names as features and clusters
categories <- levels(train_data$x_cluster)# List the features
features <- colnames(X_train)
# Convert the coefficients into a data frame
coeff_df <- data.frame(as.matrix(lasso_coefs))

# Convert each sparse matrix to a regular matrix and store them
# We'll combine them into a single large data frame
coeff_df_list <- lapply(lasso_coefs, function(coef_matrix) {
  coef_matrix_dense <- as.matrix(coef_matrix)  # Convert to regular matrix
  return(coef_matrix_dense)
})
coeff_df <- do.call(cbind, coeff_df_list)
print(coeff_df)


selected_vars <- lapply(lasso_coefs, function(x) rownames(x)[x[,1] != 0])  # Non-zero coefficients
selected_vars <- unique(unlist(selected_vars))  # Flatten list

### get classification accuracy by running predictions

# Get predicted probabilities
predictions <- predict(lasso_model, newx = X_test, s = lasso_model$lambda.min, type = "response")
# For multinomial regression, predictions are probabilities, so pick the predicted class with max probability
pred_class <- apply(predictions, 1, which.max)  # Find the class with the max probability

# Calculate accuracy
accuracy <- mean(pred_class == as.numeric(y_test))
print(accuracy)
# 83 percent accuracy

###### compare to majority class baseline

# count the classes
table(y_train)
# Find the most frequent class (majority class)
majority_class <- names(sort(table(y_train), decreasing = TRUE))[1]

# Compute baseline accuracy (predicting the majority class for all test samples)
baseline_accuracy <- mean(majority_class == as.numeric(y_test))
print(paste("Baseline Accuracy:", baseline_accuracy))

########## calculate F1 score

# Compute confusion matrix
conf_matrix <- confusionMatrix(factor(pred_class), factor(as.numeric(y_test)))

# F1 score for each class (average over all classes)
f1_score <- conf_matrix$byClass[,"F1"]
print(paste("F1 Score:", f1_score))

# Macro-average F1 score
f1_macro <- mean(conf_matrix$byClass[, "F1"])
print(paste("Macro-Average F1 Score:", f1_macro))
# 84%

# Weighted Macro-average F1 score
f1_weighted <- weighted.mean(conf_matrix$byClass[, "F1"], conf_matrix$byClass[, "Prevalence"])
print(paste("Weighted-Average F1 Score:", f1_weighted))
# 84% 


######### get incorrect predictions
# Find incorrect predictions
incorrect_predictions <- pred_class != as.numeric(y_test)
# Extract the sentences corresponding to incorrect predictions
incorrect_sentences <- test_data[incorrect_predictions] %>%
  select(x_sentence)

####### what features matter here?

# Extract the coefficients for the best lambda (lambda.min)
cf.glmnet <- coef(lasso_model, s = "lambda.min")

cf.glmnet2 <-
  cf.glmnet %>%
  lapply(as.matrix) %>%
  Reduce(cbind, x = .) 
#  t() # %>%
#  as.data.frame() %>%
#  pivot_longer(cols=-id, names_to = "feature", values_to = "value")


# Convert to a matrix for easier manipulation
coeff_matrix <- as.matrix(coefficients)

# For each class, identify the top features (highest coefficients)
top_features <- list()

# Loop over each class (excluding the intercept, which is the first row)
for (class in 2:nrow(coeff_matrix)) {
  # Sort coefficients in descending order of their absolute value
  sorted_coeffs <- sort(abs(coeff_matrix[class,]), decreasing = TRUE)
  
  # Get the indices of the top features
  top_indices <- order(abs(coeff_matrix[class,]), decreasing = TRUE)[1:10]  # Top 10 features
  
  # Get the feature names and their corresponding coefficients
  top_features[[class - 1]] <- data.frame(
    feature = rownames(coeff_matrix)[top_indices],
    coefficient = coeff_matrix[class, top_indices]
  )
}


###### RUN MANY REGRESSIONS

############## run many regressions

feature_names <- names(h.wide)[!(names(h.wide) %in% c("x_word", "x_sentence", "x_cluster", "x_id"))]

h.wide$x_cluster <- as.factor(h.wide$x_cluster)


# Initialize a list to store models
models <- list()

# Loop through each independent variable
for (var in feature_names) {
  # Extract current predictor
  x <- d.wide[[var]]
  
  # Ensure x is numeric and has variation
  if (is.numeric(x) && length(unique(x)) > 1 && !all(is.na(x))) {
    # Define formula dynamically
    formula <- reformulate(var, response = "x_cluster")
    
    # Fit multinomial logistic regression model
    model <- multinom(formula, data = d.wide, trace = FALSE)
    
    # Store model
    models[[var]] <- model
  } else {
    cat("Skipping variable:", var, " (not numeric, constant, or missing values)\n")
  }
}


results <- list()
for (var in feature_names) {
  #X_var <- matrix(h.wide[[var]], ncol = 1)  # Convert to matrix format
  
  formula <- reformulate(var, response = "x_cluster")
  #model <- cv.glmnet(X_var, y, family = "multinomial", alpha = 1)  # LASSO with cross-validation
  multinom(formula, data = h.wide, trace = FALSE)
  print(summary_model)
  #p_value <- coef(summary_model)["wordimmigrants", "Pr(>|t|)"]  
  #coeff <- coef(summary_model)["wordimmigrants", "Estimate"]
  #r_squared <- summary_model$r.squared
  #adj_r_squared <- summary_model$adj.r.squared
  results[[var]] <- model  # Store the model for later analysis
}

#############

# Run independent regressions for each variable and store results
# higher coefficients mean that value increases from aliens to immigrants
results <- map_dfr(names(h.wide)[!(names(h.wide) %in% c("x_word", "x_sentence", "x_cluster"))], function(var) {
  model <- multinom(reformulate(var, response = "x_cluster"), data = h.wide)
  summary_model <- summary(model)
  
  print(summary_model)
  print(coef(summary_model))
  print(coef(summary_model))["1"]
        
  p_value <- coef(summary_model)["1", "Pr(>|t|)"]  
  coeff <- coef(summary_model)["1", "Estimate"]
  r_squared <- summary_model$r.squared
  adj_r_squared <- summary_model$adj.r.squared
  
  tibble(Variable = var, P_Value = p_value, coeff = coeff, r_squared = r_squared, adj_r_squared = adj_r_squared )
})


model <-  multinom(reformulate("tiara", response = "x_cluster"), data = h.wide)

# Adjust p-values for multiple comparisons (optional, e.g., Bonferroni or FDR)
results <- results %>%
  mutate(P_Adjusted = p.adjust(P_Value, method = "bonferroni"))


# Print significant results (e.g., p < 0.05)
significant_results <- results %>% filter(P_Adjusted < 0.05)
print(significant_results)

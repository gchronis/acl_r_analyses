library(tidyverse)
library(lme4)
library(nnet)      # For multinomial logistic regression
library(glmnet)    # For LASSO (variable selection with collinearity)
library(caret)     # For preprocessing
library(dplyr)     # For data manipulation
library(data.table)
library(nnet)
library(corrplot) # For easy heatmap plotting
library(lsa) # for cosine


# load tidy features for word in ACL corpus
acl = read_csv("/Volumes/Macintosh HD/Users/gabriellachronis/Box Sync/src/feature_scrap/acl_human_feature_vectors_roberta_buchanan_layer7.csv")
coca = read_csv("/Volumes/Macintosh HD/Users/gabriellachronis/Box Sync/src/feature_scrap/coca_human_feature_vectors_roberta_buchanan_layer7.csv")
toks_acl = read_csv("/Volumes/data_gabriella_chronis/workspace/acl_metapragmatics/collected_tokens/acl/human.csv")
toks_coca = read_csv("/Volumes/data_gabriella_chronis/workspace/acl_metapragmatics/collected_tokens/coca/human.csv")

# dont need these until you start using the separate toks datatsets
acl_df = left_join(acl, toks_acl, by = c("token_id" = "...1"))
coca_df = left_join(coca, toks_coca, by = c("token_id" = "...1"))

# join both data sources
h <- rbind(acl_df, coca_df)
rm(acl)
rm(coca)

h$token_id <- paste0(h$source, h$token_id)


# there are <1000 ids (bc duplicates)
h$token_id %>% unique() %>% length()


########## visualize clusters

# get summary table for bubble size
h.count = h %>%
  group_by(token_id) %>%
  slice(1) %>% # take first row only
  group_by(source, cluster) %>%
  count()


cat_levels <- rev(unique(h.count$cluster))  # reverse for top-to-bottom order


# Create x-position to separate corpora
h.count <- h.count %>%
  mutate(
    cluster = factor(cluster),
    x = as.numeric(factor(cluster)),
    x = ifelse(source == "acl", x - 0.1, x + 0.1),
    y = as.numeric(factor(cluster, levels = cat_levels))  # higher on top  )
  )

    
ggplot(h.count, aes(x = x, y = y, size = n, fill = source)) +
  geom_point(shape = 21, color = "black", alpha = 0.7) +
  geom_text(aes(label = cluster), vjust = -1, size = 3) +  # <- Label bubbles
  scale_size_area(max_size = 20) +  # Make size proportional to area
  scale_x_continuous(breaks = 1:5, labels = unique(h.count$cluster)) +
  theme_minimal() +
  labs(x = "Category", y = "", size = "Sample Count") +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid.major.y = element_blank()
  )

# xscore feature predictions
h <- h %>%
  group_by(feature) %>%
  mutate(zscore_predicted_value = scale(predicted_value))


########## first do this just by averaging over feature_vectors

# get average feature vector for each dataset-cluster combo
h.avg <- h %>%
  mutate(source_cluster = paste(source, cluster, sep = "_")) %>%
  group_by(feature, source_cluster) %>%
  summarize(m = mean(zscore_predicted_value)) %>%
  pivot_wider(names_from=source_cluster, values_from=m)


# run a spearman rank correlations and store results in a matrix
df_without_names <- h.avg[, !names(h.avg) %in% "feature"] %>%
  as.matrix()
cor_matrix <- cor(df_without_names, method = "spearman") # spearman
#cor_matrix <- cor(df_without_names, method = "pearson")
#cor_matrix <- cosine(df_without_names) # cosine similarity

# Plot the correlation matrix as a heatmap
corrplot(cor_matrix, method = "color", 
         col = colorRampPalette(c("blue", "white", "red"))(200), 
         type = "full", # Only show upper triangle
         order = "original", # Hierarchical clustering of variables
         tl.cex = 0.8, # Label text size
         tl.col = "black") # Label color



################ Now get data in terms of distinctive features and check that correlation

# for each cluster/source, we want a ranked list of features that distinguishes the cluster from the others in that source


#### now use this correlation matrix to size and label the arrows in our original bubble chart


# get the transition matrix
# Matrix of values: from coca (rows) to acl (cols)
transitions <- as.data.frame(cor_matrix)
#transitions <- transitions %>%
#  rownames_to_column("rownames") %>%
#  filter(!grepl("^acl", rownames)) %>%  # Remove rows starting with "acl"
#  select(-starts_with("coca")) %>%  # Remove columns starting with "coca"
#  column_to_rownames("rownames")  # Convert back to row names if needed

# Convert to tidy data frame for plotting
edges <- as.data.frame(as.table(cor_matrix)) %>%
  rename(from = Var1, to = Var2, value = Freq)


categories = levels(h.count$cluster)

# Create position map
cat_levels <- rev(categories)
positions <- data.frame(
  category = categories,
  y = as.numeric(factor(categories, levels = cat_levels))
)

# Merge positions for source and target
edges <- edges %>%
  left_join(positions, by = c("from" = "category")) %>%
  rename(y_start = y) %>%
  left_join(positions, by = c("to" = "category")) %>%
  rename(y_end = y) %>%
  mutate(x_start = 1, x_end = 2)

ggplot(h.count, aes(x = x, y = y, size = n, fill = source)) +
  # bubbles
  geom_point(
      shape = 21, 
      color = "black", 
      alpha = 0.7) +
  # labels
  geom_text(
      aes(label = cluster), 
      vjust = -1, 
      size = 3) +  # <- Label bubbles
  scale_size_area(max_size = 20) +  # Make size proportional to area
  scale_x_continuous(breaks = 1:5, labels = unique(h.count$cluster)) +
  theme_minimal() +
  labs(x = "Category", y = "", size = "Sample Count") +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid.major.y = element_blank()
  ) +
  # Arrows
  geom_curve(
    data = edges,
    aes(x = x_start, y = y_start, xend = x_end, yend = y_end, linewidth = value),
    curvature = 0.3,
    arrow = arrow(length = unit(0.15, "inches")),
    color = "gray50",
    alpha = 0.7,
    inherit.aes = FALSE
  ) 




########### EXAMPLE

# Categories
categories <- c("A", "B", "C", "D", "E")
n_cat <- length(categories)

# Matrix of values: from Pop1 (rows) to Pop2 (cols)
transition_matrix <- matrix(
  sample(1:10, n_cat * n_cat, replace = TRUE),
  nrow = n_cat,
  dimnames = list(from = categories, to = categories)
)

# Convert to tidy data frame for plotting
edges <- as.data.frame(as.table(transition_matrix)) %>%
  rename(from = from, to = to, value = Freq)

# Create position map
cat_levels <- rev(categories)
positions <- data.frame(
  category = categories,
  y = as.numeric(factor(categories, levels = cat_levels))
)

# Merge positions for source and target
edges <- edges %>%
  left_join(positions, by = c("from" = "category")) %>%
  rename(y_start = y) %>%
  left_join(positions, by = c("to" = "category")) %>%
  rename(y_end = y) %>%
  mutate(x_start = 1, x_end = 2)

# Example bubble data (from earlier)
df <- data.frame(
  category = rep(categories, times = 2),
  population = rep(c("Pop1", "Pop2"), each = n_cat),
  count = sample(5:25, n_cat * 2, replace = TRUE)
) %>%
  mutate(
    y = as.numeric(factor(category, levels = cat_levels)),
    x = ifelse(population == "Pop1", 1, 2)
  )


ggplot() +
  # Arrows first
  geom_curve(
    data = edges,
    aes(x = x_start, y = y_start, xend = x_end, yend = y_end, size = value),
    curvature = 0.0,
    arrow = arrow(length = unit(0.15, "inches")),
    color = "gray50",
    alpha = 0.7
  ) +
  # Bubbles
  geom_point(
    data = df,
    aes(x = x, y = y, size = count, fill = population),
    shape = 21,
    color = "black",
    alpha = 0.8
  ) +
  # Labels
  geom_text(
    data = df,
    aes(x = x, y = y, label = category),
    color = "black",
    size = 3,
    vjust = 0.4
  ) +
  scale_x_continuous(breaks = c(1, 2), labels = c("Pop1", "Pop2")) +
  scale_y_continuous(breaks = 1:n_cat, labels = rev(categories)) +
  scale_size_area(max_size = 20) +
  theme_minimal() +
  labs(x = "Population", y = "Category", size = "Count / Flow") +
  theme(
    panel.grid.minor = element_blank()
  )

library(tidyverse)
library(lme4)
#library(lmvar)
library(data.table)
library(dplyr)

# load tidy features for word in ACL corpus
acl = read_csv("/Volumes/Macintosh HD/Users/gabriellachronis/Box Sync/src/feature_scrap/features/acl/human_feature_vectors_roberta_buchanan_layer7.txt")
coca = read_csv("/Volumes/Macintosh HD/Users/gabriellachronis/Box Sync/src/feature_scrap/features/coca/human_feature_vectors_roberta_buchanan_layer7.txt")
toks_acl = read_csv("/Volumes/data_gabriella_chronis/workspace/acl_metapragmatics/collected_tokens/acl/human.csv")
toks_coca = read_csv("/Volumes/data_gabriella_chronis/workspace/acl_metapragmatics/collected_tokens/coca/human.csv")

# dont need these until you start using the separate toks datatsets
acl_df = left_join(acl, toks_acl, by = c("sent" = "sentence"))
coca_df = left_join(coca, toks_coca, by = c("sent" = "sentence"))

# join both datat sources
h <- rbind(acl_df, coca_df)
rm(acl)
rm(coca)
  

any(is.na(h$sent))

# there are <1000 ids
h$id %>% unique() %>% length()

# there are 3981 features
h$feature %>% unique() %>% length()

# but we have no nan values
sum(is.na(h$predicted_value)) / 3981


h.zscore = h %>%
  group_by(feature) %>%
  mutate(predicted_value = (predicted_value - mean(predicted_value)) / sd(predicted_value))

# Convert to data.table
h.zscore <- as.data.table(h.zscore)
h.zscore$id <- paste0(h.zscore$source, h.zscore$id)
# there are ~2000 sentences
h.zscore$id %>% unique() %>% length()

# what are the average feature values for each data source?
h.sum = h.zscore %>% group_by(source, feature) %>%
  summarize(m = mean(predicted_value)) %>%
  pivot_wider(
    id_cols = c(feature),
    names_from = c(source),
    values_from = c(m)
  )


# what are the average feature values for each data source?
h.by_cluster = h.zscore %>% 
  mutate(group = paste(source, cluster, sep="_")) %>%
  group_by(group, feature) %>%
  summarize(m = mean(predicted_value)) %>%
  pivot_wider(
    id_cols = c(feature),
    names_from = c(group),
    values_from = c(m)
  )


# what are sample sentences for each source?
h.sum.sents = h.zscore %>% 
  group_by(source, cluster) %>%
  slice_sample(n=5) %>%
  select(source, cluster, sent)


# create a one-hot encoded columns that have the value if in that cluster or NA
h.s =  pivot_wider(
  h.zscore,
  id_cols = c(id, feature),
  names_from = c(source),
  values_from = c(predicted_value)
) 



# what are the most sense0-like features?
# averages all of the non-0 vectors together and then subtracts that from the average 0 vector              
h.s <- h.zscore %>%
  mutate(is_zero = cluster == 0) %>%
  group_by(feature, is_zero) %>%
  summarise(mean = mean(predicted_value, na.rm=TRUE)) %>%
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
  summarise(mean = mean(predicted_value, na.rm=TRUE)) %>%
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
  summarise(mean = mean(predicted_value, na.rm=TRUE)) %>%
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
  summarise(mean = mean(predicted_value, na.rm=TRUE)) %>%
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
  summarise(mean = mean(predicted_value, na.rm=TRUE)) %>%
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



######### reshape the data

# rename columns so they dont get lost when you pivot
h.zscore <- h.zscore %>% 
  rename(x_cluster = cluster) %>%
  rename(x_feature = feature) %>%
  rename(x_id = id) %>%
  rename(x_predicted_value = predicted_value) %>%
  rename(x_sent = sent) %>%
  rename(x_source = source) %>%
  rename(x_word = word)


# Pivot the data from long to wide format
#d.wide <- h %>%
#  pivot_wider(names_from = feature, values_from = predicted_value)



# for very large dataset
d.wide <- dcast(h.zscore, x_id + x_cluster + x_sent + x_source ~ x_feature, value.var = "x_predicted_value")

# Convert condition to a factor if it's not already
d.wide$x_source <- as.factor(d.wide$x_source)
d.wide$x_cluster <- as.factor(d.wide$x_cluster)


print(unique(d.wide$x_source))  # See what levels exist for 'source' var
d.wide$x_source <- relevel(d.wide$x_source, ref = "acl")  # Set "aliens" as the reference level if needed


############### run many regressions over the sources in general

# Run independent regressions for each variable and store results
# higher coefficients mean that value increases from aliens to immigrants
results <- map_dfr(names(d.wide)[!(names(d.wide) %in% c("x_word", "x_sentence", "x_cluster", "x_id", "x_source"))], function(var) {
  model <- lm(reformulate("x_source", response = var), data = d.wide)
  summary_model <- summary(model)
  print(summary_model)
  p_value <- coef(summary_model)["sourcecoca", "Pr(>|t|)"]  
  coeff <- coef(summary_model)["sourcecoca", "Estimate"]
  r_squared <- summary_model$r.squared
  adj_r_squared <- summary_model$adj.r.squared
  
  tibble(Variable = var, P_Value = p_value, coeff = coeff, r_squared = r_squared, adj_r_squared = adj_r_squared )
})

# Adjust p-values for multiple comparisons (optional, e.g., Bonferroni or FDR)
results <- results %>%
  mutate(P_Adjusted = p.adjust(P_Value, method = "bonferroni"))


# Print significant results (e.g., p < 0.05)
significant_results <- results %>% filter(P_Adjusted < 0.05)
print(significant_results)




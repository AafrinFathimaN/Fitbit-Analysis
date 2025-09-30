# ================================================
# Fitbit Sleep & Activity Analysis
# ================================================

# 1. Objective
# Analyze Fitbit activity and sleep data to identify patterns in users' activity and sleep habits,
# cluster users based on activity-sleep behavior, and predict sleep quality (Good/Poor) using classification models.

# ================================================
# 2. Data Acquisition
# ================================================
# Install required packages only if not already installed
packages <- c("tidyverse", "cluster", "factoextra", "corrplot", "caret", "themis", "rpart", "rpart.plot", "ggplot2")
installed <- packages %in% rownames(installed.packages())
if(any(!installed)) install.packages(packages[!installed])

# Load libraries
library(tidyverse)
library(cluster)
library(factoextra)
library(corrplot)
library(caret)
library(themis)
library(rpart)
library(rpart.plot)
library(ggplot2)

# Read datasets
activity <- read_csv("C:/Users/91754/Downloads/sleepDay_merged.csv")
sleep <- read_csv("C:/Users/91754/Downloads/dailyActivity_merged.csv")

# ================================================
# 3. Data Cleaning & Preprocessing
# ================================================
# Merge datasets by Id
fitbit <- merge(activity, sleep, by = "Id")

# Select relevant features
fitbit <- fitbit %>%
  select(Id, TotalSteps, Calories, VeryActiveMinutes, SedentaryMinutes, TotalMinutesAsleep, TotalTimeInBed)

# Remove duplicates
fitbit <- distinct(fitbit)

# Handle outliers and invalid values
fitbit <- fitbit %>%
  filter(TotalSteps > 0, TotalSteps < 50000,
         TotalMinutesAsleep > 0, TotalMinutesAsleep < 1000)

# Scale data for clustering (exclude Id)
fitbit_scaled <- scale(fitbit %>% select(-Id))

# ================================================
# 4. Exploratory Data Analysis (EDA)
# ================================================
# Summary statistics
summary(fitbit)

# Histograms
hist(fitbit$TotalSteps, main="Steps Distribution", col="skyblue")
hist(fitbit$TotalMinutesAsleep, main="Sleep Duration Distribution", col="lightgreen")

# Correlation heatmap
corrplot(cor(fitbit %>% select(-Id)), method = "color", tl.cex = 0.8)

# ================================================
# 5. Model Building: Clustering
# ================================================
# Determine optimal number of clusters
fviz_nbclust(fitbit_scaled, kmeans, method = "wss")       # Elbow
fviz_nbclust(fitbit_scaled, kmeans, method = "silhouette") # Silhouette

# Choose 3 clusters (based on visual inspection)
set.seed(123)
k3 <- kmeans(fitbit_scaled, centers = 3, nstart = 25)

# Add cluster labels
fitbit$Cluster <- factor(k3$cluster,
                         levels = c(1,2,3),
                         labels = c("Sedentary Sleepers",
                                    "Overworked Night Owls",
                                    "Active Resters"))

# Cluster visualization
fviz_cluster(k3, data = fitbit_scaled, geom = "point", ellipse.type = "convex")

# Boxplots by cluster
ggplot(fitbit, aes(x = Cluster, y = TotalSteps, fill = Cluster)) +
  geom_boxplot() +
  labs(title = "Total Steps Distribution by Cluster")
ggplot(fitbit, aes(x = Cluster, y = TotalMinutesAsleep, fill = Cluster)) +
  geom_boxplot() +
  labs(title = "Sleep Duration Distribution by Cluster")

# Cluster summary table
cluster_summary <- aggregate(. ~ Cluster, data = fitbit %>% select(-Id), mean)
print(cluster_summary)

# ================================================
# 6. Model Building: Classification (Sleep Quality)
# ================================================
# Define sleep quality
fitbit <- fitbit %>%
  mutate(SleepEfficiency = (TotalMinutesAsleep / TotalTimeInBed) * 100,
         SleepQuality = ifelse(SleepEfficiency >= 85, "Good", "Poor"))

fitbit$SleepQuality <- factor(fitbit$SleepQuality, levels = c("Poor", "Good"))

# Train-test split
set.seed(123)
train_index <- sample(1:nrow(fitbit), 0.7 * nrow(fitbit))
train <- fitbit[train_index, ]
test <- fitbit[-train_index, ]

# Logistic Regression
log_model <- glm(SleepQuality ~ TotalSteps + Calories + VeryActiveMinutes + SedentaryMinutes,
                 data = train, family = "binomial")

# Predictions
pred_probs <- predict(log_model, test, type = "response")
pred_labels <- ifelse(pred_probs > 0.5, "Good", "Poor")

# Confusion matrix
cm <- table(Predicted = pred_labels, Actual = test$SleepQuality)
print(cm)

# ROC Curve
library(pROC)
roc_obj <- roc(test$SleepQuality, as.numeric(pred_probs))
plot(roc_obj, col = "blue", main = "ROC Curve for Sleep Quality")
auc(roc_obj)

# ================================================
# 7. Cross-validation with SMOTE
# ================================================
ctrl_smote <- trainControl(method = "cv", number = 10,
                           sampling = "smote",
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

set.seed(123)
cv_model_smote <- train(SleepQuality ~ TotalSteps + Calories + VeryActiveMinutes + SedentaryMinutes,
                        data = train,
                        method = "glm",
                        family = "binomial",
                        trControl = ctrl_smote,
                        metric = "ROC")

pred_probs_smote <- predict(cv_model_smote, newdata = test, type = "prob")[, "Good"]
pred_labels_smote <- ifelse(pred_probs_smote > 0.5, "Good", "Poor")

cm_smote <- table(Predicted = pred_labels_smote, Actual = test$SleepQuality)
print(cm_smote)

roc_smote <- roc(test$SleepQuality, as.numeric(pred_probs_smote))
plot(roc_smote, col = "red", main = "ROC Curve After SMOTE")
auc(roc_smote)

# Metrics comparison
metrics <- data.frame(
  Model = c("Original", "SMOTE"),
  Accuracy = c(sum(diag(cm))/sum(cm), sum(diag(cm_smote))/sum(cm_smote)),
  Precision = c(cm[2,2]/sum(cm[2,]), cm_smote[2,2]/sum(cm_smote[2,])),
  Recall = c(cm[2,2]/sum(cm[,2]), cm_smote[2,2]/sum(cm_smote[,2])),
  F1 = c(2*(cm[2,2]/sum(cm[2,]))*(cm[2,2]/sum(cm[,2]))/( (cm[2,2]/sum(cm[2,])) + (cm[2,2]/sum(cm[,2])) ),
         2*(cm_smote[2,2]/sum(cm_smote[2,]))*(cm_smote[2,2]/sum(cm_smote[,2]))/((cm_smote[2,2]/sum(cm_smote[2,])) + (cm_smote[2,2]/sum(cm_smote[,2])))),
  AUC = c(auc(roc_obj), auc(roc_smote))
)

print(metrics)

# ================================================
# 8. Decision Tree Visualization
# ================================================
tree_model <- rpart(SleepQuality ~ TotalSteps + Calories + VeryActiveMinutes + SedentaryMinutes,
                    data = fitbit, method = "class")
rpart.plot(tree_model, main = "Decision Tree for Sleep Quality")

# ================================================
# 9. Export for Power BI
# ================================================
write.csv(fitbit, "fitbit_clusters.csv", row.names = FALSE)
write.csv(metrics, "SleepQuality_Comparison.csv", row.names = FALSE)
write.csv(cluster_summary, "Cluster_Summary.csv", row.names = FALSE)

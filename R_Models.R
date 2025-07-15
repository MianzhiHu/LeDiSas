library(ggplot2)
library(lme4)
library(lmerTest)
library(effects)
library(ggplot2)
library(dplyr)
library(mgcv)
library(sjPlot)
library(segmented)
library(survival)
library(changepoint)
library(car)
library(emmeans)

# ==============================================================================
# Read the data
# ==============================================================================
hybrid_data <- read.csv("C:/Users/zuire/PycharmProjects/LeDiSas/LeSaS1/Data/hybrid_delta_delta.csv")
hybrid_data_mv <- read.csv("C:/Users/zuire/PycharmProjects/LeDiSas/LeSaS1/Data/hybrid_decay_delta.csv")
hybrid_data$Group <- factor(hybrid_data$Group, levels = c(1, 2), 
                            labels = c('OptHighReward', 'OptLowReward'))
hybrid_data_mv$Group <- factor(hybrid_data_mv$Group, levels = c(1, 2), 
                            labels = c('OptHighReward', 'OptLowReward'))
# ==============================================================================
# General Linear Models
# ==============================================================================
glm_model <- glm(Optimal_Choice ~ Group + Weight + t + alpha + RT0Diff,
             data=hybrid_data)
summary(glm_model)
plot(allEffects(glm_model))

glm_model <- glm(Optimal_Choice ~ Group * alpha,
                 data=hybrid_data)
summary(glm_model)
plot(allEffects(glm_model))

# ==============================================================================
# Linear Mixed-Effects Models
# ==============================================================================
mixed_effect <- lmer(optimal_percentage ~ Group + window_id + Weight +  (1|participant_id),
                     data = hybrid_data_mv)

summary(mixed_effect)
anova(mixed_effect)
plot(allEffects(mixed_effect))

mixed_effect <- lmer(Weight ~ Group * poly(window_id, 2) +  (1|participant_id),
                     data = hybrid_data_mv)

summary(mixed_effect)
anova(mixed_effect)
plot(allEffects(mixed_effect))

# ==============================================================================
# Read the data
# ==============================================================================
modeled_data <- read.csv("C:/Users/zuire/PycharmProjects/LeDiSas/LeSaS1/Data/data_clean_model.csv")
modeled_data$Group <- factor(modeled_data$Group, levels = c(1, 2), 
                            labels = c('OptHighReward', 'OptLowReward'))
# modeled_data$model_type <- factor(modeled_data$model_type, 
#                                   levels = c('value_based', 'RT_based', 'RPUT_based'))
modeled_data$model_type <- factor(modeled_data$model_type, 
                                  levels = c('RT_based', 'value_based', 'RPUT_based'))
modeled_data <- modeled_data %>%
  filter(Group == 'OptHighReward')

mixed_effect <- lmer(Optimal_Choice ~  Block + model_type + (1|SubNo),
                     data = modeled_data)

mixed_effect <- lmer(Optimal_Choice ~  Block + Group * model_type + (1|SubNo),
                     data = modeled_data)
summary(mixed_effect)
anova(mixed_effect)
plot(allEffects(mixed_effect))

# Get estimated marginal means
emm <- emmeans(mixed_effect, ~ Group * model_type)

# View the table
summary(emm)

# Contrast between groups within RT-based participants
contrast(emm, method = "pairwise", by = "model_type", adjust = "none")
# ==============================================================================
# Summary
# ==============================================================================
summary_data <- read.csv("C:/Users/zuire/PycharmProjects/LeDiSas/LeSaS1/Data/summary.csv")
summary_data$Group <- factor(summary_data$Group, levels = c(1, 2), 
                             labels = c('OptHighReward', 'OptLowReward'))
summary_data$model_type <- factor(summary_data$model_type)

mixed_effect <- glm(Optimal_Choice ~  Group * model_type, data = summary_data)
summary(mixed_effect)
anova(mixed_effect)
plot(allEffects(mixed_effect))

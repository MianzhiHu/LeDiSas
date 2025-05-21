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

# ==============================================================================
# Read the data
# ==============================================================================
hybrid_data <- read.csv("C:/Users/zuire/PycharmProjects/LeDiSas/LeSaS1/Data/hybrid_delta_delta.csv")
hybrid_data_mv <- read.csv("C:/Users/zuire/PycharmProjects/LeDiSas/LeSaS1/Data/hybrid_delta_delta_mv.csv")
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
mixed_effect <- lmer(optimal_percentage ~ Group * window_id * Weight +  (1|participant_id),
                     data = hybrid_data_mv)

summary(mixed_effect)
anova(mixed_effect)
plot(allEffects(mixed_effect))
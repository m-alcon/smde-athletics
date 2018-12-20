# library("lmtest", lib.loc="~/R/win-library/3.0")
library("lmtest")
df = read.csv("../data/final_marathon.csv")
Stage0_model = lm(Stage0 ~ Age+Gender+Fitness, data=df)
Stage1_model = lm(Stage1 ~ Age+Gender+Fitness, data=df)
Stage2_model = lm(Stage2 ~ Age+Gender+Fitness, data=df)
Stage3_model = lm(Stage3 ~ Age+Gender+Fitness, data=df)
Stage4_model = lm(Stage4 ~ Age+Gender+Fitness, data=df)
Stage5_model = lm(Stage5 ~ Age+Gender+Fitness, data=df)
Stage6_model = lm(Stage6 ~ Age+Gender+Fitness, data=df)
Stage7_model = lm(Stage7 ~ Age+Gender+Fitness, data=df)
Stage8_model = lm(Stage8 ~ Age+Gender+Fitness, data=df)

models<-list(Stage0_model,Stage1_model,Stage2_model,Stage3_model,Stage4_model,
             Stage5_model,Stage6_model,Stage7_model,Stage8_model)
dw_pvalues = c()
bp_pvalues = c()
for (model in models){
  #The observations within each sample must be independent.
  #Durbin Watson 
  dw = dwtest(model, alternative ="two.sided")
  dw_pvalues = c(dw_pvalues,dw$p.value)
  
  #The populations from which the samples are selected must have equal variances (homogeneity of variance)
  #Breusch Pagan test
  bp = lmtest::bptest(model)
  bp_pvalues = c(bp_pvalues,bp$p.value)
}

# SHAPIRO only accepts up to 5000 rows of data

df = df[sample(nrow(df),5000),]

Stage0_model = lm(Stage0 ~ Age+Gender+Fitness, data=df)
Stage1_model = lm(Stage1 ~ Age+Gender+Fitness, data=df)
Stage2_model = lm(Stage2 ~ Age+Gender+Fitness, data=df)
Stage3_model = lm(Stage3 ~ Age+Gender+Fitness, data=df)
Stage4_model = lm(Stage4 ~ Age+Gender+Fitness, data=df)
Stage5_model = lm(Stage5 ~ Age+Gender+Fitness, data=df)
Stage6_model = lm(Stage6 ~ Age+Gender+Fitness, data=df)
Stage7_model = lm(Stage7 ~ Age+Gender+Fitness, data=df)
Stage8_model = lm(Stage8 ~ Age+Gender+Fitness, data=df)

models<-list(Stage0_model,Stage1_model,Stage2_model,Stage3_model,Stage4_model,
             Stage5_model,Stage6_model,Stage7_model,Stage8_model)
sp_pvalues = c()
for (model in models){
  #The populations from which the samples are selected must be normal.
  #Shapiro test
  sp = shapiro.test(residuals(model))
  sp_pvalues = c(sp_pvalues,sp$p.value)
}


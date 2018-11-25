import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf



data = pd.read_csv("NFL-OT.csv")

## We use only games that ended up with a winner 
ind = data["homescore"] != data["awayscore"]
ind20122017 = data["season"] >= 2012

dataset = data[ind & ind20122017]

## Every OT game "offers" two data points 
## One for the home team and one for the visiting team 
## We keep track whether a team won, whether they received the kickoff (coin == 1 - here we make the assumption that the team receiving won the coin. This is not 100% true but almost 100%), and how many points favorite they were before the game (negative means they were underdog). 

otData = pd.DataFrame(columns=["home","coin","spread","win","season"])

for i in dataset.index.tolist():
        if dataset["homescore"][i] != dataset["awayscore"][i]:
                # add the info for the home team
                tmp = pd.DataFrame([[1,int(dataset["home"][i] == dataset["coin"][i]),dataset["spread"][i],int(dataset["homescore"][i] > dataset["awayscore"][i]),dataset["season"][i]]],columns=["home","coin","spread","win","season"])
                otData = otData.append(tmp)
                tmp = pd.DataFrame([[0,int(dataset["away"][i] == dataset["coin"][i]),-dataset["spread"][i],int(dataset["homescore"][i] < dataset["awayscore"][i]),dataset["season"][i]]],columns=["home","coin","spread","win","season"])
                otData = otData.append(tmp)

f = "win~C(coin)+spread"
otData.replace(to_replace={'win' : {'1': 1, '0': 0}}, inplace = True)

logitfit = smf.glm(formula = str(f), data = otData.sample(n=int(np.floor(len(otData)*0.8))),family=sm.families.Binomial()).fit()

print "########################################## Model 2012-2017 ##########################################"
print(logitfit.summary())

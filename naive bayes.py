import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
file_destination = "C://Users//Groot//Downloads//phen-lipids-miRNA.xlsx"
clf = MultinomialNB()
df = pd.read_excel(file_destination)
df.fillna(0).replace(" ",0)
df4 = df[["PartA_flavoredRice_24-numberOfServingsEatenPerMonth","PartA_friedNoodles_32-numberOfServingsEatenPerMonth"]]
df5 = df[df.columns[571:844]]
print(df4)
clf.partial_fit(df4, df5, sample_weight=None)

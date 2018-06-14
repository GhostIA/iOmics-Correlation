import pandas as pd
import numpy as np
import sys

# print(sys.argv)
"""
if len(sys.argv) < 7:
	print("You didn't pass all parameters (6)")
"""
file_destination ="C:\\Users\\Groot\\Downloads\\phen-lipids-miRNA.xlsx"
# file_destination2 = sys.argv[2]

df4 = pd.read_excel(file_destination)
"""
df5 = pd.read_excel(file_destination2)
df4 = pd.concat([df4, df5], axis=1, sort=False)
"""
df4.fillna(0).replace(" ",0)
def drop_x_row(df, x, column):
     df_unwanted_row = df[df[column] == x]
     df = df.drop(df_unwanted_row.index, axis=0)
     return df

one_values = drop_x_row(df4, 0, "gender adj")
zero_values = drop_x_row(df4, 1, "gender adj")
eth3_one = drop_x_row(one_values, 2, "Eth_Derived")
eth3_one = drop_x_row(eth3_one, 1, "Eth_Derived")
eth3_zero = drop_x_row(zero_values, 1, "Eth_Derived")
eth3_zero = drop_x_row(eth3_zero, 2, "Eth_Derived")
eth2_zero = drop_x_row(zero_values, 3, "Eth_Derived")
eth2_zero = drop_x_row(eth2_zero, 1, "Eth_Derived")
eth2_one = drop_x_row(one_values, 1, "Eth_Derived")
eth2_one = drop_x_row(eth2_one, 3, "Eth_Derived")
eth1_zero = drop_x_row(zero_values, 2, "Eth_Derived")
eth1_zero = drop_x_row(eth1_zero, 3, "Eth_Derived")
eth1_one = drop_x_row(one_values, 2, "Eth_Derived")
eth1_one = drop_x_row(eth1_one, 3, "Eth_Derived")

def correlate_data(df):
     count = 0
     arr = list(df)
     for j in range(571, 844):
          for i in range(288, 569):
               corr = df[arr[i]].corr(df[arr[j]])
               if corr >= 0.2:
                    print(arr[i] + " to " + arr[j])
                    print(corr)
                    count = count +1
     print(count)
correlate_data(eth3_one)

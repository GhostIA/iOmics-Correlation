import pandas as pd 
df = pd.read_excel("C:\\Users\\Groot\\Documents\\all data.xlsx", sheet = 1)
df.replace("?", 0, inplace = True)
df.fillna(0, inplace = True)
df.sort_values(["[FGF-2]"], ascending = True, inplace = True)
def drop_x_row(df, x, column):
     df_true_false_values = df[df[column] == x]
     df = df.drop(df_true_false_values.index, axis=0)
     return df


df = drop_x_row(df, 0, "[FGF-2]")
print(df["[FGF-2]"])

def column_to_list(df, col):
	indeces = list(df.index.values)
	array = list(df)
	arr = []
	for i in range(0, len(df.axes[0])):
		arr.append(df.ix[indeces[i], col])
	return arr
df_list = column_to_list(df, "[FGF-2]")
def quartile(arr):
	median_index = int(len(arr)/2) 
	second_quartile = arr[median_index]
	first_quartile = arr[int(median_index/2)]
	third_quartile = arr[int((3*median_index)/2) + 1]
	return first_quartile, second_quartile, third_quartile
print(quartile(df_list))
q1, q2, q3 = quartile(df_list)
def categorize(value, q1, q3):
	iqr = q3 - q1
	upper_bound = q3 + (1.5 * iqr)
	lower_bound = q1 - (1.5 * iqr)
	classification = 0
	if value < lower_bound or value > upper_bound:
		classification = 1
	elif value > q3 or value < q1:
		classification = 1
	else:
		classification = 0
	return classification
class_values = []
for i in range(0, len(df_list)):
	x = categorize(df_list[i], q1, q3)
	class_values.append(x)
print(class_values)
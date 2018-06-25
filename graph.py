import pandas as pd
import numpy as np 
from sklearn import svm, datasets
import matplotlib.pyplot as plt 

def drop_x_row(df, x, column):
     df_true_false_values = df[df[column] == x]
     df = df.drop(df_true_false_values.index, axis=0)
     return df

df = pd.read_excel("C:\\Users\\Groot\\Documents\\all data.xlsx", sheet = 1)
df.fillna(0, inplace=True)
df = drop_x_row(df, "hong kong", "country_of_birth")
df = drop_x_row(df, "cambodia", "country_of_birth")
df = drop_x_row(df, "china", "country_of_birth")
df = drop_x_row(df, "vietnam", "country_of_birth")
df = drop_x_row(df, "indonesia", "country_of_birth")
df = drop_x_row(df, "korea", "country_of_birth")
df = drop_x_row(df, "myanmar", "country_of_birth")
df = drop_x_row(df, "india", "country_of_birth")
df = drop_x_row(df, "phillipines", "country_of_birth")
df.replace("malaysia", float(0.0), inplace = True)
df.replace("singapore", float(1.0), inplace = True)

X = df[["[TPO]","[IL-23]"]].copy()
y = df.loc[:,"country_of_birth"].copy()

print(y)

x_min, x_max = X.iloc[:,0].min() + 1, X.iloc[:,0].max() +1
y_min, y_max = X.iloc[:,1].min() + 1, X.iloc[:,1].max() +1
print(x_min)
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))


X_plot = np.c_[xx.ravel(), yy.ravel()]

# Create the SVC model object
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr').fit(X, y)
Z = svc.predict(X_plot)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.Set1)
plt.xscale('linear')
plt.xlabel('TPO')
plt.ylabel('IL-23')
plt.xlim(xx.min(), 500)
plt.title('SVC with linear kernel')
plt.show()


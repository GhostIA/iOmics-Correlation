import pandas as pd
import numpy as np 
from sklearn import svm, datasets
import matplotlib.pyplot as plt 

df = pd.read_excel("C:\\Users\\Groot\\Documents\\all data.xlsx", sheet = 1)
df.fillna(0, inplace=True)
X = df.loc[:,"[OPG]": "[OPN]"].copy()
y = df.loc[:,"sii_comp_consensus_bp"].copy()



x_min, x_max = X.iloc[:,0].min() + 1, X.iloc[:,0].max() +1
y_min, y_max = X.iloc[:,1].min() + 1, X.iloc[:,1].max() +1
print(y_max)
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
plt.xlabel('OPG')
plt.ylabel('OPN')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()

from sklearn import preprocessing
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
import sklearn
#
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#直接补值
y_imp = imp.fit_transform([[np.nan, 2], [6, np.nan], [7, 6]])
print(y_imp)



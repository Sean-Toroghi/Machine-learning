# mitigate imbalanced dataset via oversample-SMOTE
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X,y = oversample.fit_resample(X,y)

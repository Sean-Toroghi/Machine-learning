# mitigate imbalanced dataset via oversample-SMOTE
def oversampling_smote(X,y):
  '''apply over sampling technique SMOTE
  in: X,y
  out: X,y
  '''
  from imblearn.over_sampling import SMOTE

  oversample = SMOTE()
  X,y = oversample.fit_resample(X,y)
  return X,y

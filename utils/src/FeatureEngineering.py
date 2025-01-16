# Feature engineering

#1. cross-feature aggregation: given a list of features, generate new features by performing 4 arithmatic operators (+ - X /) on the the cross-features
#2. remove highly correlated features via corr.matrix
#3. perform one hot encoder on features with less than threshold number of unique features
#4. labelencoding based on threshold on number of unique values
# Function 1
def cross_feature_eng(df:pd.DataFrame, cols: list)->pd.DataFrame:
    '''
    Perform feature engineering by adding cross feature interaction of the following four arithmatix operations: + - / *

    Input: 
        - list of features to perform cross interaction on.
        - dataframe
    uput:
        - df with new features added
    '''
    print("< cross feature >")
    print(f"original feature count: {df.shape[1]}")
    eps = 1e-15
    for i in range(len(cols)):
        for j in range(i+1,len(cols)):
            df[cols[i]+"_pls_"+cols[j]] = df[cols[i]]+df[cols[j]]
            df[cols[i]+"_neg_"+cols[j]] = df[cols[i]]-df[cols[j]]
            df[cols[i]+"_mul_"+cols[j]] = df[cols[i]]*df[cols[j]]
            df[cols[i]+"_dvd_"+cols[j]] = df[cols[i]]/(df[cols[j]]+eps)
    print(f"Post process feature count: {df.shape[1]}")
    return df
	
# Function 2
# 2. drop high correlation features
def high_correlation_cols(df:pd.DataFrame, exclude_cols: list, corr_threshold = 0.99 )->list:
     
    '''
    Description:
    - Create correlation matrix on numerical features, and remove one of the feature in feature pairs with high correlations.
    Input:
    - datarame
    - fatures to be excluded form the investigation: target, weight, gorup kfold
    - corr_threshold: Default = 0.99
    Return:
    - list of features to be removed
    '''
    numerical_cols=[col for col in df.columns if (col not in [exclude_cols]) and (str(df[col].dtype) not in ['object','category'])]
    corr_matrix=df[numerical_cols].corr().values
    drop_cols=[]
    for i in range(len(corr_matrix)):
        if numerical_cols[i] not in drop_cols:
            for j in range(i+1,len(corr_matrix)):
                if numerical_cols[j]  not in drop_cols:
                    if abs(corr_matrix[i][j])>=corr_threshold:
                        drop_cols.append(numerical_cols[j])
    print(f"drop_cols counts {len(drop_cols)}")
    del numerical_cols
    gc.collect()
    return drop_cols	
	
# Function 3
import pandas as pd

def one_hot_encode(df: pd.DataFrame, exclude_cols, threshold: int = 10):
    '''
    Perform one-hot-encoder on features with unique values less than threshold
    '''
    encode_feature = []
    new_categorical_features = []
    
    for col in df.columns:
        if col not in exclude_cols:
            nunique = df[col].nunique()
            if nunique < threshold:
                print(f"< one hot encoder: {col} >")
                encode_feature.append(col)
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                df = pd.concat([df, dummies], axis=1)
                new_categorical_features.extend(dummies.columns)
    
    # Drop the original columns that were one-hot encoded
    df.drop(columns=[col for col in df.columns if col in exclude_cols and col not in new_categorical_features], inplace=True)

    return df, new_categorical_features, encode_feature	
	
	
# Function 4
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_encode(train: pd.DataFrame, test: pd.DataFrame, exclude_cols: list, threshold: int = 10):
    '''
    Perform label encoding on features with less than threshold number of unique features
    '''
    encode_feature = []
    label_encoders = {}
    
    for col in train.columns:
        if col not in exclude_cols:
            nunique = train[col].nunique()
            if nunique < threshold:
                print(f"< label encoder: {col} >")
                encoder = LabelEncoder()
                train[col] = encoder.fit_transform(train[col])
                test[col]  = encoder.transform(test[col])
                label_encoders[col] = encoder
                encode_feature.append(col)
    
    return train, test, encode_feature, label_encoders	
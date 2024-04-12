# Data manipulation libraries
import pandas as pd
# We import our upsample function
from upsample_func import upsample
# Machine learning libraries
from sklearn.model_selection import train_test_split

# We define our processing function

def processing(df):
    # We create a seed
    random_state = 12345

    # We split our dataset into target and features 
    features = df.drop('is_ultra',axis=1)
    target = df.is_ultra

    # We split our datasets into train and test
    features_train, features_test, target_train, target_test = train_test_split(
        features, target,test_size=0.2,random_state=random_state)
    
    # We use the function to upsample our data
    features_upsampled, target_upsampled = upsample(features_train,target_train,10)
    
    # We split our datasets into train and test
    features_train_u, features_test_u, target_train_u, target_test_u = train_test_split(
        features_upsampled, target_upsampled,test_size=0.2,random_state=random_state)
    
    return features_train, features_test, target_train, target_test,features_train_u, features_test_u, target_train_u, target_test_u
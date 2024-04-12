# We import the libreries we'll use

# Data manipulation libraries
import pandas as pd

# Progess bar
from tqdm import tqdm

# Machine learning libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def model_selection(features_train, features_test, target_train, target_test,
                    features_train_u, features_test_u, target_train_u, target_test_u):
    
    # Initialize progress bar
    with tqdm(total=6) as pbar:
        pbar.set_description('Training Models')

        # Perform model training and evaluation


        # We create a seed
        random_state = 12345

        # Decission Tree model
        param_grid = {
        'criterion':['gini','entropy'],
        'max_depth':[None,2,4,6],
        'min_samples_split':[2,4,6],
        }
    
        grid_seach = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=random_state),param_grid=param_grid,cv=5,scoring='accuracy')
    
        # We train our model
        grid_seach.fit(features_train,target_train)
    
        best_model_t = grid_seach.best_estimator_ # We save the best model on a variable
        predict_tree = best_model_t.predict(features_test) # We save the predictions on a variable
    
        # We test our model
        accuracy_test_t = accuracy_score(target_test,predict_tree)
        pbar.update(1)

        # Random Forest model
        param_grid = {
        'n_estimators':[20,50,100],
        'max_depth':[None,2,4,6],
        'min_samples_split':[2,4,6]
        }
    
        grid_seach = GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_state),param_grid=param_grid,cv=5,scoring='accuracy')
    
        # We train our model
        grid_seach.fit(features_train,target_train)
    
        best_model_rf = grid_seach.best_estimator_ # We save the best model on a variable
        predict_rf = best_model_rf.predict(features_test) # We save the predictions of our model on a variable
    
        # We test our model
        accuracy_test_rf = accuracy_score(target_test,predict_rf)
        pbar.update(1)

        # Gradient Boosting

        param_grid = {
        'n_estimators':[50,100,150],
        'min_samples_split':[2,4,6],
        'max_depth':[None,1,3],   
        }
    
        grid_seach = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=random_state),cv=5,param_grid=param_grid,scoring='accuracy')
    
        # We train our model
        grid_seach.fit(features_train,target_train)
    
        best_model_gb = grid_seach.best_estimator_ # We save the best model
        predict_gb = best_model_gb.predict(features_test)
    
        # We test our model
        accuracy_test_gb = accuracy_score(target_test,predict_gb)
        pbar.update(1)

        # Decission Tree Upsampled
        param_grid = {
        'criterion':['gini','entropy'],
        'max_depth':[None,2,4,6],
        'min_samples_split':[2,4,6],
        }
    
        grid_seach = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=random_state),param_grid=param_grid,cv=5,scoring='accuracy')
    
        # We train our model
        grid_seach.fit(features_train_u,target_train_u)
    
        best_model_t_u = grid_seach.best_estimator_ # We save the best model on a variable
        predict_tree_u = best_model_t_u.predict(features_test_u) # We save the predictions on a variable
    
        # We test our model
        accuracy_test_t_u = accuracy_score(target_test_u,predict_tree_u)
        pbar.update(1)

        # Random Forest Upsampled
        param_grid = {
        'n_estimators':[20,50,100],
        'max_depth':[None,2,4,6],
        'min_samples_split':[2,4,6]
        }
        
        grid_seach = GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_state),param_grid=param_grid,cv=5,scoring='accuracy')
        
        # We train our model
        grid_seach.fit(features_train_u,target_train_u)
        
        best_model_rf_u = grid_seach.best_estimator_ # We save the best model on a variable
        predict_rf_u = best_model_rf_u.predict(features_test_u) # We save the predictions of our model on a variable
        
        # We test our model
        accuracy_test_rf_u = accuracy_score(target_test_u,predict_rf_u)
        pbar.update(1)

        # Gradient Boosting Upsampled

        param_grid = {
        'n_estimators':[20,50,100],
        'min_samples_split':[2,4,6],
        'max_depth':[None,1,3],   
        }
        
        grid_seach = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=random_state),cv=5,param_grid=param_grid,scoring='accuracy')
        
        # We train our model
        grid_seach.fit(features_train_u,target_train_u)
        
        best_model_gb_u = grid_seach.best_estimator_ # We save the best model
        predict_gb_u = best_model_gb_u.predict(features_test_u)

        # We test our model
        accuracy_test_gb_u = accuracy_score(target_test_u,predict_gb_u)
        pbar.update(1)

    # We save the scores on a dictionary
    scores = {
        'Decission Tree': accuracy_test_t,
        'Random Forest': accuracy_test_rf,
        'Gradient Bosst': accuracy_test_gb,
        'Decission Tree Upsample': accuracy_test_t_u,
        'Random Forest Upsample': accuracy_test_rf_u,
        'Gradient Bosst Upsample': accuracy_test_gb_u,
        }
    
    # Select the best model
    best_model_name = max(scores, key=scores.get)
    best_model = None
    best_model_score = None

    if best_model_name == 'Decission Tree':
            best_model = best_model_t  
            best_model_score = accuracy_test_t 
    elif best_model_name == 'Random Forest':
            best_model = best_model_rf
            best_model_score = accuracy_test_rf
    elif best_model_name == 'Gradient Bosst':
           best_model = best_model_gb
           best_model_score = accuracy_test_gb
    elif best_model_name == 'Decission Tree Upsample':
           best_model = best_model_t_u
           best_model_score = accuracy_test_t_u
    elif best_model_name == 'Random Forest Upsample':
           best_model = best_model_rf_u
           best_model_score = accuracy_test_rf_u
    elif best_model_name == 'Gradien Boost Upsample':
           best_model = best_model_gb_u
           best_model_score = accuracy_test_gb_u
    
    return best_model, best_model_name, best_model_score 






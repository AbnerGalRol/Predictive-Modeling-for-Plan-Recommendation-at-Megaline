# We import our libreries
import pandas as pd
from processing import processing
from Best_model import model_selection
from tqdm import tqdm

# We load our dataset
print("Loading dataset...")
df = pd.read_csv(r'data/users_behavior.csv')

# We process our data
print("Processing data...")

features_train, features_test, target_train, target_test,features_train_u, features_test_u, target_train_u, target_test_u = processing(df)

# We create and evaluate our models
print("Creating and evaluating models...")
with tqdm(total=2) as pbar:
    pbar.set_description('Model Selection')
    best_model, best_model_name, best_model_score  = model_selection(features_train, features_test, target_train, target_test,
                                                                 features_train_u, features_test_u, target_train_u, target_test_u)
    pbar.update(1)  

    # We print the best model and its accuracy score
    print(f'''
    Best Model: {best_model_name}
    Best Model Accuracy: {best_model_score}
    ''')
    pbar.update(1) 

print("Task completed successfully!")

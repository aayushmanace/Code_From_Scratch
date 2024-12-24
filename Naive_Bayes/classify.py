import pickle
import os
import re
import pandas as pd
from main import classify
import argparse

# Argument parser setup
parser = argparse.ArgumentParser()

# Input directory argument
parser.add_argument('--input_dir', 
                    default="test", 
                    type=str, 
                    help='Input Directory of Test Data')

# # Save argument to store results as CSV
# parser.add_argument('--save', 
#                     action='store_true',  # Changed to store_true to handle boolean flag
#                     type=int,
#                     help='Bool value to store prediction values as csv')

args = parser.parse_args()

# Load the model parameters
with open('naive_bayes_model.pkl', 'rb') as f:
    model_params = pickle.load(f)

P_spam = model_params['P_spam']
P_ham = model_params['P_ham']
P_word_given_spam = model_params['P_word_given_spam']
P_word_given_ham = model_params['P_word_given_ham']

# Example classification
print("Example Usage to test classify function for:")
print("'WINNER!! This is the secret code to unlock the money: C3421.'")
result = classify('WINNER!! This is the secret code to unlock the money: C3421.', verbose=0)
print("Classification Result:", result)

folder_path = args.input_dir
save = True


# Function to classify files
def classify_files(folder_path):
    texts = []
    file_names = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):  # Assuming text files
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                texts.append(file.read())
                file_names.append(file_name)
    
    # Sort file_names and texts based on the integer value of the extracted number
    sorted_indices = sorted(range(len(file_names)), 
                            key=lambda i: int(re.findall(r'\d+', file_names[i])[0]) if re.findall(r'\d+', file_names[i]) else float('inf'))
    
    texts = [texts[i] for i in sorted_indices]
    file_names = [file_names[i] for i in sorted_indices]  # Keep the order consistent

    return texts, file_names


# Classify the files in the folder
print("-"*10)
print("Now Classifying all the files in 'Test' folder")
texts, file_names = classify_files(folder_path)
df = pd.DataFrame({"Email": texts, "File_Name": file_names})
df["Email"] = df['Email'].str.replace('\W',' ')
df["Email"] = df['Email'].str.lower()
df["Email"] = df['Email'].fillna("")
df['predicted'] = df.Email.apply(classify, verbose=0)
df['predicted'] = df['predicted'].map({'spam': 1, 'ham': 0})

# Save the results as CSV if --save is passed
if save:
    print("-" * 10)
    df[["File_Name", "predicted"]].to_csv("DA24S016_predictions.csv", index=False)
    print("Saved the predicted values as 'DA24S016_predictions.csv'")

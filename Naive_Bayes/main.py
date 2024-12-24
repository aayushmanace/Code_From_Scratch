import pandas as pd
import numpy as np

from itertools import chain
import re
from collections import defaultdict


from sklearn.metrics import classification_report


def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    df.columns = ["Email", "Label"]
    df['Label'] = df['Label'].map({1: 'spam', 0: 'ham'})
    return df

df = load_data('spam_or_not_spam.csv')

print("-"*5 + "About Data" + "-"*5)
print(df.describe())
print(df.shape)


# Split the dataset into training and testing sets
def split_data(df, train_fraction=0.8, random_state=1):
    print("-"*5 + "Splitting Data" + "-"*5)
    randomized = df.sample(frac=1, random_state=random_state)
    train_size = round(len(randomized) * train_fraction)
    train = randomized[:train_size].reset_index(drop=True)
    test = randomized[train_size:].reset_index(drop=True)
    train.Label.value_counts(normalize=True)
    test.Label.value_counts(normalize=True)
    return train, test

train, test = split_data(df)


train.Email.head(10)

def preprocess_emails(emails):
    emails = emails.str.replace('\W', ' ', regex=True).str.lower().fillna('')
    return emails

train['Email'] = preprocess_emails(train['Email'])
messages = train.Email.str.split()

words = list(chain(*messages))
vocabulary = pd.Series(words).unique()
len(vocabulary)



no_of_messages = len(train.Email)
word_counts = {word: [0] * no_of_messages for word in vocabulary}

for index,mssge in enumerate(messages):
    for word in mssge:
        word_counts[word][index] += 1



print("Training Begins.....")


class CustomTokenizer:
    def __init__(self):
        # Regular expression pattern to match words
        self.pattern = re.compile(r'\b[a-zA-Z0-9]+\b')
    def tokenize(self, text):
        # Find all words in the text using regex and return them as lowercase
        return self.pattern.findall(text.lower())

class CountVectorizerCustom:
    def __init__(self):
        self.vocabulary = {}
        self.tokenizer = CustomTokenizer()

    def fit(self, documents):
        # Create vocabulary from the documents
        unique_words = set()
        for doc in documents:
            tokens = self.tokenizer.tokenize(doc)
            unique_words.update(tokens)

        # Create a mapping of words to indices
        self.vocabulary = {word: i for i, word in enumerate(unique_words)}

    def transform(self, documents):
        # Create a document-term matrix
        matrix = np.zeros((len(documents), len(self.vocabulary)), dtype=int)

        for doc_idx, doc in enumerate(documents):
            tokens = self.tokenizer.tokenize(doc)
            for token in tokens:
                if token in self.vocabulary:
                    matrix[doc_idx, self.vocabulary[token]] += 1

        return matrix

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self):
        return list(self.vocabulary.keys())



# List of common English stop words
stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn",
    "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "needn't", "need", "needn't"
])

# Initialize and fit the custom count vectorizer
count_vectorizer = CountVectorizerCustom()
count_matrix = count_vectorizer.fit_transform(train['Email'])

# Display the feature names (vocabulary)
vocabulary = count_vectorizer.get_feature_names()

# Remove stop words
vocabulary = [word for word in vocabulary if word not in stop_words]

# Update the count vectorizer with the filtered vocabulary
count_vectorizer.vocabulary = {word: i for i, word in enumerate(vocabulary)}
count_matrix = count_vectorizer.transform(train['Email'])

print("Length of Training Vocabulary:", len(vocabulary))



vector_space = pd.DataFrame(word_counts)
train = pd.concat([train,vector_space],axis=1)

alpha = 1
P_spam = train.Label.value_counts(normalize=True)['spam']
P_ham = train.Label.value_counts(normalize=True)['ham']
N_spam = len(list(chain(*messages[train.Label == 'spam'])))
N_ham = len(list(chain(*messages[train.Label == 'ham'])))
N_vocab = len(vocabulary)





P_word_given_spam = {word: 0 for word in vocabulary}
P_word_given_ham = {word: 0 for word in vocabulary}

Spam_messages = train[train.Label == 'spam']
Ham_messages = train[train.Label == 'ham']

for word in vocabulary:
    try:
        N_word_given_spam = Spam_messages[word].sum()
    except KeyError:
        # If the word is not found in the Spam_messages DataFrame, skip to the next word
        N_word_given_spam = 0  # You can set it to 0 or handle it differently if needed

    try:
        N_word_given_ham = Ham_messages[word].sum()
    except KeyError:
        # If the word is not found in the Ham_messages DataFrame, skip to the next word
        N_word_given_ham = 0  # You can set it to 0 or handle it differently if needed

    # Calculate probabilities
    P_word_given_spam[word] = (
        (N_word_given_spam + alpha) / (N_spam + (alpha * N_vocab))
    )

    P_word_given_ham[word] = (
        (N_word_given_ham + alpha) / (N_ham + (alpha * N_vocab))
    )




import pickle

# After calculating the probabilities and before the classify function
model_params = {
    'P_spam': P_spam,
    'P_ham': P_ham,
    'P_word_given_spam': P_word_given_spam,
    'P_word_given_ham': P_word_given_ham
}

print("Finished training now saving model_parameters.....")

# Save the model parameters to a file
with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(model_params, f)



def classify(message,verbose):
    """
    Classifies a given email message as 'spam' or 'ham' (non-spam) using a Naive Bayes classifier.

    The function computes the probabilities of the message being 'spam' or 'ham' based on the 
    words contained in the message. It compares these probabilities to determine the classification.

    Parameters:
    - message (str): The email message to be classified.
    - verbose (bool): If True, the function will print detailed outputs of the classification process,
                        including intermediate probabilities. Default is False.

    Returns:
    - str: Returns 'spam' if the message is classified as spam, 'ham' if classified as non-spam,
            or 'Not classified' if the probabilities for both classes are equal.
    """
    message = re.sub('\W', ' ', message)
    message = message.lower()
    message = message.split()

    spam = 1
    ham = 1

    for word in message:
        if word in P_word_given_spam.keys():
            spam *= P_word_given_spam[word]
        if word in P_word_given_ham.keys():
            ham *= P_word_given_ham[word]

    P_spam_given_message = P_spam * spam
    P_ham_given_message = P_ham * ham

    if verbose:
        print("P(spam|message) = ",P_spam_given_message)
        print("P(ham|message) = ",P_ham_given_message)

    if P_spam_given_message > P_ham_given_message:
        if verbose:
            print('Label: spam')
        return 'spam'
    elif P_spam_given_message <= P_ham_given_message:
        if verbose:
            print('Label: ham')
        return 'ham'
    else:
        if verbose:
            print('Human assistance needed, equal probabilities')
        return 'Not classified'

train['predicted'] = train.Email.apply(classify,verbose=0)      
print(classification_report(train.predicted, train.Label))

if __name__ == '__main__':
    print("-"*10)
    print("CLASSIFYING --- 'WINNER!! This is the secret code to unlock the money: C3421.'")
    classify('WINNER!! This is the secret code to unlock the money: C3421.',verbose=1)


    print("-"*10)
    print("CLASSIFYING ---- 'Sounds good, Tom, then see u there'")
    classify('Sounds good, Tom, then see u there',verbose=1)

    test['predicted'] = test.Email.apply(classify,verbose=0) #try putting verbose >1 and see the output of the model
    test.head(5)

    test.predicted.value_counts(normalize=True)

    train['predicted'] = train.Email.apply(classify,verbose=0)
    accuracy = sum(train.Label == train.predicted)/len(train)
    print(f"Accuracy on Train Data: {accuracy: .6f}")

    print("-"*10)
    print("Result on Test Data:")

    accuracy = sum(test.Label == test.predicted)/len(test)
    print(f"Accuracy on Test Data:  {accuracy: .6f}")

    train.predicted.value_counts(normalize=True)

    def clean(message):

        message = re.sub('\W',' ',message)
        message = message.lower()

        return message

    test.Email = test.Email.apply(clean)
    missclassified = test[test.Label != test.predicted]
    missclassified


    """ Saving the model parameters
    """
    
    import pickle

    # After calculating the probabilities and before the classify function
    model_params = {
        'P_spam': P_spam,
        'P_ham': P_ham,
        'P_word_given_spam': P_word_given_spam,
        'P_word_given_ham': P_word_given_ham
    }

    # Save the model parameters to a file
    with open('naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(model_params, f)




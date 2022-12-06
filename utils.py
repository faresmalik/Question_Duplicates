import numpy as np 
from nltk.tokenize import word_tokenize
from collections import defaultdict
import torch

def get_vocab_train_questions(question1_train, question2_train): 

    Q1_train = []
    Q2_train = []

    vocab = defaultdict(lambda: 0)
    vocab['<PAD>'] = 1
    idx = 0 

    for q1, q2 in zip(question1_train, question2_train): 

        q1_tokenized = word_tokenize(q1)
        q2_tokenized = word_tokenize(q2)
        Q1_train.append(q1_tokenized)
        Q2_train.append(q2_tokenized)

        for w1, w2 in zip(q1_tokenized, q2_tokenized):
            if w1 not in vocab: 
                vocab[w1] = len(vocab)+1
            if w2 not in vocab: 
                vocab[w2] = len(vocab)+1
    return vocab, Q1_train, Q2_train

def get_test_questions(question1_test, question2_test): 
    Q1_test = []
    Q2_test = []

    for q1, q2 in zip(question1_test, question2_test):
        Q1_test.append(word_tokenize(q1))
        Q2_test.append(word_tokenize(q2)) 

    return Q1_test, Q2_test 
    

def get_tensors(question1, question2, vocab):
    Q1 = []
    Q2 = []

    idx = 0

    for q1, q2 in zip(question1, question2):
        Q1.append(torch.tensor([vocab[word] for word in q1]))
        Q2.append(torch.tensor([vocab[word] for word in q2]))
    
    return Q1, Q2

def predict(vocab,model,threshold, device):
    q1 = input('Enter the first Question')
    q2 = input('Enter the second Question')

    print(f'First Questions is: {q1}')
    print(f'Second Questions is: {q2}')

    q1_tokenized_ = word_tokenize(q1)
    q2_tokenized_ = word_tokenize(q2)

    encode1 = torch.tensor([vocab[i] for i in q1_tokenized_]).to(device).unsqueeze(dim=0)
    encode2 = torch.tensor([vocab[i] for i in q2_tokenized_]).to(device).unsqueeze(dim=0)

    q1_embedding, q2_embedding = model(encode1, encode2)
    q1_embedding = q1_embedding.squeeze()
    q2_embedding = q2_embedding.squeeze()    

    score = torch.dot(q1_embedding,q2_embedding).item()
    result = 'Duplicate' if score > threshold else 'Not Duplicate'

    print(result)
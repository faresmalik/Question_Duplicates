import pandas as pd 
from sklearn.model_selection import train_test_split
from utils import get_test_questions, get_vocab_train_questions, get_tensors
import torch 
from torch.utils.data import DataLoader 
from dataset import DuplicateDataset
from train_evaluate import TrianEvaluate
from model import DuplicatesNetwork
import argparse
import os 

d_model = 512
n_lstm = 3
padding_word = '<PAD>'
batch_first = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Train ViT')

parser.add_argument('--questions_path', type=str , default ='questions.csv',
                    help='Questions CSV Path')
parser.add_argument('--test_size', type=float , default=0.1,
                    help='Pass CSV File Path. default: train.csv')
parser.add_argument('--random_state', type=int , default=42,
                    help='Random State For Splitting The data')
parser.add_argument('--num_workers', type=int , default = 12,
                    help='Num Workers')
parser.add_argument('--batch_size', type=int , default = 64,
                    help='Choose The Size of the Batch [16,32,64,128]')
parser.add_argument('--max_length', type=int , default = 55,
                    help='Max Length for the Sequence')
parser.add_argument('--epochs', type=int , default = 10,
                    help='Number of Epochs')
parser.add_argument('--margin', type=float , default=0.80,
                    help='Margin used in the triplet loss')
parser.add_argument('--threshold', type=float , default=0.80,
                    help='Threshold')
parser.add_argument('--lr', type=float , default= 1e-3,
                    help='Learning Rate')
parser.add_argument('--shuffle_train', default = True,
                    help='Shuffle Training Data When Creating DataLoader')
parser.add_argument('--shuffle_test', default = False,
                    help='Shuffle Testing Data When Creating DataLoader')
parser.add_argument('--gpu_number', type=int , default= 0,
                    help='GPU number ')

args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']= str(args.gpu_number)

print('Loading the dataset as Dataframe')
data = pd.read_csv(args.questions_path)
data = data.drop(['id', 'qid2', 'qid1'], axis = 1)

print('Get duplicates question only ')
duplicaate_data = data[data['is_duplicate'] == 1]

print('Split the dataset into training/testing')
train_data, test_data = train_test_split(duplicaate_data, test_size = args.test_size, random_state = args.random_state)

question1_train = train_data.question1.values
question1_test = test_data.question1.values

question2_train = train_data.question2.values
question2_test = test_data.question2.values

print('Get vocabulary and tokenize the training questions')
vocab, q1_train_tokenized , q2_train_tokenized = get_vocab_train_questions(question1_train, question2_train) 
idx_to_word = {i:d for d,i in vocab.items()}
pad_idx = vocab[padding_word]

print('Tokenize the testing questions')
q1_test_tokenized, q2_test_tokenized = get_test_questions(question1_test, question2_test)

print('Get the tokenized questions as tensor --- This step will take a minute')
q1_train_tensor, q2_train_tensor = get_tensors(q1_train_tokenized, q2_train_tokenized, vocab)
q1_test_tensor, q2_test_tensor = get_tensors(q1_test_tokenized, q2_test_tokenized, vocab)

print('Get the train/test datasets and dataloaders')
train_dataset = DuplicateDataset(question1= q1_train_tensor, question2= q2_train_tensor, max_len= args.max_length)
test_dataset = DuplicateDataset(question1= q1_test_tensor, question2= q2_test_tensor, max_len= args.max_length)
train_dataloader = DataLoader(train_dataset, batch_size= args.batch_size, shuffle=args.shuffle_train, num_workers= args.num_workers)
test_dataloader = DataLoader(test_dataset, batch_size= args.batch_size, shuffle=args.shuffle_test, num_workers= args.num_workers)

print('Instantiate the model')
model = DuplicatesNetwork(vocab_size = len(vocab), d_model = d_model, n_LSTM = n_lstm, padd = pad_idx, batch_first = batch_first).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr = args.lr)
trainer = TrianEvaluate(model= model, train_loader= train_dataloader, test_loader= test_dataloader, optimizer= optimizer, epochs = args.epochs, margin = args.margin, device= device)

train_acc, train_loss = trainer.fit(threshold= args.threshold)


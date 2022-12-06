import torch 

class DuplicatesNetwork(torch.nn.Module): 
    def __init__(self, vocab_size, d_model, n_LSTM, padd = 1, batch_first = True) -> None:
        super(DuplicatesNetwork, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_LSTM = n_LSTM  
        self.padd = padd
        self.batch_first = batch_first
        self.Embedding = torch.nn.Embedding(num_embeddings= self.vocab_size, embedding_dim= self.d_model, padding_idx= self.padd) 
        self.LSTM = torch.nn.LSTM(input_size = d_model, hidden_size = d_model, num_layers = self.n_LSTM, batch_first = self.batch_first)

    def forward_once(self, q1, q2): 
        q1_output = self.Embedding(q1)
        q2_output = self.Embedding(q2) 
        q1_output, _ = self.LSTM(q1_output)
        q1_output = torch.mean(q1_output, dim=1)
        q1_output = torch.nn.functional.normalize(q1_output, dim=-1)
        q2_output, _= self.LSTM(q2_output)
        q2_output = torch.mean(q2_output, dim=1)
        q2_output = torch.nn.functional.normalize(q2_output, dim=-1)
        return q1_output, q2_output
    
    def forward(self, q1, q2): 
        q1_seq, q2_seq = self.forward_once(q1,q2)
        return q1_seq, q2_seq

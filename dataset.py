import torch 
import numpy 
from torch.utils.data import Dataset 

class DuplicateDataset(Dataset): 
    
    def __init__(self, question1, question2, max_len = 60) -> None:
        super(DuplicateDataset).__init__()
        self.question1 = question1
        self.question2 = question2
        self.max_lenght = max_len
       

    def __len__(self): 
        return len(self.question1)

    def __getitem__(self, index):
        q1 = self.question1[index]
        q2 = self.question2[index]

        if len(q1) > self.max_lenght: 
            q1 = q1[:self.max_lenght]
        if len(q2) > self.max_lenght: 
            q2 = q2[:self.max_lenght]
    
        req_len_q1 = self.max_lenght - len(q1)
        req_len_q2 = self.max_lenght - len(q2)

        if req_len_q1 > 0: 
            q1 = torch.cat((q1, torch.ones(req_len_q1)), dim=0)
        if req_len_q2 > 0: 
            q2 = torch.cat((q2, torch.ones(req_len_q2)), dim=0)

        return (q1.long(), q2.long())
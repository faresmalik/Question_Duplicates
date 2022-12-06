import torch
from tqdm import tqdm 

class TrianEvaluate(): 
    def __init__(self, model,train_loader, test_loader, optimizer, epochs, margin, device) -> None:
        super(TrianEvaluate).__init__()
        self.train_loader = train_loader        
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.epochs = epochs
        self.model = model
        self.margin = margin
        self.device = device

    def fit(self, threshold): 
        accs = []
        losses = []
        for epoch in range(self.epochs):
            self.model.train()
            print(f'Epoch {epoch+1}/{self.epochs}')
            loss_train = 0.0
            train_acc = 0.0 
            for data in tqdm(self.train_loader): 
                self.optimizer.zero_grad()
                q1_encoding, q2_encoding = self.model.forward_once(data[0].to(self.device), data[1].to(self.device))
                loss = self.triplet_loss(q1_encoding, q2_encoding, margin=self.margin)
                loss.backward()
                self.optimizer.step()
                loss_train += loss.item()
                acc_inter = 0
                for question_idx in range(len(q1_encoding)):
                    similarity = torch.dot(q1_encoding[question_idx], q2_encoding[question_idx]).item()
                    if similarity > threshold: 
                        acc_inter +=1 
                train_acc += acc_inter / len(data[0])
            losses.append(loss_train/len(self.train_loader))
            accs.append(train_acc/len(self.train_loader))
            print(f' Loss = {(loss_train/len(self.train_loader)):.3f}')
            print(f'Accuracy = {(train_acc/len(self.train_loader)):.3f} \n')

            if ((epoch+1)%3 == 0) and (epoch!=0): 
                self.eval(threshold= threshold)
        print('==================== Final Evaluation ====================')
        self.eval(threshold= threshold)
        return accs, losses
        
    def triplet_loss(self, q1,q2, margin):
        scores_matrix = torch.mm(q1, q2.T) #get nxn matrix (dot product)
        batch_size = scores_matrix.shape[0] #batch size 
        positives = scores_matrix.diag() #get the diagonals of the scores matrix (POSITIVE SCORES)
        negative_scores = scores_matrix * (1.0 - torch.eye(batch_size, device= self.device)) #Get only negatives score from the scores matrix
        mean_negatives = torch.sum(negative_scores, dim=-1, keepdim=True) / (batch_size-1) #Get the mean negative in each row 
        mask1 = torch.eye(batch_size).to(device=self.device) == 1  #Set the diagonal = True --> exclude the positives when calculate closest mean 
        mask2 = negative_scores > positives.reshape(batch_size,1) #exclude negatives that exceed the positive in each row 
        joint_mask = mask1 | mask2  #join two masks togther to mask diagonals and negative values > positive in each row
        negative_no_positives = negative_scores - 3.0 * joint_mask #push the diaginals and negative values greater than the diagonal in each row away from 0 to avoid counting them
        closest_negative, _ = torch.max(negative_no_positives, dim=-1,keepdim= True)
        loss1= torch.maximum(torch.zeros((batch_size,1), device = self.device), margin + mean_negatives - positives.reshape(batch_size,1))
        loss2 = torch.maximum(torch.zeros((batch_size,1), device = self.device), margin + closest_negative - positives.reshape(batch_size,1))
        loss = torch.mean(loss2+loss1)
        return loss

    def eval(self, threshold): 
        self.model.eval()
        print(f'=== Evaluation Phase ===')
        test_acc = 0.0 
        for data in tqdm(self.test_loader): 
            with torch.no_grad():
                q1_encoding, q2_encoding = self.model.forward_once(data[0].to(self.device), data[1].to(self.device))
            acc_inter = 0.0
            for question_idx in range(len(q1_encoding)):
                similarity = torch.dot(q1_encoding[question_idx], q2_encoding[question_idx]).item()
                if similarity > threshold: 
                    acc_inter +=1 
            test_acc += acc_inter / len(data[0])
        print(f'Eval Accuracy = {(test_acc/len(self.test_loader)):.3f}')
        print(' ============================================================== ')
        return        

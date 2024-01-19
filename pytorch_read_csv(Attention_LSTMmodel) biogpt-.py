import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
def check_create(path_name):
      from pathlib import Path
      import os
      my_file = Path(path_name)
      if my_file.is_dir():
        print('dir exist')
      else:
        print('not exist')
        os.makedirs(my_file)
        print('making a folder, finished')
        
def return_xy_from_cssv(dfname,model,MODE):
    print(dfname)
    df = pd.read_csv(dfname)  # 读取数据并赋予列名
    sc_name=dfname[0:-4]
    #print(type(df))  # <class 'pandas.core.frame.DataFrame'>

    #print(df.columns)
    y = df['label']
    y = np.array(y)
    y = torch.unsqueeze(torch.FloatTensor(y), dim=1)
    #print(type(y))  # <class 'torch.Tensor'>
    print(y.shape)  # torch.Size([97, 1])#sample size
    testlabel=df['label'].values.tolist()
    testDB=df['DBID1'].values.tolist()
    test_RNA=df['DBID2'].values.tolist()
    df=df.drop(['label','DBID1','DBID2'],axis=1)
    #print(df)
    x = df
    
    #print(type(x))  # <class 'pandas.core.series.Series'>
    x = np.array(x)
    #print(type(x))  # <class 'numpy.ndarray'>
    #print(x.shape)
    #print(len(x[0]))
    import math
    embedding_length=math.ceil(len(x[0])**0.5)
    
    print(embedding_length)
    diver=embedding_length**2-len(x[0])
    print(diver)
                                  
    origianl_length=len(x)
    x=np.pad(x,(0,diver),'constant', constant_values=(0,0))
    x=x[0:origianl_length]
    if MODE=='Train' or MODE=='TRAIN':
          from sklearn.preprocessing import StandardScaler
          scaler= StandardScaler()
          xs=scaler.fit_transform(x)
          import pickle
          pickle.dump(scaler, open(f'sc_name.pkl','wb'))

          
    elif MODE=='Test' or MODE=='TEST':
           from sklearn.preprocessing import StandardScaler
           import pickle
           scaler = pickle.load(open(f'sc_name.pkl', 'rb'))
           xs = scaler.transform(x)
    if model=='CNN':
      xs=x.reshape(len(x),embedding_length,embedding_length)
    #print(x.shape)

    #x = x.tolist()
    #print(type(x))  # <class 'list'>
    xs = torch.unsqueeze(torch.FloatTensor(xs), dim=1)
    #print(type(x))  # <class 'torch.Tensor'>
    #print(x.shape)  # torch.Size([97, 1])
   
    if MODE=='Train' or MODE=='TRAIN':
      return xs,y,embedding_length
    if MODE=='Test' or MODE=='TEST':
          return xs,y,embedding_length,testlabel,testDB,test_RNA


class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
list_results=[]
dataset='0DDI_Major1'
test_TYPE='five_fold'
valid_results_dir=r'.\\valid_results\\{}_{}'.format(dataset,test_TYPE)
check_create(valid_results_dir)
#Resnet50_dbioGPTEsum
vkey='bioGPTesum'
dkey='bioGPTesum'
listauc=[]
listaupr=[]
frs=0
model_name='attLSTM'
#check=r'./5folds/0RNAmigoxHARIBOSS_vDoc2ve(34567_2048)_d3DResnet50_frs0_dfs0_epoch1.csv'
import os
print(os.getcwd())
print(os.path.abspath(os.path.dirname(__file__)))
#pdf=pd.read_csv(r'.\\5folds\\0RNAmigoxHARIBOSS_vDoc2ve(34567_2048)_d3DResnet50_frs0_drs0_epoch1.csv')
#print(pdf.shape)
for divide_rs in range(3,5):
 for fepoch in range(1,6):
       #./5folds/0RNAmigoxHARIBOSS_vDoc2ve(34567_2048)_d3DResnet50_frs0_dfs0_epoch1.csv'
                #0RNAmigoxHARIBOSS_vDoc2ve(34567_2048)_d3DResnet50_frs0_drs0_epoch1.csv
    x_train,y_train,edsize=return_xy_from_cssv(r'./5folds/{}_v{}_d{}_frs_{}_rs{}_epoch{}.csv'.format(dataset,vkey,dkey,frs,divide_rs,fepoch),model=model_name,MODE='TRAIN')
    x_test,y_test,edsize,tlabel,dblabel,rnalabel=return_xy_from_cssv(r'./5folds/{}_v{}_d{}_frs_{}_rs{}_epoch{}_test.csv'.format(dataset,vkey,dkey,frs,divide_rs,fepoch),model=model_name,MODE='TEST')
    torch_data_train = GetLoader(x_train,y_train)
    torch_data_test=GetLoader(x_test,y_test)
    from torch.utils.data import DataLoader

    # 读取数据
    train_dataloader = DataLoader(torch_data_train, batch_size=32, shuffle=True, drop_last=False, num_workers=0)
    test_dataloader = DataLoader(torch_data_test, batch_size=32, shuffle=False, drop_last=False, num_workers=0)#这里的shuffle要换成false 才能最后对齐
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")
    
    # Define model
    import torch.nn.functional as F
    
    class Rnn(nn.Module):
      def __init__(self, in_dim, hidden_dim, n_layer, n_class, bidirectional):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True,
                            bidirectional=bidirectional)
        if self.bidirectional:
            self.classifier = nn.Linear(hidden_dim * 2, n_class)
        else:
            self.classifier = nn.Linear(hidden_dim, n_class)

      def forward(self, x):

        out, (hn, _) = self.lstm(x)
        if self.bidirectional:
            out = torch.hstack((hn[-2, :, :], hn[-1, :, :]))
        else:
            out = out[:, -1, :]
        out = self.classifier(out)
        return out
    class Attention(nn.Module):
      def __init__(self, rnn_size: int):
        super(Attention, self).__init__()
        self.w = nn.Linear(rnn_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

      def forward(self, H):
        # eq.9: M = tanh(H)
        M = self.tanh(H)  # (batch_size, word_pad_len, rnn_size)

        # eq.10: α = softmax(w^T M)
        alpha = self.w(M).squeeze(2)  # (batch_size, word_pad_len)
        alpha = self.softmax(alpha)  # (batch_size, word_pad_len)

        # eq.11: r = H
        r = H * alpha.unsqueeze(2)  # (batch_size, word_pad_len, rnn_size)
        r = r.sum(dim=1)  # (batch_size, rnn_size)

        return r, alpha


    class AttBiLSTM(nn.Module):
      def __init__(
            self,
            n_classes: int,
            emb_size: int,
            rnn_size: int,
            rnn_layers: int,
            dropout: float
    ):
        super(AttBiLSTM, self).__init__()

        self.rnn_size = rnn_size

        # bidirectional LSTM
        self.BiLSTM = nn.LSTM(
            emb_size, rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )

        self.attention = Attention(rnn_size)
        self.fc = nn.Linear(rnn_size, n_classes)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

      def forward(self, x):
        rnn_out, _ = self.BiLSTM(x)

        H = rnn_out[:, :, : self.rnn_size] + rnn_out[:, :, self.rnn_size:]

        # attention module
        r, alphas = self.attention(
            H)  # (batch_size, rnn_size), (batch_size, word_pad_len)

        # eq.12: h* = tanh(r)
        h = self.tanh(r)  # (batch_size, rnn_size)

        scores = self.fc(self.dropout(h))  # (batch_size, n_classes)

        return scores
      
    model = AttBiLSTM(2,edsize*edsize , 256, 2, 0.2).to(device)
    print(model)
    '''Using cuda device'''


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    '''In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and backpropagates the prediction error to adjust the model’s parameters.
    '''
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            #print(X.shape)

            # Compute prediction error
            pred = model(X)
            #print(y)
            y = y.squeeze(1)
            #print(y)
            y=y.to(torch.int64) 
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    from sklearn.metrics import roc_auc_score
    from sklearn import metrics 
    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            score=[]
            real_label=[]
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                y = y.squeeze(1)
                #print(pred[:,1],y)
                temp=pred[:,1].cpu().numpy().tolist()
                score+=temp
                real_label+=y.cpu().numpy().tolist()
                y=y.to(torch.int64) 
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        #print(score,real_label)
        
       
        test_loss /= num_batches
        correct /= size
        print(correct)
        auc=roc_auc_score(real_label,score)
        print("AUC:{:.4f}".format(auc))
        aupr=metrics.average_precision_score(real_label,score,average='macro',pos_label=1,sample_weight=None)
        print("AUPR:{:.4f}".format(aupr))
        #print("F1-Score:{:.4f}".format(f1_score(real_label,score)))
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct,auc,aupr,score,real_label
    epochs =200
    for t in range(epochs):
        if t%10==0:
          print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
    acc,auc,aupr,score,real_label=test(test_dataloader, model, loss_fn)
    listauc.append(round(auc,3))
    listaupr.append(round(aupr,3))
    key={'auc':auc,'aupr':aupr,'acc':acc,'f-fold':fepoch,'frs':0,'train_epoch':epochs,'divide_rs':divide_rs,'model_name':model_name}
    print("Done!")
   
    list_results.append(key)
    
    #tlable,dblabel,rnalabel
    #valid_model=r'.\\{}\\{}_frs{}_V{}_D{}_RS{}_epoch{}.m'.format(valid_model_dir,dataset,frs,vkey,dkey,rs,k)
    valid_result=r'.\\{}\\{}_frs{}_V{}_D{}_rs{}_epoch{}.csv'.format(valid_results_dir,dataset,frs,vkey,dkey,divide_rs,fepoch)
    DF_Ypredict=pd.DataFrame(real_label)
    DF_Ypredict=DF_Ypredict.rename(columns={'0':'Real_label'})
    #DF_Ypredict['real_label']=Y_test.values
    DF_Ypredict['predict_prob']=score
    DF_Ypredict['DBID1']=dblabel
    DF_Ypredict['DBID2']=rnalabel
    DF_Ypredict['Real_label']=tlabel
    #DF_Ypredict['predict_prob_2']=score[:,0].values
            #DF_Ypredict['original index']=y_test_index
    DF_Ypredict.to_csv(valid_result,index=None)                               
ls=pd.DataFrame(list_results)
from numpy import *
mauc=round(mean(listauc),3)
maupr=round(mean(listaupr),3)
ls.to_csv('{}_v{}_d{}_{}_auc{}_aupr{}_epoch{}.csv'.format(dataset,vkey,dkey,model_name,mauc,maupr,epochs),index=None)
#torch.save(model.state_dict(), r".\model\Torch_first_model.pth")
#print("Saved PyTorch Model State to model.pth")

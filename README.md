Addressing Imbalanced Drug-Drug Interaction Risk Levels in  Datasets using Pre-trained Deep Learning Model Embeddings (DDintensity)
Currently Under Revision 


Features generation:
The embedding generation rules (For NLP and CV embeddings) are the same as those in our previous work [1].
The code can be downloaded at:
http://deepseq2drug.cs.cityu.edu.hk/codes/

Deep-Learning information:  
torch                   2.0.0+cu118
torch-geometric         2.3.1
torchaudio              2.0.1+cu118
torchmetrics            0.11.4
torchvision             0.15.1+cu118

Steps of reproducing DDintensity:
Drug Features files:
Embeddings extracted from different pre-trained deep-learning models
The drug features extracted from different deep learning models can be found:
__________________
Constructed dataset:
DDI_Major1_114958.csv

* Step one:  building Embedding dataset  
First, make sure we have folders:0_drug_feature_pool  
and put unzipped drug features files into the folder 0_drug_feature_pool  
change 
keys='DinoVitb16' in dataset_construction.py  
run the file  
it will generate an embedding dataset for ML downstream tasks.  
Generally look like
For pre-train deep learning of DinoVit, the files should looks like:  
0DDI_Major1_vDinoVitb16_dDinoVitb16_frs_0_rs0.csv  
For pre-trained deep learning of Biogpt, the files should looks like:  
0DDI_Major1_vbioGPTesum_dbioGPTesum_frs_0_rs0.csv   
'DinoVitb16' ,'bioGPTesum' are the key to selecting the pre-trained deep learning models and an identifier for the following experiments.  

* Step 2: Building 5 folds embedding dataset
in .\Embedding dataset\：  
Copy-paste and rename the last number from 0-4:  
0DDI_Major1_v{}_d{}_frs_0_rs0.csv  
0DDI_Major1_v{}_d{}_frs_0_rs1.csv  
0DDI_Major1_v{}_d{}_frs_0_rs2.csv  
0DDI_Major1_v{}_d{}_frs_0_rs3.csv  
0DDI_Major1_v{}_d{}_frs_0_rs4.csv  
And run 
Step2 embedding 2 filesfolds.py  
Please change:  
vkey='DinoVitb16'  
dkey='DinoVitb16'  
Into the pre-trained deep learning models name you chose:  
We don't use any negative sampling.  
Thus, the imbalanced dataset can be the same. We only need to change the random seeds to cut the dataset.  
All 5folds embedding files are in:  
0DDI_Major1_feature_pod\5folds  

* Step3: run prediction files using:
the put the pytorch_read_csv(Attention_LSTMmodel) biogpt-.py into 
Embedding dataset\0DDI_Major1_feature_pod\  
run pytorch_read_csv(Attention_LSTMmodel) biogpt-.py  

Make sure you have corresponding 5-fold embedding files in:  
Embedding dataset\0DDI_Major1_feature_pod\5folds   
Should name like this:    
0DDI_Major1_v{key}_d{key}_frs_0_rs{rs for cutting file}_epoch{epoch}.csv  
0DDI_Major1_v{key}_d{key}_frs_0_rs{rs for cutting file}_epoch{epoch}_test.csv  
for e.g:  for biogpt model
0DDI_Major1_vbioGPTesum_dbioGPTesum_frs_0_rs0_epoch1.csv  
0DDI_Major1_vbioGPTesum_dbioGPTesum_frs_0_rs0_epoch1_test.csv    
In total of five-fold cross validation, and five repeation, it will generate 50 files for each experiments.  


it will automatically generate scores of the DDI_test in the valid_results folder  
and generate a scores file for five-fold cross-validation  

Reference
[1] W. Xie et al., ‘DeepSeq2Drug: An expandable ensemble end-to-end anti-viral drug repurposing benchmark framework by multi-modal embeddings and transfer learning’, Comput Biol Med, vol. 175, p. 108487, Jun. 2024, doi: 10.1016/j.compbiomed.2024.108487.

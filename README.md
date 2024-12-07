# Addressing Imbalanced Drug-Drug Interaction Risk Levels in  Datasets using Pre-trained Deep Learning Model Embeddings (DDintensity)  
Citation:  Currently Under Revision     


Embedding generation:  
The embedding generation rules (For NLP-based and CV-based embeddings) are the same as those in our previous work [1].
Those Codes/ Embeddings can be downloaded at:
http://deepseq2drug.cs.cityu.edu.hk/codes/

Deep-Learning information:  
torch                   2.0.0+cu118  
torch-geometric         2.3.1  
torchaudio              2.0.1+cu118  
torchmetrics            0.11.4  
torchvision             0.15.1+cu118  
or using requirements.txt.  

Some concepts in this manuscript:  
Original Datasets	Provide Positive Pairs/ Negative Pairs (Known Pairs)  
Constructed Datasets	Provide positive/ negative pairs with random seeds. (for drugbank dataset only)  
Embedding Datasets	Generate from constructed datasets via embedding files and provide training/ testing set for downstream machine-learning algorithms.  
Independent Datasets	To verify the performance of the trained model  

Dataset In this manuscript:  
Original Datasets	  
DDI2013 (Small, Imbalanced), Drug bank (Large, Balanced, Sample), DDinter (Large, Imbalanced),
Independent Datasets  
MecDDI

 Details of each dataset  
| Dataset|	Balanced	|Sampling	|DD-Type	|Positive |	Negative	|File_size|  
|DDI2013|	No|	No|	DDI|	637	|3491|	0.12GB|  
|DrugBank	|Yes	|Yes|	DDI	|191808	|191808|	11.57 GB|  
|DDinter	|No	|No	|DD intensity (Major or not)	|24647	|90311|	3.43GB|  


Steps of reproducing DDintensity:
Drug Features files:
Embeddings extracted from different pre-trained deep-learning models
The drug features extracted from different deep learning models can be found:
__________________
Constructed dataset:
DDI_Major1_114958.csv   *DDI_major means, DDinter dataset, Major as 1, the rest set as o.  
TrainDDI2013_4128.csv  

## * Step one:  building Embedding dataset  
First, make sure we have folders:
./0_drug_feature_pool  
and put unzipped drug features files into the folder ./0_drug_feature_pool
like this:
<img width="508" alt="1733402201568" src="https://github.com/user-attachments/assets/8659b95f-b1a3-4d7a-99e5-5c14e46f2a6b">

change or keep (depends on the embedding you select)
keys='DinoVitb16' in dataset_construction.py  
run the file

it will generate an embedding dataset for ML downstream tasks.  
Generally  
For pre-train deep learning of DinoVit, the files should looks like:  
0DDI2013_vDinoVit_dDinoVit_frs_0_rs0.csv  
For pre-trained deep learning of Biogpt, the files should looks like:  
0DDI2013_vbioGPTesum_dbioGPTesum_frs_0_rs0.csv
'DinoVitb16' ,'bioGPTesum' are the key to selecting the pre-trained deep learning models and an identifier for the following experiments.  
  <img width="402" alt="1733403069094" src="https://github.com/user-attachments/assets/784eae25-b110-47d7-8f2f-8e4d2f976688">


## * Step 2: Building 5 folds embedding dataset
### If multiple experiments are needed to validate statistical significance for the DDinter dataset, we can focus solely on the random seed used in five-fold cross-validation, as no negative sampling was performed.
in .\Embedding dataset\：  
Copy-paste and rename the last number from 0-4:  
0DDI_Major1_v{}_d{}_frs_0_rs0.csv  
0DDI_Major1_v{}_d{}_frs_0_rs1.csv  
0DDI_Major1_v{}_d{}_frs_0_rs2.csv  
0DDI_Major1_v{}_d{}_frs_0_rs3.csv  
0DDI_Major1_v{}_d{}_frs_0_rs4.csv
### For datasets that require negative sampling, such as Drugbank, we need to consider two random seeds: one for the negative sampling and another for the five-fold cross-validation.

And run 
Step2 embedding 2 filesfolds.py  
![image](https://github.com/user-attachments/assets/32f92112-54fe-4385-9cdb-49fc69c69866)
All 5folds embedding files are in:  
{Dataset}_feature_pod\5folds
Make sure you have corresponding 5-fold embedding files in:  
Embedding dataset\{dataset}_feature_pod\5folds   
Should name like this:    
0DDI_Major1_v{key}_d{key}_frs_0_rs{rs for cutting file}_epoch{epoch}.csv  
0DDI_Major1_v{key}_d{key}_frs_0_rs{rs for cutting file}_epoch{epoch}_test.csv  
for e.g:  for biogpt model
0DDI_Major1_vbioGPTesum_dbioGPTesum_frs_0_rs0_epoch1.csv  
0DDI_Major1_vbioGPTesum_dbioGPTesum_frs_0_rs0_epoch1_test.csv    
In total of five-fold cross validation, and five repeation, it will generate 50 files for each experiments.  
looks like:
![image](https://github.com/user-attachments/assets/0d5c4e73-e8a5-411f-b764-dd96b8b89e92)



## * Step3: run prediction files using:
the put the pytorch_read_csv(Attention_LSTMmodel) biogpt-DDI2013.py into 
Embedding dataset\{dataset}_feature_pod\  
![image](https://github.com/user-attachments/assets/7cdf62d6-b33f-467c-911c-6596ec2653f5)  
change how many times of repetition:  
![image](https://github.com/user-attachments/assets/bdada3da-f8b8-4406-b713-76dc651fb286)  
for example:
If we have only one file:  
0DDI2013_vbioGPTesum_dbioGPTesum_frs_0_rs0.csv
divide rs should range from 0 to 1.  
two files,repeat 2 times:  
0DDI2013_vbioGPTesum_dbioGPTesum_frs_0_rs0.csv  
0DDI2013_vbioGPTesum_dbioGPTesum_frs_0_rs1.csv   
divide rs should range from 0 to 2.  

change epochs here:  
<img width="158" alt="1733404148987" src="https://github.com/user-attachments/assets/719d5f43-06b2-4c01-8ad1-29935652c428">    
For DDI2013 epoch=2000  
For DDinter epoch=200  
For DrugBank epoch=200  

run pytorch_read_csv(Attention_LSTMmodel) biogpt-.py  
if it works, you should see:
<img width="362" alt="1733404292695" src="https://github.com/user-attachments/assets/06681c4b-b6aa-41a5-bd9f-e62d43aed84d">  
if it is DDinter dataset, More training data shows:  
![image](https://github.com/user-attachments/assets/217ab089-b5b8-4b8a-8ff9-4b53af8ea458)

Then wait:
it will automatically generate scores of the DDI_test in the valid_results folder  
<img width="446" alt="1733404349023" src="https://github.com/user-attachments/assets/3748be62-a6f2-4a6e-811d-8b4ec8609bc8">

After finishing:   
#### (to save time, only 200 epochs for DDI2013, AUC AND AUPR not satisfied, set epoch to 2000 would be better)  

<img width="236" alt="1733404830878" src="https://github.com/user-attachments/assets/669a7025-69ed-490f-af37-d1095ae1a408">   

#### for DDinter and Drugbank epoch 200;  
#### for Biogpt, 200 epochs, repeat 1,DDinter database:  '

<img width="375" alt="1733544998043" src="https://github.com/user-attachments/assets/27ff4202-07e5-439d-8071-6b4a6d06014f">


#### for Biogpt, 200 epochs, repeat 1,DDI2013:  
and generate a scores file for five-fold cross-validation
<img width="185" alt="1733404480190" src="https://github.com/user-attachments/assets/119f7679-fd9a-4f27-a0b2-fd1f41d6ae45">
![image](https://github.com/user-attachments/assets/fd2c8528-7195-4385-99cd-89b320d1c21a)

#### for Biogpt, 200 epochs, repeat 1,DDinter database:  
![image](https://github.com/user-attachments/assets/880524cb-a0df-485a-a810-c5b8698c74b1)


Reference:  
[1] W. Xie et al., ‘DeepSeq2Drug: An expandable ensemble end-to-end anti-viral drug repurposing benchmark framework by multi-modal embeddings and transfer learning’, Comput Biol Med, vol. 175, p. 108487, Jun. 2024, doi: 10.1016/j.compbiomed.2024.108487.

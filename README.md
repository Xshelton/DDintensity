Steps of reproducing DDintensity:
Drug Features files:
Embeddings extracted from different pre-trained deep learning models


Step one:  building Embedding dataset
First, make sure we have folders:0_drug_feature_pool
and put unzipped drug features files into the folder of 0_drug_feature_pool
change 
keys='DinoVitb16' in dataset_construction.py
run the file 
it will generate an embedding dataset for ML downstream tasks.

Step 2: buiding 5 folds embedding dataset
in Embedding dataset

Step3: run prediction files using:
pytorch_read_csv(Attention_LSTMmodel) biogpt-.py

it will automatically generate scores of DDI_test in valid_results folder
and generate a scores for five-fold cross-validation

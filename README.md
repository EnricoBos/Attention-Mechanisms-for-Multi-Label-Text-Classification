# TensorFlow Multi-Label Text Classification with Attention Mechanisms on Toxic Comment Dataset

* This project implements and compares multiple models for multi-label text classification using various attention mechanisms in TensorFlow. The comparison is based on the 'Toxic Comment Classification' dataset, which contains six classes. Each comment can belong to one or more classes. The Toxic Comment Classification dataset can be downloaded from Kaggle.


## Environment
* Python 3.10.10 
* Tensorflow V.2.10.0 

## Implementation
* Implemented models for multi label text classification including:
	-  bidirectional lstm with no attention mechanism
	-  bidirectional lstm with simple dot attention mechanism
	-  bidirectional lstm with bahdanau (additive) attention mechanism
	-  bidirectional lstm with single head attention mechanism
	-  bidirectional lstm with multi head attention mechanism




## Executing program
* To train a model, set var choice = 'enable_train' and model_type = '...' with the desired attention mechanism (For example, to use multi-head attention, set model_type = 'multi_head_attention'). The trained model will be saved in the specified folder.
* To evaluate the model's performance and save the confusion matrix as a PNG, set var choice = 'eval_performance'.

## Confusion matrix generated using multi-head attention

![multi_h_attention_confusion_matrix_percentages](https://github.com/user-attachments/assets/456baec8-5e8e-42d8-be12-f79c03680086)


## Authors

* Enrico Boscolo

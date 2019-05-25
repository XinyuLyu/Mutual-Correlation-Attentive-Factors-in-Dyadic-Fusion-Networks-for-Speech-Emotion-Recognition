## Mutual Correlation Attentive Factors in Dyadic Fusion Networks for Speech Emotion Recognition (ACM Multimedia 2019)
### News!! Rebutal: Strong accept, Strong accept, weak accept. May, 23th, 2019 
Here is just a demo of our research program. Our contribution includes:
1. Instead of using a recurrent neural network to extract temporal associations as in most previous research, we introduce multiple sub-view attention layers to compute the relevant dependencies among sequential utterances; this significantly improves model efficiency.   
2. To improve fusion performance, we design a learnable mutual correlation factor inside each attention layer to compute associations across different modalities.   
3. To overcome the label disagreement issue, we embed the labels from all annotators into a k-dimensional vector and transform the categorical problem into a regression problem; this method provides more accurate annotation information and fully uses the entire dataset.  
   We evaluate the proposed model on two published multimodal emotion recognition datasets: IEMOCAP and MELD. Our model significantly outperforms previous state-of-the-art research by 3.8%-7.5% accuracy, using a more efficient model.  
### Dataset: IEMOCAP DATABASE
https://sail.usc.edu/iemocap/

* Environments:
  1. Python 3.6
  2. Keras 2.0.9
  3. TensorFlow 1.3.0
  4. sklearn 0.19.1
  5. numpy 1.15.3
  6. scipy 1.1.0
* Hardware:
  1. Our model was trained on a GTX 1080 GPU with 32GB RAM.
  
* Codes: 

  1. attention.py: Modality Fusion with Mutual Correlation Attentive Factor.

  2. model.py : the hybrid model.

* data:  
For this study, we only use audio and text data. The dataset consists of 10039 utterances from 151 dialogs and contains 10 categories including ‘neutral’, ‘exciting’, ‘sadness’, ‘frustration’, ‘happiness’, ‘angry’, ‘other’, ‘surprised’, ‘disgust’, and ‘fear’. 
  1. label:  For each utterance, we include the labels from all annotators and embed it as a 10-dimensional vector. 
  2. We follow previous research to split the data into training, validation, and testing sets at the session level. 
  3. The split considers the speakers independent. The final dataset has 3 sessions for training, 1 session for validation, and 1 session for testing.
     (dev means validation dataset, test means test dataset, train means train dataset)
     
* model:
We provide a trained model, with the results shown below.  
acc:0.4951267056530214  
final result:   
{'9': 7, '7': 19, '0': 258, '8': 0, '3': 481, '5': 327, '4': 65, '1': 238, '2': 143, '6': 1}  
0 {'9': 0, '7': 0, '0': 60, '8': 0, '3': 91, '5': 5, '4': 41, '1': 54, '2': 7, '6': 0}  
1 {'9': 0, '7': 0, '0': 10, '8': 0, '3': 8, '5': 0, '4': 67, '1': 152, '2': 1, '6': 0}  
2 {'9': 0, '7': 0, '0': 10, '8': 0, '3': 30, '5': 1, '4': 2, '1': 2, '2': 98, '6': 0}  
3 {'9': 0, '7': 0, '0': 111, '8': 0, '3': 319, '5': 23, '4': 2, '1': 11, '2': 15, '6': 0}  
4 {'9': 0, '7': 0, '0': 4, '8': 0, '3': 3, '5': 0, '4': 23, '1': 33, '2': 2, '6': 0}  
5 {'9': 0, '7': 0, '0': 35, '8': 0, '3': 164, '5': 110, '4': 4, '1': 8, '2': 6, '6': 0}  
6 {'9': 0, '7': 0, '0': 0, '8': 0, '3': 1, '5': 0, '4': 0, '1': 0, '2': 0, '6': 0}  
7 {'9': 0, '7': 0, '0': 5, '8': 0, '3': 8, '5': 0, '4': 2, '1': 3, '2': 1, '6': 0}  
8 {'9': 0, '7': 0, '0': 0, '8': 0, '3': 0, '5': 0, '4': 0, '1': 0, '2': 0, '6': 0}  
9 {'9': 0, '7': 0, '0': 3, '8': 0, '3': 0, '5': 0, '4': 0, '1': 4, '2': 0, '6': 0}  

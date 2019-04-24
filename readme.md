## ACM Multimedia 2019  
Here is just a demo of our research program.
Our contribution includes:
1. A high-efficiency multimodal network that extracts the word-level textual and acoustic features only rely on multi-level attention mechanism and without using any convolutional neural networks and recurrent neural networks.
2. A hybrid architecture that combines the feature extraction module and fusion module into a single network, which uses two independent fusion factors to learn the across-modality associations in each attention layer during feature extraction.
3. An end-to-end modeling strategy that synchronizes feature representation and modality fusion, and extremely facilitates model training. 

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

  1. word2vec.py: Help to extract texture features.

  4. DataLoader_hybrid.py : Help load/save audio/text/label data to the hybrid model.
  
  5. Self_attention_hybrid.py: Define self-attention layer, combining the feature extraction module and fusion module into a single network.

* Models:

  1. final model: fusion model

  2. audio model: audio branch model
    
  3. text audio: texture model.

* data:

  1. audio: For audio data takes up 7 GB, we just put some samples in the filder.

  2. label_output_new.txt : The emotion labels for dataset.

  3. text_output_new.txt : Texture data. 

  4. glove.6B.50d: The embedding matrix for texture features. Actually, we use glove.6B.200d, but it is too large to put here.

  For the audio dataset is too large, we can't provide a demo for you to make. We have included our test results in the report with the models included in this folder. And if you have any questions about the project, please be free to email us.

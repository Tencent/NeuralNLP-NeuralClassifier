Configuration of NeuralClassifier uses JSON.

## Common

* **task\_info**
    * **label_type**:  Candidates: "single_label", "multi_label".
    * **hierarchical**: Boolean. Indicates whether it is a hierarchical classification.
    * **hierar_taxonomy**: A text file describes taxonomy. 
    * **hierar_penalty**: Float.
* **device**: Candidates: "cuda", "cpu".
* **model\_name**: Candidates: "FastText", "TextCNN", "TextRNN", "TextRCNN", "DRNN", "VDCNN", "DPCNN", "AttentiveConvNet", "Transformer".
* **checkpoint\_dir**: checkpoint directory
* **model\_dir**: model directory
* **data**
    * **train\_json\_files**: train input data.
    * **validate\_json\_files**: validation input data.
    * **test\_json\_files**: test input data.
    * **generate\_dict\_using\_json\_files**: generate dict using train data.
    * **generate\_dict\_using\_all\_json\_files**: generate dict using train, validate, test data.
    * **generate\_dict\_using\_pretrained\_embedding**: generate dict from pre-trained embedding.
    * **dict\_dir**: dict directory.
    * **num\_worker**: number of porcess to load data.


## Feature

* **feature\_names**: Candidates: "token", "char".
* **min\_token\_count**
* **min\_char\_count**
* **token\_ngram**: N-Gram, for example, 2 means bigram. 
* **min\_token\_ngram\_count**
* **min\_keyword\_count**
* **min\_topic\_count**
* **max\_token\_dict\_size**
* **max\_char\_dict\_size**
* **max\_token\_ngram\_dict\_size**
* **max\_keyword\_dict\_size**
* **max\_topic\_dict\_size**
* **max\_token\_len**
* **max\_char\_len**
* **max\_char\_len\_per\_token**
* **token\_pretrained\_file**: token pre-trained embedding.
* **keyword\_pretrained\_file**: keyword pre-trained embedding.


## Train

* **batch\_size**
* **eval\_train\_data**: whether evaluate training data when training.
* **start\_epoch**: start number of epochs.
* **num\_epochs**: number of epochs.
* **num\_epochs\_static\_embedding**: number of epochs that input embedding does not update.
* **decay\_steps**: decay learning rate every decay\_steps.
* **decay\_rate**: Rate of decay for learning rate.
* **clip\_gradients**: Clip absolute value gradient bigger than threshold.
* **l2\_lambda**: l2 regularization lambda value.
* **loss\_type**: Candidates: "SoftmaxCrossEntropy", "SoftmaxFocalCrossEntropy", "SigmoidFocalCrossEntropy", "BCEWithLogitsLoss".
* **sampler**: If loss type is NCE, sampler is needed. Candidate: "fixed", "log", "learned", "uniform".
* **num\_sampled**: If loss type is NCE, need to sample negative labels.
* **hidden\_layer\_dropout**: dropout of hidden layer.
* **visible\_device\_list**: GPU list to use.


## Embedding

* **type**: Candidates: "embedding", "region_embedding".
* **dimension**: dimension of embedding.
* **region\_embedding\_type**: config for Region embedding. Candidates: "word\_context", "context\_word".
* **region_size** region size, must be odd number. Config for Region embedding.
* **initializer**: Candidates: "uniform", "normal", "xavier\_uniform", "xavier\_normal", "kaiming\_uniform", "kaiming\_normal", "orthogonal".
* **fan\_mode**: Candidates: "FAN\_IN", "FAN\_OUT".
* **uniform\_bound**: If embedding_initializer is uniform, this param will be used as bound. e.g. [-embedding\_uniform\_bound,embedding\_uniform\_bound].
* **random\_stddev**: If embedding_initializer is random, this param will be used as stddev.
* **dropout**: dropout of embedding layer.


## Optimizer

* **optimizer\_type**: Candidates: "Adam", "Adadelta"
* **learning\_rate**: learning rate.
* **adadelta\_decay\_rate**: useful when optimizer\_type is Adadelta.
* **adadelta\_epsilon**: useful when optimizer\_type is Adadelta.


## Eval

* **text\_file**
* **threshold**: float trunc threshold for predict probabilities.
* **dir**: output dir of evaluation.
* **batch\_size**: batch size of evaluation.
* **is\_flat**: Boolean, flat evaluation or hierarchical evaluation.


## Log

* **logger\_file**: log file path.
* **log\_level**: Candidates: "debug", "info", "warn", "error".


## Encoder

### TextCNN

* **kernel\_sizes**: kernel size.
* **num\_kernels**: number of kernels.
* **top\_k\_max\_pooling**: max top-k pooling.

### TextRNN

* **hidden\_dimension**: dimension of hidden layer.
* **rnn\_type**: Candidates: "RNN", "LSTM", "GRU".
* **num\_layers**: number of layers.
* **doc\_embedding\_type**: Candidates: "AVG", "Attention", "LastHidden".
* **attention\_dimension**: dimension of self-attention.
* **bidirectional**: Boolean, use Bi-RNNs.

### RCNN

see TextCNN and TextRNN

### DRNN

* **hidden\_dimension**: dimension of hidden layer.
* **window\_size**: window size.
* **rnn\_type**: Candidates: "RNN", "LSTM", "GRU".
* **bidirectional**: Boolean.
* **cell\_hidden\_dropout**

### VDCNN

* **vdcnn\_depth**: depth of VDCNN.
* **top\_k\_max\_pooling**: max top-k pooling.

### DPCNN

* **kernel\_size**: kernel size.
* **pooling\_stride**: stride of pooling.
* **num\_kernels**: number of kernels.
* **blocks**: number of blocks for DPCNN.

### AttentiveConvNet

* **attention\_type**: Candidates: "dot", "bilinear", "additive_projection".
* **margin\_size**: attentive width, must be odd.
* **type**:  Candidates: "light", "advanced".
* **hidden\_size**: size of hidder layer.

### Transformer

* **d\_inner**: dimension of inner nodes.
* **d\_k**: dimension of key.
* **d\_v**: dimension fo value.
* **n\_head**: number of heads.
* **n\_layers**: number of layers.
* **dropout**
* **use\_star**: whether use Star-Transformer, see [Star-Transformer](https://arxiv.org/pdf/1902.09113v2.pdf "Star-Transformer") 

### HMCN 
* **hierarchical_depth**: hierarchical depth of each layer
* **global2local**: list of dimensions from global to local

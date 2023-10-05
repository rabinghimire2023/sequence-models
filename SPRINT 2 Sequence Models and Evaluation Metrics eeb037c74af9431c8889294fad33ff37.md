# SPRINT 2: Sequence Models and Evaluation Metrics

# What are Sequence models?

Machine learning models that input or output data sequences are known as sequence models. Text streams, audio clips, video clips, time-series data, and other types of sequential data are examples of sequential data.

### Some applications of Sequence models:

1. Speech recognition
2. Sentiment Classification
3. Video Activity Recognition

# What is Sequential Data?

When the points in the dataset are dependent on the other points in the dataset, the data is termed sequential. A Timeseries is a common example of this, with each point reflecting an observation at a certain point in time, such as a stock price or sensor data. Sequences, DNA sequences, and meteorological data are examples of sequential data.

![Sequential-data-can-be-ordered-in-many-ways-including-A-temperature-measured-over-time.png](SPRINT%202%20Sequence%20Models%20and%20Evaluation%20Metrics%20eeb037c74af9431c8889294fad33ff37/Sequential-data-can-be-ordered-in-many-ways-including-A-temperature-measured-over-time.png)

# Different Sequential Models

## Recurrent Neural Network

Recurrent Neural Network (RNN) is a Deep learning algorithm and it is a type of Artificial Neural Network architecture that is specialized for processing sequential data. RNNs are mostly used in the field of Natural Language Processing (NLP). **RNNs maintain internal memory**, and due to this, they are very efficient for machine learning problems that involve sequential data. RNNs are also used in time series predictions as well.

![download (1).png](SPRINT%202%20Sequence%20Models%20and%20Evaluation%20Metrics%20eeb037c74af9431c8889294fad33ff37/download_(1).png)

Figure 1: RNN (unfolded)

## Benefits of RNN Over CNN

/

1. Sequential Data Handling: RNNs are particularly well-suited for sequential data, where the order of the data points matters. They have a memory component that allows them to capture dependencies and patterns over time, making them ideal for tasks like natural language processing (NLP), speech recognition, and time series forecasting.
2. Variable-Length Sequences: RNNs can handle sequences of varying lengths, which makes them versatile for tasks involving text, audio, and other time-dependent data. CNNs, on the other hand, typically require fixed-size inputs.
3. Temporal Modeling: RNNs can model temporal dependencies, which means they can capture information from past time steps and use it to make predictions at the current time step. This is useful for tasks like predicting the next word in a sentence or forecasting future values in a time series.

# Computational Graphs

![download.png](SPRINT%202%20Sequence%20Models%20and%20Evaluation%20Metrics%20eeb037c74af9431c8889294fad33ff37/download.png)

Figure 2. Computational Graphs (Unfolded)

On a computational graph architecture, the Recurrent Neural Network consists of an Input layer, a Hidden Layer, and an output layer.

U, V, and W are the weight parameters that represent the weights associated with different layers. In this figure, U represents the weights associated between the inputs and the hidden layer, W represents the weights associated with hidden layers and V represents the weights associated between hidden layers and output layers.

The definition formula for RNN is:

$$
h_t= f(h_{t-1} ,x_t;\theta) -------------(1) 
$$

# RNN Forward Propagation

Hidden State:

$$
h_t = tanh(U_{x_t}+ Wh_{t-1})
$$

Output:

$$
z_t = Vh_t
$$

$$
\hat{y_t} = softmax(z_t)
$$

Loss:

$$
L_t = -y_tlog(\hat{y_t})
$$

$$
L = \Sigma_{j}^{T} L_j
$$

# Drawbacks of RNN

Recurrent Neural Networks (RNNs) are a powerful class of neural network architectures for sequential data, but they have several drawbacks that can limit their effectiveness in certain situations. Here are some of the main drawbacks of RNNs:

1. **Vanishing and Exploding Gradients**: RNNs are susceptible to the vanishing and exploding gradient problem. When gradients are backpropagated through many time steps during training, they can become extremely small (vanishing gradients) or extremely large (exploding gradients). This makes it difficult for the network to learn long-term dependencies in data.
2. **Difficulty in Capturing Long-Term Dependencies**: Standard RNNs have a limited ability to capture long-range dependencies in sequential data. This limitation makes them less effective in tasks where understanding the context from distant past time steps is crucial.
3. **Sequential Processing Limitation**: RNNs process data sequentially, one time step at a time. This sequential nature can lead to slow training and inference, making them less suitable for real-time applications.

# LSTM

The major drawback of a Recurrent Neural Network(RNN) is that the information is not retained in the memory when there is a large number of hidden timesteps. So, it suffers from vanishing gradient problems. The motivation behind the development of LSTM is to solve the vanishing gradient problem we face while training RNN for a long input sequence.

![download (2).png](SPRINT%202%20Sequence%20Models%20and%20Evaluation%20Metrics%20eeb037c74af9431c8889294fad33ff37/download_(2).png)

Figure: A single LSTM cell

# LSTM cell state

In a Long Short-Term Memory (LSTM) network, the cell state, often referred to as the "memory cell" or "cell state," is a crucial component that distinguishes LSTMs from standard recurrent neural networks (RNNs). The cell state serves as a long-term memory storage unit that can store and carry information over extended sequences. It is modified and controlled by various gates within the LSTM architecture. Here's an explanation of the LSTM cell state:

1. **Initialization**: At the beginning of processing a sequence, the cell state is initialized. This initialization can be set to a default value, often zeros, or learned during training.
2. **Information Flow**: The cell state interacts with the input and previous hidden state (short-term memory) through a set of gates, including the input gate, forget gate and output gate. These gates regulate the flow of information into and out of the cell state.
    - **Input Gate**: The input gate determines how much new information should be added to the cell state. It takes the current input and previous hidden state as input and produces a candidate update, which is then scaled by the input gate's output.
    - **Forget Gate**: The forget gate decides what information from the previous cell state should be discarded or "forgotten." It takes the current input and previous hidden state as input and produces a forgetting factor for each element in the cell state.
    - **Output Gate**: The output gate controls how much of the current cell state should be exposed to the next hidden state. It produces an output based on the current input and previous hidden state, which is then multiplied by the output gate's output to determine the next hidden state and the output of the LSTM cell.
3. **Memory Update**: The cell state is updated based on the information from the input gate, forget gate and output gate. The forget gate's output determines which elements of the previous cell state are retained, the input gate's output determines which new information is added, and the output gate's output determines which parts of the cell state are exposed.

# Bidirectional LSTM

A Bidirectional Long Short-Term Memory (BiLSTM) is a type of recurrent neural network (RNN) architecture that extends the traditional LSTM by processing input sequences in both forward and backward directions simultaneously. It combines information from past and future time steps to improve the understanding of context and dependencies within the sequence. BiLSTMs are widely used in natural language processing (NLP), speech recognition, and other sequential data tasks. Here's how a Bidirectional LSTM works:

1. **Forward and Backward Passes**: In a standard LSTM, information flows from the past to the future, processing the input sequence one time step at a time. In a BiLSTM, two separate LSTM layers are employed: one processes the sequence in the forward direction (from the beginning to the end), and the other processes it in the backward direction (from the end to the beginning).
2. **Hidden States**: Each LSTM layer (forward and backward) maintains its own hidden states, which capture information from the past and future of the input sequence, respectively. These hidden states are updated at each time step as the sequence is processed.
3. **Output Concatenation**: After both the forward and backward passes, the hidden states from each direction are concatenated element-wise for each time step. This creates a new representation for each time step that combines information from both the past and future, providing a more comprehensive view of the context.
4. **Final Output**: The concatenated hidden states can be used as the final output of the BiLSTM for various tasks, such as sequence classification, sentiment analysis, or named entity recognition. Alternatively, additional layers, such as fully connected layers or softmax layers, can be added on top of the concatenated outputs for specific tasks.

# Gated Recurrent Units (GRUs)

Gated Recurrent Units (GRUs) were introduced as a simplified alternative to Long Short-Term Memory (LSTM) networks, motivated by a desire to reduce the complexity of recurrent neural networks while maintaining their ability to capture long-term dependencies in sequential data. GRUs achieve this by using a simplified gating mechanism that combines the forget and input gates of LSTMs into a single update gate, resulting in fewer parameters and a more straightforward architecture. This reduction in complexity makes GRUs faster to train and computationally more efficient, while still offering competitive performance in tasks that require modeling sequential data.

![download (3).png](SPRINT%202%20Sequence%20Models%20and%20Evaluation%20Metrics%20eeb037c74af9431c8889294fad33ff37/download_(3).png)

Figure: A single GRU cell

# GRU gates

## Update gate

The update gate in a Gated Recurrent Unit (GRU) is a key component responsible for controlling how much of the previous cell state should be retained and how much of the new candidate cell state should be added to the current cell state at each time step in a recurrent neural network (RNN).

## Reset gate

The reset gate in a Gated Recurrent Unit (GRU) is a crucial component that helps control how much of the previous hidden state should be forgotten or reset at each time step in a recurrent neural network (RNN).

# RNN pros and cons

**Pros of RNNs:**

1. **Sequential Data Handling:** RNNs are specifically designed to work with sequential data, such as time series, text, and speech. They can capture temporal dependencies and patterns in the data, making them suitable for tasks like natural language processing, speech recognition, and time series prediction.
2. **Variable-Length Sequences:** RNNs can handle sequences of varying lengths, making them versatile for tasks where input data length is not fixed, unlike some other neural network architectures like CNNs.
3. **Online Learning:** RNNs are capable of online or incremental learning, meaning they can update their internal state and adapt to new information as it arrives in a streaming fashion. This is useful for applications that require continuous learning.
4. **Generative Modeling:** RNNs can be used for generative tasks, such as text generation, music composition, and image captioning, where the network learns to produce sequences of data.
5. **Natural Language Processing (NLP):** RNNs have been widely employed in NLP tasks, such as machine translation, sentiment analysis, and named entity recognition, due to their ability to model sequential text data effectively.

**Cons of RNNs:**

1. **Vanishing and Exploding Gradients:** RNNs are susceptible to the vanishing gradient problem, which hampers their ability to capture long-term dependencies. In some cases, gradients can also explode during training, making optimization difficult.
2. **Limited Context:** Standard RNNs have a limited context window, meaning they can only consider a fixed number of previous time steps. This makes it challenging to capture very long-term dependencies in data.
3. **Sequential Processing:** RNNs process data sequentially, which can lead to slow training and inference times, especially when dealing with long sequences. Parallelization is limited.
4. **Difficulty with Irregular Time Steps:** RNNs assume equally spaced time steps, which may not hold true in some real-world applications. Dealing with irregular time steps can be challenging.
5. **Sensitivity to Hyperparameters:** RNNs are sensitive to hyperparameters, including the learning rate, the choice of activation functions, and the network architecture. Finding the right set of hyperparameters can be time-consuming.
6. **Memory Requirements:** RNNs can be memory-intensive, especially when dealing with deep networks and large batch sizes. This can be a limitation in resource-constrained environments.
7. **Difficulty in Capturing Non-Sequential Patterns:** While RNNs excel at capturing sequential patterns, they may not perform well on tasks that require understanding non-sequential patterns and long-range dependencies.

# Neural Machine Translation

 

NMT is a large neural network that is trained in an end-to-end fashion for translating one language into another. Neural Machine Translation (NMT) is a subfield of machine translation that uses artificial neural networks, particularly deep learning models, to automatically translate text or speech from one language to another. NMT has significantly improved the quality of machine translation and has become the dominant approach in the field.

![1_y0fDASwnhK1buork4sGBcg.webp](SPRINT%202%20Sequence%20Models%20and%20Evaluation%20Metrics%20eeb037c74af9431c8889294fad33ff37/1_y0fDASwnhK1buork4sGBcg.webp)

Figure: Neural Machine Translation 

# Sequence to Sequence Architecture

The Sequence-to-Sequence (Seq2Seq) architecture is a neural network architecture used in various natural language processing (NLP) tasks, including machine translation, text summarization, and speech recognition. It is designed to handle input and output sequences of arbitrary lengths, making it suitable for tasks where the input and output have different lengths or where context matters.

## Encoder-Decoder Model:

- **Encoder**: The encoder is responsible for processing the input sequence and encoding it into a fixed-length context or representation, often referred to as the "thought vector." This context captures the information from the input sequence and serves as the foundation for generating the output sequence. In machine translation, for example, the encoder reads the source sentence.
- **Decoder**: The decoder takes the encoded context from the encoder and generates the output sequence step by step. It typically uses another recurrent neural network (RNN) or a similar architecture. In machine translation, the decoder produces the target sentence.

![download 12.png](SPRINT%202%20Sequence%20Models%20and%20Evaluation%20Metrics%20eeb037c74af9431c8889294fad33ff37/download_12.png)

## NMT without attention

NMT without attention can produce reasonable translations, it has some limitations compared to attention-based NMT:

1. **Handling Long Sequences**: NMT without attention may struggle with very long input sequences because it relies on compressing all the information into a fixed-size context vector. Attention mechanisms allow the model to focus on relevant parts of the input, making it better suited for handling long sequences.
2. **Alignment**: Without attention, it can be challenging for the model to learn word alignments between the source and target languages. Attention mechanisms explicitly capture these alignments, which improves translation quality.
3. **Limited Context**: The fixed-size context vector may not capture all the relevant context from the input sequence, leading to suboptimal translations, especially in cases where word order and context are crucial.

# Attention Mechanism

It enables models to focus on specific parts of input data when making predictions, allowing them to capture and weigh the relevance of different elements within the input. Attention mechanisms have greatly improved the performance of various machine learning tasks and are a cornerstone of many state-of-the-art models.

Attention takes two sentences  and turns them into a matrix where the words of one sentence form the columns, and the words of another sentence form the rows, and then it makes matches, identifying relevant context. This is very useful in machine translation.

# ****Types Of Attention Mechanism****

### 1. Generalized Attention

When a sequence of words or an image is fed to a generalized attention model, it verifies each element of the input sequence and compares it against the output sequence. So, each iteration involves the mechanism's encoder capturing the input sequence and comparing it with each element of the decoder's sequence.

### 2. Self-Attention

The self-attention mechanism is also sometimes referred to as the intra-attention mechanism. It is so-called because it picks up particular parts at different positions in the input sequence and over time it computes an initial composition of the output sequence.

### 3. Multi-Head Attention

Multi-head attention is a transformer model of attention mechanism. When the attention module repeats its computations over several iterations, each computation forms parallel layers known as attention heads. Each separate head independently passes the input sequence and corresponding output sequence element through a separate head.

### 4. Additive Attention

This type of attention also known as the Bahdanau attention mechanism makes use of attention alignment scores based on a number of factors. These alignment scores are calculated at different points in a neural network. Source or input sequence words are correlated with target or output sequence words but not to an exact degree.

### 5. Global Attention

This type of attention mechanism is also referred to as the Luong mechanism. This is a multiplicative attention model which is an improvement over the Bahdanau model. In situations where neural machine translations are required, the Luong model can either attend to all source words or predict the target sentence, thereby attending to a smaller subset of words.

# BLEU Score

[BLEU (**B**ilingual **E**valuation **U**nderstudy)](https://en.wikipedia.org/wiki/BLEU) is a metric for automatically evaluating machine-translated text. The BLEU score is a number between zero and one that measures the similarity of the machine-translated text to a set of high-quality reference translations. A value of 0 means that the machine-translated output has no overlap with the reference translation (low quality) while a value of 1 means there is perfect overlap with the reference translations (high quality).

The Mathematical defination of the BLEU score is:

$$
\text{BLEU} = \underbrace{\vphantom{\prod_i^4}\min\Big(1,
       \exp\big(1-\frac{\text{reference-length}}
    {\text{output-length}}\big)\Big)}_{\text{brevity penalty}}
 \underbrace{\Big(\prod_{i=1}^{4}
    precision_i\Big)^{1/4}}_{\text{n-gram overlap}}
$$

with

$$
precision_i = \dfrac{\sum_{\text{snt}\in\text{Cand-Corpus}}\sum_{i\in\text{snt}}\min(m^i_{cand}, m^i_{ref})}
 {w_t^i = \sum_{\text{snt'}\in\text{Cand-Corpus}}\sum_{i'\in\text{snt'}} m^{i'}_{cand}}
$$

where,

$$
m_{cand}^i\hphantom{xi} 
$$

is the count of i-gram in candidate matching the reference translation.

$$
m_{ref}^i\hphantom{xxx}
$$

is the count of i-gram in the reference translation.

$$
w_t^i\hphantom{m_{max}}
$$

is the total number of i-grams in candidate translation.

The formula consists of two parts: the brevity penalty and the n-gram overlap.

1. Brevity penalty

The brevity penalty penalizes generated translations that are too short compared to the closest reference length with an exponential decay. The brevity penalty compensates for the fact that the BLEU score has no [recall](https://developers.google.com/machine-learning/crash-course/glossary#recall) term.

1. N-Gram Overlap

The n-gram overlap counts how many unigrams, bigrams, trigrams, and four-grams (*i*=1,...,4) match their n-gram counterpart in the reference translations. This term acts as a [precision](https://developers.google.com/machine-learning/crash-course/glossary#precision) metric. Unigrams account for *adequacy* while longer n-grams account for *fluency* of the translation.

| BLEU Score | Interpretation |
| --- | --- |
| < 10 | Almost useless |
| 10 - 19 | Hard to get the gist |
| 20 - 29 | The gist is clear, but has significant grammatical errors |
| 30 - 40 | Understandable to good translations |
| 40 - 50 | High quality translations |
| 50 - 60 | Very high quality, adequate, and fluent translations |
| > 60 | Quality often better than human |

### Properties

- **BLEU is a Corpus-based Metric**
    
    The BLEU metric performs badly when used to evaluate individual sentences. For example, both example sentences get very low BLEU scores even though they capture most of the meaning. Because n-gram statistics for individual sentences are less meaningful, BLEU is by design a corpus-based metric; that is, statistics are accumulated over an entire corpus when computing the score. Note that the BLEU metric defined above cannot be factorized for individual sentences.
    
- **No distinction between content and function words**
    
    The BLEU metric does not distinguish between content and function words, that is, a dropped function word like "a" gets the same penalty as if the name "NASA" were erroneously replaced with "ESA".
    
- **Not good at capturing meaning and grammaticality of a sentence**
    
    The drop of a single word like "not" can change the polarity of a sentence. Also, taking only n-grams into account with n≤4 ignores long-range dependencies and thus BLEU often imposes only a small penalty for ungrammatical sentences. 
    
- **Normalization and Tokenization**
    
    Prior to computing the BLEU score, both the reference and candidate translations are normalized and tokenized. The choice of normalization and tokenization steps significantly affect the final BLEU score.
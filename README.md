## Machine Learning

This subject is designed to give a learner an in-depth understanding of the methodologies, technologies, mathematics and algorithms currently used in machine learning. Learners gain exposure the theory behind a range of machine learning tools and practice applying the tools to different applications. It covers topics such as classification, linear models, learning theory, generative models, graphical models and learning paradigm. This subject covers theoretical concepts such as inductive bias, the PAC learning framework, Bayesian learning methods, margin-based learning, and Occam’s Razor. Learners will be given short programming assignments that include hands-on experiments using machine learning algorithms and methods.


A course on neural networks that starts all the way at the basics. The course is a series of YouTube videos where we code and train neural networks together. The Jupyter notebooks we build in the videos are then captured here inside the [lectures](lectures/) directory. Every lecture also has a set of exercises included in the video description. (This may grow into something more respectable).

---

**Lecture 1: Career opportunities, key concepts, applications, types of algorithms and software (Python SciPy/Numpy and BigML )**

Where are the surging career opportunities and fastest-growing companies in the world? Machine learning (ML), a subset of Artificial Intelligence (AI).

ML is a collection of algorithms and programming techniques using statistical models for automatically discovering patterns and relationships in data. With ML algorithms, you can use historical data to make predictions for sales trends, risk analysis, weather and other forecasts. Alternatively, ML algorithms can predict clusters and classify data for tasks such as making recommendations for places to visit or shopping items and fraud detection. The hype around ML and strong association with AI stem primarily from predictive capability. An ability to predict associates strongly with the intelligence of humans. 

This subject focuses on classical ML or shallow learning. In shallow learning, the extraction of features (something measurable) from the input data is manual and requires domain expertise. This contrasts with deep learning (DL) which is about automatically learning features to eventually recognise differences between different objects, animals or humans without having to describe the key characteristics. DL achieves an output through processes mimicking the operation of the human brain (neural networks). Multiple layers of neural networks (hence the name "deep") extract knowledge and transform the raw data into higher-level features comprising the object, animal or human. Even though DL is seen to be "cutting edge", the technology is not without drawbacks. DL requires substantial infrastructure, the black-box nature (the internal mechanism is similar to a software version of the human brain represented by layers of neural networks and not readily understood)prevents an ability to explain what is going on in human terms. Additionally, DL suffers from magnifying the challenges facing shallow learning (ML). This includes the tuning of parameters for optimal learning (hyperparameter tuning) as well as bias in data. Hence, a shallow ML solution first mindset to solving problems is essential. This subject itself will not cover DL any further. To understand more about DL see ISY503 Intelligent Systems.  

The main types of ML algorithms are supervised learning, wherein labelled input examples are provided for the learning by the algorithm. Labels represent the object or output of our focus. The second type is unsupervised learning, with no labelling of the input data. The majority of ML projects utilise a common set of algorithms. These algorithms and software code are available as part of the open-source scikit-learn (https://scikit-learn.org/stable/) Python library. Classical or shallow algorithms include regression, support vector machines, decision trees, K-means and perceptron. During this course, you will gain hands on experience with each of these ML algorithms.

Previously, these type of ML algorithms were the domain of university researchers and the major technology vendors. Open source Jupyter notebook, Google Colab, Python ecosystem of libraries and cloud services ensure cloud-based ML is available for everyone.
 

- [YouTube video lecture](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Jupyter notebook files](lectures/micrograd)
- [micrograd Github repo](https://github.com/karpathy/micrograd)

---

**Lecture 2: Managing ML projects: CRISP-DM, ethics by design (Australasia) and datasets**

We implement a bigram character-level language model, which we will further complexify in followup videos into a modern Transformer language model, like GPT. In this video, the focus is on (1) introducing torch.Tensor and its subtleties and use in efficiently evaluating neural networks and (2) the overall framework of language modeling that includes model training, sampling, and the evaluation of a loss (e.g. the negative log likelihood for classification).

- [YouTube video lecture](https://www.youtube.com/watch?v=PaCmpygFfXo)
- [Jupyter notebook files](lectures/makemore/makemore_part1_bigrams.ipynb)
- [makemore Github repo](https://github.com/karpathy/makemore)

---

**Lecture 3: Supervised Learning and Linear Regression**

We implement a multilayer perceptron (MLP) character-level language model. In this video we also introduce many basics of machine learning (e.g. model training, learning rate tuning, hyperparameters, evaluation, train/dev/test splits, under/overfitting, etc.).

- [YouTube video lecture](https://youtu.be/TCH_1BHY58I)
- [Jupyter notebook files](lectures/makemore/makemore_part2_mlp.ipynb)
- [makemore Github repo](https://github.com/karpathy/makemore)

---

**Lecture 4: Decision trees**

We dive into some of the internals of MLPs with multiple layers and scrutinize the statistics of the forward pass activations, backward pass gradients, and some of the pitfalls when they are improperly scaled. We also look at the typical diagnostic tools and visualizations you'd want to use to understand the health of your deep network. We learn why training deep neural nets can be fragile and introduce the first modern innovation that made doing so much easier: Batch Normalization. Residual connections and the Adam optimizer remain notable todos for later video.

- [YouTube video lecture](https://youtu.be/P6sfmUTpUmc)
- [Jupyter notebook files](lectures/makemore/makemore_part3_bn.ipynb)
- [makemore Github repo](https://github.com/karpathy/makemore)

---

**Lecture 5: Generative models:Bayes Rule Maximum Likelihood Estimation**

We take the 2-layer MLP (with BatchNorm) from the previous video and backpropagate through it manually without using PyTorch autograd's loss.backward(). That is, we backprop through the cross entropy loss, 2nd linear layer, tanh, batchnorm, 1st linear layer, and the embedding table. Along the way, we get an intuitive understanding about how gradients flow backwards through the compute graph and on the level of efficient Tensors, not just individual scalars like in micrograd. This helps build competence and intuition around how neural nets are optimized and sets you up to more confidently innovate on and debug modern neural networks.

I recommend you work through the exercise yourself but work with it in tandem and whenever you are stuck unpause the video and see me give away the answer. This video is not super intended to be simply watched. The exercise is [here as a Google Colab](https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wupFQX0wh5ZC-z-?usp=sharing). Good luck :)

- [YouTube video lecture](https://youtu.be/q8SA3rM6ckI)
- [Jupyter notebook files](lectures/makemore/makemore_part4_backprop.ipynb)
- [makemore Github repo](https://github.com/karpathy/makemore)

---

**Lecture 6: Support Vector Machines**

We take the 2-layer MLP from previous video and make it deeper with a tree-like structure, arriving at a convolutional neural network architecture similar to the WaveNet (2016) from DeepMind. In the WaveNet paper, the same hierarchical architecture is implemented more efficiently using causal dilated convolutions (not yet covered). Along the way we get a better sense of torch.nn and what it is and how it works under the hood, and what a typical deep learning development process looks like (a lot of reading of documentation, keeping track of multidimensional tensor shapes, moving between jupyter notebooks and repository code, ...).

- [YouTube video lecture](https://youtu.be/t3YJ5hKiMQ0)
- [Jupyter notebook files](lectures/makemore/makemore_part5_cnn1.ipynb)

---


**Lecture 7: Explainable and Automated ML**

We build a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3. We talk about connections to ChatGPT, which has taken the world by storm. We watch GitHub Copilot, itself a GPT, help us write a GPT (meta :D!) . I recommend people watch the earlier makemore videos to get comfortable with the autoregressive language modeling framework and basics of tensors and PyTorch nn, which we take for granted in this video.

- [YouTube video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY). For all other links see the video description.

---

**Lecture 8: Classification and Logistic Regression**

We build a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3. We talk about connections to ChatGPT, which has taken the world by storm. We watch GitHub Copilot, itself a GPT, help us write a GPT (meta :D!) . I recommend people watch the earlier makemore videos to get comfortable with the autoregressive language modeling framework and basics of tensors and PyTorch nn, which we take for granted in this video.

- [YouTube video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY). For all other links see the video description.

---

**Lecture 9: K-means clustering**

We build a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3. We talk about connections to ChatGPT, which has taken the world by storm. We watch GitHub Copilot, itself a GPT, help us write a GPT (meta :D!) . I recommend people watch the earlier makemore videos to get comfortable with the autoregressive language modeling framework and basics of tensors and PyTorch nn, which we take for granted in this video.

- [YouTube video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY). For all other links see the video description.

---

**Lecture 10:Learning Theory: PAC**

We build a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3. We talk about connections to ChatGPT, which has taken the world by storm. We watch GitHub Copilot, itself a GPT, help us write a GPT (meta :D!) . I recommend people watch the earlier makemore videos to get comfortable with the autoregressive language modeling framework and basics of tensors and PyTorch nn, which we take for granted in this video.

- [YouTube video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY). For all other links see the video description.


**Lecture 11: Perceptron**

We build a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3. We talk about connections to ChatGPT, which has taken the world by storm. We watch GitHub Copilot, itself a GPT, help us write a GPT (meta :D!) . I recommend people watch the earlier makemore videos to get comfortable with the autoregressive language modeling framework and basics of tensors and PyTorch nn, which we take for granted in this video.

- [YouTube video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY). For all other links see the video description.

---

**Lecture 12:Future of ML: Enterprise Grade ML and Weak Supervision**

We build a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3. We talk about connections to ChatGPT, which has taken the world by storm. We watch GitHub Copilot, itself a GPT, help us write a GPT (meta :D!) . I recommend people watch the earlier makemore videos to get comfortable with the autoregressive language modeling framework and basics of tensors and PyTorch nn, which we take for granted in this video.

- [YouTube video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY). For all other links see the video description.

---



---



Ongoing...

**License**

MIT

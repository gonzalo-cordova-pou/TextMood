---
language:
- en
tags:
- Natural language processing
- Sentiment classification
- Mood detection
- Tweets analysis
datasets:
- Sentiment140
metrics:
- Accuracy
- Cross entropy loss
results on test data:
- Accuracy = 0.688
- Cross entropy loss = 0.491
co2_eq_emissions:
emissions: number (in grams of CO2)
source: "source of the information, either directly from AutoTrain, code
carbon or from a scientific article documenting the model"
training_type: "pre-training or fine-tuning"
geographical_location: "as granular as possible, for instance Quebec, Canada
or Brooklyn, NY, USA. To check your compute's electricity grid, you can check
out https://app.electricitymap.org."
hardware_used: "how much compute and what kind, e.g. 8 v100 GPUs"
---
# TextMood Model Card

For a QuickStart click [here](./doc/gettingStarted.md).

## Table of Contents
- [Model details](#Model-details)
- [Intended use](#Intended-use)
- [Factors](#Factors)
- [Metrics](#Metrics)
- [Training data](#Training-data)
- [Evaluation data](#Evaluation-data)
- [Quantitative analyses](#Quantitative-analyses)
- [Ethical considerations](#Ethical-considerations)

### Model details
* Sentiment classification task with deep neural networks using social networks data
* In particular, detection of users' mood classifying it as positive or negative by simply reading their tweets
* Developed by the TextMood team in the context of TAED II course
* Model date: September 2022
* Model version: 3.0
* Send questions or comments about the model to textmoodupc@gmail.com

![Model architecture](./static/new_nn.jpg)
### Intended use
* Intended to be strictly used to detect and classify user's mood. It is not allowed to benefit or make a profit from this information.
* Not intended to make judgments about specific users
### Factors
* Subjectivity when evaluating the polarity of the tweet (0 = negative, 4 = positive) may affect the performance and trustworthiness of the model
* The model just evaluates the language. Other factors such as users' race, gender, age or health are not taken into account as the data used are simply tweets extracted by the Twitter API without collecting user's personal information.
### Metrics
* Model trained using tl.CrossEntropyLoss optimized with the trax.optimizers.Adam optimizer
* Tracking the accuracy using tl.Accuracy metric. We also track tl.CrossEntropyLoss on the validation set.
### Training data
* Dataset used: Sentiment140 dataset with 1.6 million tweets https://www.kaggle.com/datasets/kazanova/sentiment140 (80% used for training)
* Tweets preprocessing: Removing stop words, stemming, removing hyperlinks and hashtags (only the sign # from the word) and tokenizing the tweets. Once the tweet is cleaned, we convert it to a tensor (using tweet2tensor function).
* For further information such as dataset description, structure and creation see our Dataset Card: https://github.com/gonzalo-cordova-pou/TextMood/blob/main/DatasetCard.md
### Evaluation data
* Same dataset: Sentiment140 dataset with 1.6 million tweets https://www.kaggle.com/datasets/kazanova/sentiment140 (20% remaining used for testing)
* The same preprocessing steps are applied as we use the same dataset
### Quantitative analyses

We executed three different versions of our model and we tracked their results using
MLflow. As we can see, we used three combinations of the parameters batch_size and epochs
(32-100, 32-50 and 16-50) and compared the accuracy and the cross entropy loss values
obtained on test data.

We selected the second model (batch_size = 32 and epochs = 50) as the best one, as it has the
highest accuracy value (the third too) and the lowest cross entropy loss value. Naturally, we
are open to improving these models with new techniques during the next few weeks, tracking
their results and updating them.

![Results](./static/quantitative_analysis.png)
### Ethical considerations
* TextMood team follows values such as transparency, privacy, non-discrimination and societal and environmental wellbeing
* As previosuly stated, this model cannot be used for gaining personal or commercial profit by knowing users' mood.

---
language:
- en
tags:
- Natural language processing
- Classification
- Tweets analysis
- Mood detection
datasets:
- Sentiment140
metrics:
- Accuracy
- Cross entropy loss
---
# TextMood Model Card

### Model details
* Sentiment analysis task with deep neural network using social network data
* In particular, detection of users' mood classifying it as positive or negative by simply reading their tweets
* Developed by the TextMood team in the context of TAED II course
* Model date: September 2022
* Model version: 2.0
* Send questions or comments about the model to textmoodupc@gmail.com

![Model architecture](./static/nn.jpg)

### Intended use
* Intended to be used by corportations to detect and analyse user's mood by reading their social text messages
* Not intended to make judgments about specific users
### Factors
* Subjectivity when evaluating the polarity of the tweet (0 = negative, 4 = positive) may affect the performance and trustworthiness of the model
* The model just evaluates the language. Other factors such as users' race, gender, age or health are not taken into account as the data used are simply tweets extracted by the Twitter API without collecting user's personal information.
### Metrics
* Model trained using tl.CrossEntropyLoss optimized with the trax.optimizers.Adam optimizer
* Tracking the accuracy using tl.Accuracy metric. We also track tl.CrossEntropyLoss on the validation set.
### Training data
* Dataset: Sentiment140 dataset with 1.6 million tweets https://www.kaggle.com/datasets/kazanova/sentiment140 (80% used for training)
* Preprocessing...
* For further information see the Dataset Card
### Evaluation data
* Same dataset: Sentiment140 dataset with 1.6 million tweets https://www.kaggle.com/datasets/kazanova/sentiment140 (20% remaining used for testing)
### Quantitative analyses
*
### Ethical considerations
* TextMood team follows values such as transparency, privacy, non-discrimination and societal and environmental wellbeing

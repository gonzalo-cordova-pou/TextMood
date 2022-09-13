# TextMood Model - Model Card

### Model details
* Sentiment analysis with deep neural network using social network data
* Developed by the TextMood team in the context of TAED II course
* Model date: September 2022
* Model version: 1.0
* Send questions or comments about the model to ...

![Model architecture](./static/nn.jpg)


### Intended use
* Intended to be used by corportations to detect and analyse user's mood by reading their social text messages
* Not intended to make judgments about specific users
### Factors
* Subjectivity when evaluating the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive) may affect the performance and trustworthiness of the model
* Other factors such as race, gender, sexual orientation or health are not taken into account as the data used are simply tweets extracted by the Twitter API
### Metrics
*
### Training data
* Dataset: Sentiment140 dataset with 1.6 million tweets https://www.kaggle.com/datasets/kazanova/sentiment140 (%)
* Preprocessing...
* For further information see the Dataset Card
### Evaluation data
* Dataset: Sentiment140 dataset with 1.6 million tweets https://www.kaggle.com/datasets/kazanova/sentiment140 (%)
### Quantitative analyses
*
### Ethical considerations
* TextMood team follows values such as transparency, privacy, non-discrimination and societal and environmental wellbeing
### Caveats and recommendations

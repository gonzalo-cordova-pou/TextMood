---
language:
- en
language_creators:
- found
multilinguality:
- monolingual
pretty_name: Sentiment Analysis
size_categories:
- 1M<n<10M
source_datasets:
- original
tags:
- twitter
- tweet
- tweets
task_categories:
- text-classification
task_ids:
- multi-label-classification
- sentiment-classification
---

# Dataset Card for Sentiment140

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Initial Data Collection and annotation](#Initial-Data-Collection-and-annotation)
## Dataset Description

- **Homepage:** [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

### Dataset Summary

The sentiment140 dataset contains 1,600,000 tweets extracted using the twitter api. The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment.

### Supported Tasks and Leaderboards

* `twitter-sentiment-analysis` The dataset can be used to train a model to predict user sentiment, which consists in detecting the mood of the users from the tweets they wrote. Success on this is typically measured by achieving high accuracy. For example, we have found a [model](https://www.kaggle.com/code/paoloripamonti/twitter-sentiment-analysis) that currently achieves the following accuracy score: 0.791134375.

### Languages

The text in the dataset is in English. The associated BCP-47 code is `en`.

## Dataset Structure

### Data Instances

A typical data point comprises a `text`, which is the content of the tweet, with some information extra about the tweet, such as the `user` or the `date`. There is also the `target` variable, which indicates the mood level of the tweet.

```
{
  'target': 0,
  'id': 1467810672,
  'date': Mon Apr 06 22:19:49 PDT 2009,
  'flag': NO_QUERY,
  'user': scotthamilton,
  'text': is upset that he can't update his Facebook by texting it... and might cry as a result School today ...
}
```


### Data Fields

* `target`: the polarity of the tweet (0 = negative, 4 = positive).
* `ids`: The id of the tweet (2087).
* `date`: the date of the tweet (Sat May 16 23:58:44 UTC 2009).
* `flag`: The query (lyx). If there is no query, then this value is NO_QUERY.
* `user`: the user that tweeted (robotickilldozr).
* `text`: the text of the tweet (Lyx is cool).

### Data Splits

The data is not split, so the user is expected to do it in order to train and test the model.

## Dataset Creation

#### Initial Data Collection and annotation

Dataset creators highlight that their approach was unique because the training data was automatically created, as opposed to having humans manual annotate tweets. They assume that any tweet with positive emoticons, like :), were positive, and tweets with negative emoticons, like :(, were negative. They used the Twitter Search API to collect these tweets by using keyword search.


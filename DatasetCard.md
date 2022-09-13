---
TODO: Add YAML tags here. Copy-paste the tags obtained with the online tagging app: https://huggingface.co/spaces/huggingface/datasets-tagging
---

# Dataset Card for Sentiment140

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

### Dataset Summary

The sentiment140 dataset contains 1,600,000 tweets extracted using the twitter api. The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment.

### Supported Tasks and Leaderboards

* `twitter-sentiment-analysis` The dataset can be used to train a model to predict user sentiment, which consists in detecting the mood of the users from the tweets they wrote. Success on this is typically measured by achieving high accuracy. The model currently achieves the following score: 0.791134375.

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

* `target`: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive).
* `ids`: The id of the tweet (2087).
* `date`: the date of the tweet (Sat May 16 23:58:44 UTC 2009).
* `flag`: The query (lyx). If there is no query, then this value is NO_QUERY.
* `user`: the user that tweeted (robotickilldozr).
* `text`: the text of the tweet (Lyx is cool).

### Data Splits

The data is not split, so the user is expected to do it in order to train and test the model.

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

[More Information Needed]

### Annotations

#### Annotation process

[More Information Needed]

#### Who are the annotators?

[More Information Needed]

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

[More Information Needed]

### Citation Information

[More Information Needed]

### Contributions

Thanks to [@github-username](https://github.com/<github-username>) for adding this dataset.

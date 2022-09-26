The directory structure of our project: 

```
├── data
│   └── raw
│       └── training.1600000.processed.noemoticon.csv.dvc
├── src                  <- Source code for use in this project 
│   ├── evaluate.py      <- Script for model prediction and model evaluation
│   ├── our_model.py     <- Script to define our classifier
│   ├── prepare.py       <- Script for data ingestion and data preparation
│   ├── train.py         <- Script for training setup and model training 
│   ├── trax_models.py
│   └── utils.py
├── static
│   └── nn.jpg
├── DatasetCard.md       <- Detailed information about the dataset used in this project
├── ProjectStructure.md  <- Directory structure of our project
├── README.md            <- The top-level README following a model card format
└── requirements.txt     <- Requirements file for reproducing the analysis environment



── data
│   └── raw
│       └── training.1600000.processed.noemoticon.csv.dvc
├── models
│   ├── eval
│   │   └── events.out.tfevents.1664152011.gonzalo-QC71BUBU6000
│   ├── train
│   │   └── events.out.tfevents.1664152011.gonzalo-QC71BUBU6000
│   ├── config.gin
│   ├── data_counters0.pkl
│   ├── model.opt_slots0.npy.gz
│   ├── model.pkl.gz
│   └── model.weights.npy.gz
├── src
│   ├── mlruns
│   │   └── 0
│   │       ├── 326748306457436a9fd05d1c6c983992
│   │       │   ├── metrics
│   │       │   │   ├── metrics
│   │       │   │   │   ├── Accuracy
│   │       │   │   │   └── CrossEntropyLoss
│   │       │   │   └── training
│   │       │   │       ├── gradients_l2
│   │       │   │       ├── learning_rate
│   │       │   │       ├── loss
│   │       │   │       ├── steps per second
│   │       │   │       └── weights_l2
│   │       │   ├── tags
│   │       │   │   ├── mlflow.source.git.commit
│   │       │   │   ├── mlflow.source.name
│   │       │   │   ├── mlflow.source.type
│   │       │   │   └── mlflow.user
│   │       │   └── meta.yaml
│   │       └── meta.yaml
│   ├── our_model.py
│   ├── prepare.py
│   ├── train_and_eval.py
│   ├── trax_models.py
│   └── utils.py
├── static
│   └── nn.jpg
├── DatasetCard.md
├── ProjectStructure.md
├── README.md
├── requirements.txt
└── Untitled.ipynb



```

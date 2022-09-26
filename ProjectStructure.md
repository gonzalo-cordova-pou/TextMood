The directory structure of our project: 

```
├── data                   <- DVC
│   └── raw
│       └── training.1600000.processed.noemoticon.csv.dvc
├── models                 <- Trained models
├── src                    <- Source code and MLflow
│   ├── mlruns             <- Experiment tracking
│   ├── our_model.py       <- Classifier
│   ├── prepare.py         <- Tweets and vocabulary processing
│   ├── train_and_eval.py  <- Model training and evaluation
│   ├── trax_models.py     <- Layers definition
│   └── utils.py           <- More processing and training functions
├── static                 <- Provisional static files
│   └── nn.jpg
├── DatasetCard.md         <- Dataset information
├── ProjectStructure.md    <- Directory structure of our project
├── README.md              <- The top-level README following a model card format
├── requirements.txt       <- Requirements file for reproducing the analysis environment
└── Untitled.ipynb         <- Tweets processing
```

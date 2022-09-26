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
│   ├── trax_models.py   <- 
│   └── utils.py
├── static
│   └── nn.jpg
├── DatasetCard.md       
├── ProjectStructure.md  
├── README.md            
└── requirements.txt     



├── data                  <- DVC
│   └── raw
│       └── training.1600000.processed.noemoticon.csv.dvc
├── models
├── src                   <- Soure code and MLflow
│   ├── mlruns            <- Experiment tracking
│   ├── our_model.py      <- Classifier
│   ├── prepare.py        <- Tweets and vocabulari processing
│   ├── train_and_eval.py <- Model training and evaluation
│   ├── trax_models.py    <- Layers definition
│   └── utils.py          <- Def
├── static                 <- Provisional static files
│   └── nn.jpg
├── DatasetCard.md         <- Directory structure of our project
├── ProjectStructure.md    <- Directory structure of our project
├── README.md              <- The top-level README following a model card format
├── requirements.txt       <- Requirements file for reproducing the analysis environment
└── Untitled.ipynb         <- Tweets processing



```

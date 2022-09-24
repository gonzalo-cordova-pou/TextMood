The directory structure of our project looks like this: 

```
├── data
│   └── raw
│       └── training.1600000.processed.noemoticon.csv.dvc
├── src                  <- Source code for use in this project 
│   ├── evaluate.py      <- Script for model prediction and model evaluation
│   ├── our_model.py     <- Script to define our classifier
│   ├── prepare.py
│   ├── train.py         <- Script for training setup and model training 
│   ├── trax_models.py
│   └── utils.py
├── static
│   └── nn.jpg
├── DatasetCard.md       <- Detailed information about the dataset used in this project
├── ProjectStructure.md  <- Directory structure of our project
├── README.md            <- The top-level README following a model card format
└── requirements.txt     <- Requirements file for reproducing the analysis environment

```

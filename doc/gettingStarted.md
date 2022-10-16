
# QuickStart for TextMood

- [Training and testing new models](#training-and-testing-new-models)
  - [See your results](#see-results)
    - [Comet interface](#comet-interface)
    - [CodeCarbon](#codecarbon:-co2-footprint)
  - [Share your results](#share-results)
- [Using trained models](#using-trained-models)

---

- Make sure to set up your Python3 virtual enviroment. This is not mandatory, but highly recommended. To know more about python virtual enviroments access [here](https://realpython.com/python-virtual-environments-a-primer/#why-do-you-need-virtual-environments)

- Make sure to install all the required packages using:
```
pip install -r requirements.txt
```

- Import all necessary static files with DVC
```
dvc pull
```

## TRAINING AND TESTING NEW MODELS

- Modify `training_and_eval.py` main 

Make sure to change the model name to a not existing one.

```
NAME = 'MODEL_xlarge_11'
TRAINING_BATCH_SIZE = 256
VALIDATION_BATCH_SIZE = 128
STEPS = 500
SIZE = 1600000
TRAINING_PERCENTAGE = 0.7
EMBEDDING_DIM = 256
INNER_DIM = 50
LR = 0.01
OPT = "Adam" # choices are "Adam", "SGD"
```

- Run on the command line:
```
python src/train_and_eval.py
```
or
```
python3 src/train_and_eval.py
```

### See results

#### Comet Interface

Access out Commet Project Experiments app to see results

https://www.comet.com/textmood/textmood-co2-tracking/view/new/experiments

#### CodeCarbon: CO2 Footprint

- Run un terminal:

```
carbonboard --filepath="emissions.csv" --port=5000
```
### Share results

- If you want everyone to see and use your models make sure to push your DVC files

```
dvc push
```

## USING TRAINED MODELS

Access the [Demo Jupyter Notebook](./../src/demo.ipynb) to see how you can run existing models for your own app.

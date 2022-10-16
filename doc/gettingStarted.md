
- Make sure to set up your enviroment
- Install required packages using:
```
pip install -r requirements.txt
```

- Get the dvc files
```
dvc pull
```
### Run training_and_eval.py

- Run on the command line from the repo root directory (choose a new name for the run and replace run_name):
```
dvc run -n run_name -d src/train_and_eval.py -d data/training.1600000.processed.noemoticon.csv -o ./mlruns -o ./models python src/train_and_eval.py
```

The directory structure of our project: 

```
── base
│   ├── admin.py
│   ├── api
│   │   ├── __init__.py
│   │   ├── serializers.py
│   │   ├── urls.py
│   │   └── views.py
│   ├── apps.py
│   ├── __init__.py
│   ├── migrations
│   │   ├── 0001_initial.py
│   │   └── __init__.py
│   ├── model
│   │   └── empty.txt
│   ├── model.dvc
│   ├── models.py
│   ├── templates
│   │   └── base
│   │       ├── enroll.html
│   │       ├── login_register.html
│   │       ├── main_page.html
│   │       └── navbar.html
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── data
│   └── training.1600000.processed.noemoticon.csv.dvc
├── doc                             <- More project documentation         
│   ├── DatasetCard.md
│   ├── gettingStarted.md
│   └── ProjectStructure.md
├── dvc.lock
├── dvc.yaml
├── emissions.csv.dvc
├── load_model.py
├── manage.py
├── Notebooks                       <- Notebooks (include the data analysis with GE)
│   └── GreatExpectations.ipynb
├── README.md                       <- The top-level README following a model card format
├── requirements.txt                <- Requirements for reproducing the analysis environment
├── src                             <- Source code
│   ├── demo.ipynb
│   ├── our_model.py
│   ├── prepare.py
│   ├── run_model.py
│   ├── textmood.py
│   ├── train_and_eval.py
│   ├── trax_models.py
│   └── utils.py
├── static                          <- Provisional static files
│   ├── new_nn.jpg
│   └── quantitative_analysis.png
└── TextMood_API
    ├── asgi.py
    ├── __init__.py
    ├── settings.py
    ├── urls.py
    └── wsgi.py

```

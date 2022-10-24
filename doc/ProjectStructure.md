The directory structure of our project: 

``` 
── base                             <- Django application
│   ├── admin.py
│   ├── api                         <- API files
│   │   ├── __init__.py
│   │   ├── serializers.py
│   │   ├── urls.py
│   │   └── views.py
│   ├── apps.py
│   ├── __init__.py
│   ├── migrations
│   │   ├── 0001_initial.py
│   │   └── __init__.py
│   ├── model                       <- Where the model production is stored
│   │   └── empty.txt
│   ├── model.dvc
│   ├── models.py
│   ├── templates                   <- HTML templates for the front-end
│   │   └── base
│   │       ├── enroll.html
│   │       ├── login_register.html
│   │       ├── main_page.html
│   │       └── navbar.html
│   ├── tests.py
│   ├── urls.py                     <- Sub URLs
│   └── views.py                    <- Back-end functions
├── data                            <- DVC file directory for the dataset
│   └── training.1600000.processed.noemoticon.csv.dvc
├── doc                             <- More project documentation         
│   ├── DatasetCard.md
│   ├── gettingStarted.md
│   └── ProjectStructure.md
├── dvc.lock                        <- DVC configuration
├── dvc.yaml                        <- DVC configuration
├── emissions.csv.dvc
├── load_model.py                   <- Loading model in production
├── manage.py                       <- Main python script for Django (use framework commands)
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
└── TextMood_API                    <- Django configuration 
    ├── asgi.py
    ├── __init__.py
    ├── settings.py
    ├── urls.py
    └── wsgi.py

```

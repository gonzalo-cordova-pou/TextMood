#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from os.path import exists

DIRNAME = os.path.dirname(os.path.abspath(__file__))

def main():
    """Run administrative tasks."""

    file_exists = exists(os.path.join(DIRNAME, 'base/model/checkpoint.pkl.gz'))

    if not file_exists:
        print('Loading model...')
        import load_model
        load_model.load_model('base/model')
        print('Model loaded')
    
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TextMood_API.settings')

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TextMood_API.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()

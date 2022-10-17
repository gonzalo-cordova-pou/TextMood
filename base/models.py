from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

class Classification(models.Model):
    sentence = models.TextField(null = True, blank =  True)
    classified_model = models.BooleanField(null = True)
    verified_user = models.BooleanField(null = True)
    
    def __str__(self):
        return self.sentence

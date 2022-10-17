import os
import sys
from unicodedata import category
from django.shortcuts import render, redirect
from django import forms
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.db.models import Q
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from .models import Classification

DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '../src/'))
import textmood as tm

def loginPage(request):
    page = 'login'
    if request.user.is_authenticated:
        return redirect('base:main_page')
    if request.method == 'POST':
        print(request.POST)
        print()
        username = request.POST.get('username').lower()
        password = request.POST.get('password')
        try:
            user = User.objects.get(username = username)
        except:
            messages.error(request, 'User does not exist')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('base:main_page') 
        else:
            messages.error(request, 'Username or password does not exist')
    context = {'page':page}
    return render(request, "base/login_register.html", context)

def logoutPage(request):
    logout(request)
    return redirect('base:main_page')

def saveDataP(request, pk):
    Instance = Classification.objects.get(id=pk)
    print(Instance)
    Instance.verified_user = True
    Instance.save()
    print('Correct ', pk)
    return redirect('base:main_page')

def saveDataN(request, pk):
    Instance = Classification.objects.get(id=pk)
    Instance.verified_user = False
    Instance.save()
    return redirect('base:main_page')

def indexMain(request):
    context = {'classified':False}
    if request.method == 'POST':
        sentence = request.POST.get('sentence')
        print(sentence)
        try:
            print('hola')
            my_model = tm.TextMoodModel("model", True)
            print('adios')
            my_model.initialize_model()
            print('Initialised model')
            category = my_model.predict(sentence)
        except:
            messages.error(request, 'server error')
        context['classified'] = True
        context['result'] = False
        context['sentence'] = sentence
        if category == 1:
            context['result'] = True
        instance = Classification.objects.create(
            sentence = sentence,
            classified_model= context['result']
        )
        context['id'] = instance.id
        return render(request, "base/enroll.html", context)
        
    return render(request, 'base/enroll.html', context)

def registerPage(request):
    form = UserCreationForm()
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit = False)
            user.username = user.username.lower()
            user.save()
            login(request, user)
            return redirect('base:main_page')
        else:
            messages.error(request, 'An error occured during registration')
    return render(request, "base/login_register.html", {'form':form, 'page':'register'})

@login_required(login_url = 'tasks:login')
def classifyTweet(request, pk):
    context = {}
    return render(request, 'base/main_page.html', context)

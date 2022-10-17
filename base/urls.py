from django.urls import path
from . import views

app_name = "base"
urlpatterns = [ 
    path("login", views.loginPage, name = "login"),
    path("register", views.registerPage, name = "register"),
    path("savedata/<str:pk>/", views.saveDataP, name = "savedataP"),
    path("savedata/<str:pk>/", views.saveDataN, name = "savedataN"),
    path("logout", views.logoutPage, name = "logout"),
    path("", views.indexMain, name = "main_page")
]
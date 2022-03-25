from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from .models import Post
from .form import PostForm
from . import views
from django.shortcuts import HttpResponse, render, redirect
from django.template import loader


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")
def home(request):
   # template = loader.get_template('validationForm/home.html')
    if request.method == 'POST':
        details = PostForm(request.POST)
        
        if details.is_valid():
            post = details.save(commit=False)
            
            post.save()
            
            return HttpResponse("data submit success")
        else:
            return render(request, "validationForm/home.html", {'form':details})
    else:
        form = PostForm(None)
        return render(request, 'validationForm/home.html', {'form':form})
from django.shortcuts import render
from .models import Posts
from django.views import generic
from django.views.decorators.http import require_GET
from django.http import HttpResponse

# Create your views here.
class PostList(generic.ListView):
    queryset = Posts.objects.filter(status=1).order_by('-created_on')
    template_name = 'home.html'
    paginate_by = 4

class PostDetail(generic.ListView):
    model = Posts
    template_name = "posts.html"
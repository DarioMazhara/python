from . import views
from django.urls import path, include
from .views import *
#from .feeds import blogFeed

urlpatterns = [
    path('', views.PostList.as_view(), name = 'posts'),
    # route for posts
    path('<slug:slug>/', views.PostDetail.as_view(), name = 'post_detail'),
]
from django.urls import path
from . import views


urlpatterns = [
    path('', views.HomePage.as_view(), name='home'),
    path('form/', views.CreateReview.as_view(), name='form'),
    path('statistic/<slug:post_slug>/', views.Statistic.as_view(), name='statistic'),
    path('search/', views.Search.as_view(), name='search')
]

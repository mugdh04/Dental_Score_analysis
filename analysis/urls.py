from django.urls import path
from . import views

app_name = 'analysis'

urlpatterns = [
    path('', views.upload_view, name='upload'),
    path('processing/<uuid:pk>/', views.processing_view, name='processing'),
    path('results/<uuid:pk>/', views.results_view, name='results'),
    path('api/status/<uuid:pk>/', views.check_status, name='check_status'),
    path('lookup/', views.lookup_view, name='lookup'),
]

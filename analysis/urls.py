from django.urls import path
from django.contrib.auth import views as auth_views

from . import views
from .forms import PortalAuthenticationForm

app_name = 'analysis'

urlpatterns = [
    path('', views.home_view, name='home'),
    path('login/', auth_views.LoginView.as_view(template_name='analysis/login.html', authentication_form=PortalAuthenticationForm), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='analysis:login'), name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('profile/', views.profile_view, name='profile'),
    path('dashboard/admin/', views.admin_dashboard_view, name='admin_dashboard'),
    path('dashboard/admin/create-user/', views.admin_create_user_view, name='admin_create_user'),
    path('dashboard/admin/delete-user/<int:user_id>/', views.admin_delete_user_view, name='admin_delete_user'),
    path('dashboard/admin/set-password/<int:user_id>/', views.admin_set_user_password_view, name='admin_set_user_password'),
    path('dashboard/admin/reset-password/<int:user_id>/', views.admin_reset_user_password_view, name='admin_reset_user_password'),
    path('dashboard/dentist/', views.dentist_dashboard_view, name='dentist_dashboard'),
    path('dashboard/patient/', views.patient_dashboard_view, name='patient_dashboard'),
    path('dashboard/dentist/create-patient/', views.create_patient_account_view, name='create_patient_account'),
    path('upload/', views.upload_view, name='upload'),
    path('processing/<uuid:pk>/', views.processing_view, name='processing'),
    path('results/<uuid:pk>/', views.results_view, name='results'),
    path('review/<uuid:pk>/', views.review_report_view, name='review_report'),
    path('download/<uuid:pk>/', views.download_report_pdf_view, name='download_report_pdf'),
    path('api/status/<uuid:pk>/', views.check_status, name='check_status'),
    path('lookup/', views.lookup_view, name='lookup'),
]

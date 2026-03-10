from django.contrib import admin
from .models import PatientAnalysis


@admin.register(PatientAnalysis)
class PatientAnalysisAdmin(admin.ModelAdmin):
    list_display = ['patient_name', 'unique_code', 'status', 'mgi_score', 'ohi_score', 'gei_score', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['patient_name', 'unique_code']
    readonly_fields = ['id', 'unique_code', 'created_at', 'updated_at']

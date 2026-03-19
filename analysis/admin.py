from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import DentalUser, PatientAnalysis, ReportRevision


@admin.register(DentalUser)
class DentalUserAdmin(UserAdmin):
    list_display = ['username', 'display_name', 'phone_number', 'role', 'dentist_owner', 'is_active', 'is_staff']
    list_filter = ['role', 'is_active', 'is_staff']
    search_fields = ['username', 'first_name', 'last_name', 'email', 'phone_number']
    fieldsets = UserAdmin.fieldsets + (
        ('Role & Ownership', {'fields': ('role', 'phone_number', 'dentist_owner')}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Role & Ownership', {'fields': ('role', 'phone_number', 'dentist_owner')}),
    )


@admin.register(ReportRevision)
class ReportRevisionAdmin(admin.ModelAdmin):
    list_display = ['analysis', 'edited_by', 'old_mgi_score', 'new_mgi_score', 'created_at']
    list_filter = ['created_at']
    search_fields = ['analysis__unique_code', 'analysis__patient_name', 'edited_by__username']
    readonly_fields = [
        'analysis',
        'edited_by',
        'old_mgi_score',
        'old_ohi_score',
        'old_gei_score',
        'new_mgi_score',
        'new_ohi_score',
        'new_gei_score',
        'reason',
        'created_at',
    ]


@admin.register(PatientAnalysis)
class PatientAnalysisAdmin(admin.ModelAdmin):
    list_display = [
        'patient_name',
        'unique_code',
        'status',
        'review_status',
        'dentist_owner',
        'patient_user',
        'mgi_score',
        'ohi_score',
        'gei_score',
        'created_at',
    ]
    list_filter = ['status', 'review_status', 'created_at', 'dentist_owner']
    search_fields = ['patient_name', 'unique_code', 'patient_user__username', 'dentist_owner__username']
    readonly_fields = ['id', 'unique_code', 'created_at', 'updated_at', 'reviewed_at']

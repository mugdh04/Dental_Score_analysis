import random
import string
import uuid

from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models


def generate_unique_code():
    """Generate a 6-digit unique patient code."""
    return ''.join(random.choices(string.digits, k=6))


class DentalUser(AbstractUser):
    """Custom user model with role support for admin, dentist, and patient."""

    ROLE_ADMIN = 'admin'
    ROLE_DENTIST = 'dentist'
    ROLE_PATIENT = 'patient'
    ROLE_CHOICES = [
        (ROLE_ADMIN, 'Admin'),
        (ROLE_DENTIST, 'Dentist'),
        (ROLE_PATIENT, 'Patient'),
    ]

    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_PATIENT, db_index=True)
    phone_number = models.CharField(max_length=20, unique=True, null=True, blank=True, db_index=True)
    dentist_owner = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='patients',
        limit_choices_to={'role': ROLE_DENTIST},
        help_text='For patient accounts, identifies the responsible dentist.',
    )

    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'

    def __str__(self):
        return f"{self.username} ({self.role})"

    @property
    def display_name(self):
        full_name = self.get_full_name().strip()
        return full_name or self.username

    @property
    def is_role_admin(self):
        return self.role == self.ROLE_ADMIN or self.is_superuser

    @property
    def is_role_dentist(self):
        return self.role == self.ROLE_DENTIST

    @property
    def is_role_patient(self):
        return self.role == self.ROLE_PATIENT


class PatientAnalysis(models.Model):
    """Stores each patient upload and predicted dental index scores."""

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    REVIEW_UNREVIEWED = 'unreviewed'
    REVIEW_APPROVED = 'approved'
    REVIEW_REJECTED = 'rejected'
    REVIEW_CHOICES = [
        (REVIEW_UNREVIEWED, 'Unreviewed Reports'),
        (REVIEW_APPROVED, 'Approved Reports'),
        (REVIEW_REJECTED, 'Rejected Reports'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient_name = models.CharField(max_length=200)
    unique_code = models.CharField(max_length=6, unique=True, default=generate_unique_code)

    patient_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='patient_reports',
        help_text='Patient account that owns this report.',
    )
    dentist_owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='dentist_reports',
        limit_choices_to={'role': DentalUser.ROLE_DENTIST},
        help_text='Dentist responsible for reviewing this report.',
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='created_reports',
        help_text='User who triggered the upload.',
    )

    # Uploaded images
    frontal_image = models.ImageField(upload_to='uploads/frontal/')
    left_lateral_image = models.ImageField(upload_to='uploads/left_lateral/')
    right_lateral_image = models.ImageField(upload_to='uploads/right_lateral/')

    # Processing status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    # Predicted scores (nullable until processing completes)
    mgi_score = models.IntegerField(null=True, blank=True, help_text='Modified Gingival Index (0-4)')
    ohi_score = models.IntegerField(null=True, blank=True, help_text='Oral Hygiene Index (0-3)')
    gei_score = models.IntegerField(null=True, blank=True, help_text='Gingival Enlargement Index (0-2)')

    # Immutable AI baseline scores for audit purposes
    ai_mgi_score = models.IntegerField(null=True, blank=True)
    ai_ohi_score = models.IntegerField(null=True, blank=True)
    ai_gei_score = models.IntegerField(null=True, blank=True)

    # Confidence scores
    mgi_confidence = models.FloatField(null=True, blank=True)
    ohi_confidence = models.FloatField(null=True, blank=True)
    gei_confidence = models.FloatField(null=True, blank=True)

    # Immutable AI baseline confidence for audit purposes
    ai_mgi_confidence = models.FloatField(null=True, blank=True)
    ai_ohi_confidence = models.FloatField(null=True, blank=True)
    ai_gei_confidence = models.FloatField(null=True, blank=True)

    review_status = models.CharField(
        max_length=20,
        choices=REVIEW_CHOICES,
        default=REVIEW_UNREVIEWED,
        db_index=True,
    )
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='reviewed_reports',
        limit_choices_to={'role__in': [DentalUser.ROLE_DENTIST, DentalUser.ROLE_ADMIN]},
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)
    dentist_note = models.TextField(blank=True, default='')
    clinical_recommendations = models.JSONField(blank=True, default=list)

    # Additional plaque-related metrics (migration 0004)
    plaque_score = models.IntegerField(
        null=True,
        blank=True,
        help_text='Plaque Level (Turesky-modified Quigley-Hein, 0-5)',
    )
    plaque_confidence = models.FloatField(null=True, blank=True)
    ai_plaque_score = models.IntegerField(null=True, blank=True)
    ai_plaque_confidence = models.FloatField(null=True, blank=True)

    # Grad-CAM heatmap images (generated during inference)
    gradcam_frontal = models.ImageField(upload_to='gradcam/', null=True, blank=True)
    gradcam_left = models.ImageField(upload_to='gradcam/', null=True, blank=True)
    gradcam_right = models.ImageField(upload_to='gradcam/', null=True, blank=True)

    # Error message if processing fails
    error_message = models.TextField(blank=True, default='')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Patient Analyses'
        indexes = [
            models.Index(fields=['dentist_owner', 'review_status', 'created_at']),
            models.Index(fields=['patient_user', 'created_at']),
        ]

    def __str__(self):
        return f"{self.patient_name} ({self.unique_code})"

    @property
    def has_dentist_revision(self):
        return self.revisions.exists()

    @property
    def status_label(self):
        if self.review_status == self.REVIEW_UNREVIEWED:
            return 'Yet to be approved by the dentist'
        if self.review_status == self.REVIEW_APPROVED:
            return f"Approved by Dr. {self.reviewed_by.display_name}" if self.reviewed_by else 'Approved by dentist'
        if self.review_status == self.REVIEW_REJECTED and self.reviewed_by:
            return f"Report by Dr. {self.reviewed_by.display_name}"
        return 'Pending review'

    def save(self, *args, **kwargs):
        # Ensure unique_code is unique
        while not self.unique_code or PatientAnalysis.objects.filter(
            unique_code=self.unique_code
        ).exclude(pk=self.pk).exists():
            self.unique_code = generate_unique_code()

        if self.clinical_recommendations is None:
            self.clinical_recommendations = []

        super().save(*args, **kwargs)


class ReportRevision(models.Model):
    """Immutable revision history whenever a dentist changes AI-generated scores."""

    analysis = models.ForeignKey(PatientAnalysis, on_delete=models.CASCADE, related_name='revisions')
    edited_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='report_revisions',
    )
    old_mgi_score = models.IntegerField(null=True, blank=True)
    old_ohi_score = models.IntegerField(null=True, blank=True)
    old_gei_score = models.IntegerField(null=True, blank=True)
    old_plaque_score = models.IntegerField(null=True, blank=True)
    new_mgi_score = models.IntegerField(null=True, blank=True)
    new_ohi_score = models.IntegerField(null=True, blank=True)
    new_gei_score = models.IntegerField(null=True, blank=True)
    new_plaque_score = models.IntegerField(null=True, blank=True)
    reason = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Revision for {self.analysis.unique_code} at {self.created_at:%Y-%m-%d %H:%M}"

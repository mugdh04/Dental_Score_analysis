import uuid
import random
import string
from django.db import models


def generate_unique_code():
    """Generate a 6-digit unique patient code."""
    return ''.join(random.choices(string.digits, k=6))


class PatientAnalysis(models.Model):
    """Stores each patient upload and predicted dental index scores."""

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient_name = models.CharField(max_length=200)
    unique_code = models.CharField(max_length=6, unique=True, default=generate_unique_code)

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

    # Confidence scores
    mgi_confidence = models.FloatField(null=True, blank=True)
    ohi_confidence = models.FloatField(null=True, blank=True)
    gei_confidence = models.FloatField(null=True, blank=True)

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

    def __str__(self):
        return f"{self.patient_name} ({self.unique_code})"

    def save(self, *args, **kwargs):
        # Ensure unique_code is unique
        while not self.unique_code or PatientAnalysis.objects.filter(
            unique_code=self.unique_code
        ).exclude(pk=self.pk).exists():
            self.unique_code = generate_unique_code()
        super().save(*args, **kwargs)

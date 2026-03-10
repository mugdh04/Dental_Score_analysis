from django import forms
from .models import PatientAnalysis


class PatientUploadForm(forms.ModelForm):
    """Form for uploading patient dental photographs."""

    class Meta:
        model = PatientAnalysis
        fields = ['patient_name', 'frontal_image', 'left_lateral_image', 'right_lateral_image']
        widgets = {
            'patient_name': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 rounded-xl border-2 border-gray-200 '
                         'focus:border-teal-500 focus:ring-2 focus:ring-teal-200 '
                         'transition-all duration-300 text-gray-700 bg-white',
                'placeholder': 'Enter patient full name',
                'autocomplete': 'off',
            }),
            'frontal_image': forms.FileInput(attrs={
                'class': 'hidden',
                'accept': 'image/*',
                'id': 'frontal-upload',
            }),
            'left_lateral_image': forms.FileInput(attrs={
                'class': 'hidden',
                'accept': 'image/*',
                'id': 'left-upload',
            }),
            'right_lateral_image': forms.FileInput(attrs={
                'class': 'hidden',
                'accept': 'image/*',
                'id': 'right-upload',
            }),
        }
        labels = {
            'patient_name': 'Patient Name',
            'frontal_image': 'Frontal View',
            'left_lateral_image': 'Left Lateral View',
            'right_lateral_image': 'Right Lateral View',
        }

    def clean_patient_name(self):
        name = self.cleaned_data.get('patient_name', '').strip()
        if len(name) < 2:
            raise forms.ValidationError('Patient name must be at least 2 characters.')
        return name

from django import forms
from django.contrib.auth.forms import AuthenticationForm

from .models import DentalUser, PatientAnalysis


class PortalAuthenticationForm(AuthenticationForm):
    """Styled login form for the authentication landing page."""

    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none',
                'placeholder': 'Username or phone number',
                'autocomplete': 'username',
            }
        )
    )
    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none',
                'placeholder': 'Password',
                'autocomplete': 'current-password',
            }
        )
    )

    def clean(self):
        identifier = self.cleaned_data.get('username', '').strip()
        digits = ''.join(ch for ch in identifier if ch.isdigit())
        if digits:
            try:
                user = DentalUser.objects.get(phone_number=digits)
                self.cleaned_data['username'] = user.username
            except DentalUser.DoesNotExist:
                pass
        return super().clean()


class PatientUploadForm(forms.ModelForm):
    """Form for uploading patient dental photographs."""

    patient_user = forms.ModelChoiceField(
        queryset=DentalUser.objects.none(),
        required=False,
        help_text='Only dentists/admins can upload for linked patients.',
        widget=forms.Select(
            attrs={
                'class': 'w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-teal-500 focus:ring-2 focus:ring-teal-200 transition-all duration-300 text-gray-700 bg-white',
            }
        ),
    )

    class Meta:
        model = PatientAnalysis
        fields = ['patient_name', 'patient_user', 'frontal_image', 'left_lateral_image', 'right_lateral_image']
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
            'patient_user': 'Patient Account',
            'frontal_image': 'Frontal View',
            'left_lateral_image': 'Left Lateral View',
            'right_lateral_image': 'Right Lateral View',
        }

    def __init__(self, *args, **kwargs):
        actor = kwargs.pop('actor', None)
        super().__init__(*args, **kwargs)

        # Hide patient selector by default (patient self-upload flow).
        self.fields['patient_user'].widget = forms.HiddenInput()

        if actor and (actor.is_role_dentist or actor.is_role_admin):
            self.fields['patient_user'].widget = forms.Select(
                attrs={
                    'class': 'w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-teal-500 focus:ring-2 focus:ring-teal-200 transition-all duration-300 text-gray-700 bg-white',
                }
            )
            if actor.is_role_dentist:
                self.fields['patient_user'].queryset = DentalUser.objects.filter(role=DentalUser.ROLE_PATIENT, dentist_owner=actor).order_by('username')
            else:
                self.fields['patient_user'].queryset = DentalUser.objects.filter(role=DentalUser.ROLE_PATIENT).order_by('username')

    def clean_patient_name(self):
        name = self.cleaned_data.get('patient_name', '').strip()
        if len(name) < 2:
            raise forms.ValidationError('Patient name must be at least 2 characters.')
        return name


class DentistCreatePatientForm(forms.Form):
    """Dentist form to generate a patient account with system credentials."""

    patient_name = forms.CharField(
        max_length=200,
        widget=forms.TextInput(
            attrs={
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none',
                'placeholder': 'Enter patient full name',
            }
        ),
    )
    phone_number = forms.CharField(
        max_length=20,
        widget=forms.TextInput(
            attrs={
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none',
                'placeholder': 'Enter patient phone number',
            }
        ),
    )

    def clean_patient_name(self):
        name = self.cleaned_data['patient_name'].strip()
        if len(name) < 2:
            raise forms.ValidationError('Patient name must be at least 2 characters.')
        return name

    def clean_phone_number(self):
        raw = self.cleaned_data['phone_number']
        normalized = ''.join(ch for ch in raw if ch.isdigit())
        if len(normalized) < 8:
            raise forms.ValidationError('Phone number must contain at least 8 digits.')
        return normalized


class ReviewReportForm(forms.Form):
    """Dentist review form to approve AI scores or edit and reject AI output."""

    ACTION_APPROVE = 'approve'
    ACTION_EDIT = 'edit'
    ACTION_CHOICES = [
        (ACTION_APPROVE, 'Approve AI report'),
        (ACTION_EDIT, 'Edit report (reject AI values)'),
    ]

    action = forms.ChoiceField(
        choices=ACTION_CHOICES,
        widget=forms.RadioSelect,
        initial=ACTION_APPROVE,
    )
    mgi_score = forms.IntegerField(min_value=0, max_value=4, required=False)
    ohi_score = forms.IntegerField(min_value=0, max_value=3, required=False)
    gei_score = forms.IntegerField(min_value=0, max_value=2, required=False)
    reason = forms.CharField(required=False, widget=forms.Textarea(attrs={'rows': 3}))

    def __init__(self, *args, **kwargs):
        patient = kwargs.pop('patient', None)
        super().__init__(*args, **kwargs)
        if patient:
            self.fields['mgi_score'].initial = patient.mgi_score
            self.fields['ohi_score'].initial = patient.ohi_score
            self.fields['gei_score'].initial = patient.gei_score

    def clean(self):
        cleaned = super().clean()
        action = cleaned.get('action')

        if action == self.ACTION_EDIT:
            for field in ('mgi_score', 'ohi_score', 'gei_score'):
                if cleaned.get(field) is None:
                    self.add_error(field, 'This value is required when editing a report.')
            if not cleaned.get('reason', '').strip():
                self.add_error('reason', 'Please provide a reason for editing the AI report.')
        return cleaned


class AdminCreateUserForm(forms.Form):
    """Admin-only form to create platform users of any role."""

    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(
            attrs={
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none',
                'placeholder': 'Username',
                'autocomplete': 'off',
            }
        ),
    )
    first_name = forms.CharField(
        max_length=150,
        required=False,
        widget=forms.TextInput(
            attrs={
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none',
                'placeholder': 'First name (optional)',
            }
        ),
    )
    last_name = forms.CharField(
        max_length=150,
        required=False,
        widget=forms.TextInput(
            attrs={
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none',
                'placeholder': 'Last name (optional)',
            }
        ),
    )
    role = forms.ChoiceField(
        choices=DentalUser.ROLE_CHOICES,
        widget=forms.Select(
            attrs={
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none bg-white',
            }
        ),
    )
    phone_number = forms.CharField(
        max_length=20,
        required=False,
        widget=forms.TextInput(
            attrs={
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none',
                'placeholder': 'Phone number (optional)',
            }
        ),
    )
    dentist_owner = forms.ModelChoiceField(
        queryset=DentalUser.objects.filter(role=DentalUser.ROLE_DENTIST).order_by('username'),
        required=False,
        widget=forms.Select(
            attrs={
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none bg-white',
            }
        ),
        help_text='For patient users, optionally assign a dentist owner.',
    )
    password = forms.CharField(
        max_length=128,
        widget=forms.PasswordInput(
            attrs={
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none',
                'placeholder': 'Set initial password',
                'autocomplete': 'new-password',
            }
        ),
    )

    def clean_username(self):
        username = self.cleaned_data['username'].strip()
        if DentalUser.objects.filter(username=username).exists():
            raise forms.ValidationError('This username is already taken.')
        return username

    def clean_phone_number(self):
        raw = self.cleaned_data.get('phone_number', '')
        normalized = ''.join(ch for ch in raw if ch.isdigit())
        if not normalized:
            return ''
        if len(normalized) < 8:
            raise forms.ValidationError('Phone number must contain at least 8 digits.')
        if DentalUser.objects.filter(phone_number=normalized).exists():
            raise forms.ValidationError('This phone number is already registered.')
        return normalized


class AdminSetUserPasswordForm(forms.Form):
    """Admin-only form for changing an existing user's password."""

    new_password = forms.CharField(
        max_length=128,
        widget=forms.PasswordInput(
            attrs={
                'class': 'w-full rounded-xl border border-gray-300 px-3 py-2 focus:border-dental-600 focus:outline-none',
                'placeholder': 'New password',
                'autocomplete': 'new-password',
            }
        ),
    )

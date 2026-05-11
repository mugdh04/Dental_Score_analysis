from django import forms
from django.contrib.auth.forms import AuthenticationForm
from django.utils import timezone

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
        if len(normalized) < 10 or len(normalized) > 10:
            raise forms.ValidationError('Phone number must contain 10 digits.')
        return normalized


class DentistSuggestionForm(forms.Form):
    """Form for dentists to send suggestions/guidance to linked patients."""

    patient = forms.ModelChoiceField(
        queryset=DentalUser.objects.none(),
        widget=forms.Select(
            attrs={
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none bg-white',
            }
        ),
    )
    message = forms.CharField(
        max_length=1200,
        widget=forms.Textarea(
            attrs={
                'rows': 4,
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none',
                'placeholder': 'Write suggestions, reminders, or next-step instructions for the patient.',
            }
        ),
    )

    def __init__(self, *args, **kwargs):
        dentist = kwargs.pop('dentist', None)
        super().__init__(*args, **kwargs)

        qs = DentalUser.objects.filter(role=DentalUser.ROLE_PATIENT).order_by('first_name', 'username')
        if dentist is not None:
            qs = qs.filter(dentist_owner=dentist)
        self.fields['patient'].queryset = qs

    def clean_message(self):
        message = self.cleaned_data['message'].strip()
        if len(message) < 5:
            raise forms.ValidationError('Suggestion message must be at least 5 characters.')
        return message


class AppointmentRequestForm(forms.Form):
    """Form for patients to request an appointment from their linked dentist."""

    request_note = forms.CharField(
        required=False,
        max_length=600,
        widget=forms.Textarea(
            attrs={
                'rows': 3,
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none',
                'placeholder': 'Optional note for the dentist (preferred day, concern, urgency).',
            }
        ),
    )

    def clean_request_note(self):
        return self.cleaned_data.get('request_note', '').strip()


class AppointmentSlotsForm(forms.Form):
    """Form for dentists to propose three appointment slots."""

    slot_option_1 = forms.DateTimeField(
        input_formats=['%Y-%m-%dT%H:%M'],
        widget=forms.DateTimeInput(
            attrs={
                'type': 'datetime-local',
                'class': 'w-full rounded-xl border border-gray-300 px-3 py-2 focus:border-dental-600 focus:outline-none bg-white',
            }
        ),
    )
    slot_option_2 = forms.DateTimeField(
        input_formats=['%Y-%m-%dT%H:%M'],
        widget=forms.DateTimeInput(
            attrs={
                'type': 'datetime-local',
                'class': 'w-full rounded-xl border border-gray-300 px-3 py-2 focus:border-dental-600 focus:outline-none bg-white',
            }
        ),
    )
    slot_option_3 = forms.DateTimeField(
        input_formats=['%Y-%m-%dT%H:%M'],
        widget=forms.DateTimeInput(
            attrs={
                'type': 'datetime-local',
                'class': 'w-full rounded-xl border border-gray-300 px-3 py-2 focus:border-dental-600 focus:outline-none bg-white',
            }
        ),
    )

    def clean(self):
        cleaned = super().clean()
        slots = [
            cleaned.get('slot_option_1'),
            cleaned.get('slot_option_2'),
            cleaned.get('slot_option_3'),
        ]
        if any(slot is None for slot in slots):
            return cleaned

        normalized = []
        for slot in slots:
            if timezone.is_naive(slot):
                slot = timezone.make_aware(slot, timezone.get_current_timezone())
            normalized.append(slot)

        now = timezone.now()
        if any(slot <= now for slot in normalized):
            raise forms.ValidationError('All proposed slots must be in the future.')

        if len(set(normalized)) != len(normalized):
            raise forms.ValidationError('Please provide three different slot times.')

        cleaned['slot_option_1'] = normalized[0]
        cleaned['slot_option_2'] = normalized[1]
        cleaned['slot_option_3'] = normalized[2]
        return cleaned


class AppointmentSlotSelectionForm(forms.Form):
    """Form for patients to select one dentist-proposed appointment slot."""

    selected_slot_key = forms.ChoiceField(widget=forms.RadioSelect)

    SLOT_KEYS = ('slot_option_1', 'slot_option_2', 'slot_option_3')

    def __init__(self, *args, **kwargs):
        appointment = kwargs.pop('appointment', None)
        super().__init__(*args, **kwargs)

        choices = []
        if appointment is not None:
            for key in self.SLOT_KEYS:
                slot_value = getattr(appointment, key)
                if slot_value is not None:
                    choices.append((key, timezone.localtime(slot_value).strftime('%A, %d %b %Y %I:%M %p')))
        self.fields['selected_slot_key'].choices = choices

    def clean_selected_slot_key(self):
        slot_key = self.cleaned_data['selected_slot_key']
        if slot_key not in self.SLOT_KEYS:
            raise forms.ValidationError('Invalid slot selection.')
        return slot_key


class ReviewReportForm(forms.Form):
    """Dentist review form to approve AI scores or edit and reject AI output."""

    ACTION_APPROVE = 'approve'
    ACTION_EDIT = 'edit'
    ACTION_CHOICES = [
        (ACTION_APPROVE, 'Approve AI report'),
        (ACTION_EDIT, 'Edit report'),
    ]

    action = forms.ChoiceField(
        choices=ACTION_CHOICES,
        widget=forms.Select(attrs={
            'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none bg-white',
            'id': 'review_action_select'
        }),
        initial=ACTION_APPROVE,
    )
    mgi_score = forms.IntegerField(
        min_value=0, max_value=4, required=False,
        widget=forms.NumberInput(attrs={
            'class': 'w-full text-center border-y border-gray-300 py-3 focus:outline-none focus:ring-0',
            'id': 'id_mgi_score'
        })
    )
    ohi_score = forms.IntegerField(
        min_value=0, max_value=3, required=False,
        widget=forms.NumberInput(attrs={
            'class': 'w-full text-center border-y border-gray-300 py-3 focus:outline-none focus:ring-0',
            'id': 'id_ohi_score'
        })
    )
    gei_score = forms.IntegerField(
        min_value=0, max_value=3, required=False,
        widget=forms.NumberInput(attrs={
            'class': 'w-full text-center border-y border-gray-300 py-3 focus:outline-none focus:ring-0',
            'id': 'id_gei_score'
        })
    )
    plaque_score = forms.IntegerField(
        min_value=0, max_value=5, required=False,
        widget=forms.NumberInput(attrs={
            'class': 'w-full text-center border-y border-gray-300 py-3 focus:outline-none focus:ring-0',
            'id': 'id_plaque_score'
        })
    )
    reason = forms.CharField(required=False, widget=forms.Textarea(attrs={
        'rows': 3,
        'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none bg-white placeholder-gray-400',
        'placeholder': 'Add clinical reasoning/notes if editing scores...'
    }))

    def __init__(self, *args, **kwargs):
        patient = kwargs.pop('patient', None)
        super().__init__(*args, **kwargs)
        if patient:
            self.fields['mgi_score'].initial = patient.mgi_score
            self.fields['ohi_score'].initial = patient.ohi_score
            self.fields['gei_score'].initial = patient.gei_score
            self.fields['plaque_score'].initial = (
                patient.plaque_score
                if patient.plaque_score is not None
                else patient.ai_plaque_score
            )

    def clean(self):
        cleaned = super().clean()
        action = cleaned.get('action')

        if action == self.ACTION_EDIT:
            for field in ('mgi_score', 'ohi_score', 'gei_score', 'plaque_score'):
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
    full_name = forms.CharField(
        max_length=300,
        required=False,
        widget=forms.TextInput(
            attrs={
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none',
                'placeholder': 'Full name (optional)',
            }
        ),
    )
    role = forms.ChoiceField(
        choices=[('', 'Select Role')] + DentalUser.ROLE_CHOICES,
        widget=forms.Select(
            attrs={
                'class': 'w-full rounded-xl border border-gray-300 px-4 py-3 focus:border-dental-600 focus:outline-none bg-white',
                'id': 'role_select',
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
                'id': 'dentist_owner_select',
            }
        ),
        help_text='Required for patient users. Assign a dentist owner.',
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

    def clean(self):
        cleaned_data = super().clean()
        role = cleaned_data.get('role')
        dentist_owner = cleaned_data.get('dentist_owner')
        if role == DentalUser.ROLE_PATIENT and not dentist_owner:
            self.add_error('dentist_owner', 'Dentist owner is required for patient users.')
        return cleaned_data

    def clean_username(self):
        username = self.cleaned_data.get('username', '').strip()
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


class UserProfileForm(forms.ModelForm):
    full_name = forms.CharField(max_length=300, required=False)

    class Meta:
        model = DentalUser
        fields = ['full_name', 'email', 'phone_number']
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance:
            self.fields['full_name'].initial = f"{self.instance.first_name} {self.instance.last_name}".strip()
        for field in self.fields.values():
            field.widget.attrs.update({'class': 'w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-dental-500 focus:border-dental-500 transition-all outline-none bg-gray-50'})

    def save(self, commit=True):
        user = super().save(commit=False)
        full_name = self.cleaned_data.get('full_name', '').strip()
        parts = full_name.split(' ', 1)
        if len(parts) == 2:
            user.first_name = parts[0]
            user.last_name = parts[1]
        else:
            user.first_name = full_name
            user.last_name = ''
        if commit:
            user.save()
        return user

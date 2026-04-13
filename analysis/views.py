import os
import uuid
import json
import threading
import random
import string
import logging
from pathlib import Path
from urllib.parse import urlencode

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db import models as db_models
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.conf import settings
from django.utils import timezone
from django.views.decorators.cache import never_cache
from PIL import Image
import numpy as np

from .forms import (
    AdminCreateUserForm,
    AdminSetUserPasswordForm,
    DentistCreatePatientForm,
    PatientUploadForm,
    ReviewReportForm,
)
from .models import DentalUser, PatientAnalysis, ReportRevision

# -----------------------------------------------------------------------------
# Change Note (2026-04-03)
# Integrated plaque-index persistence/display, added PI computation in the
# background analysis path, and introduced startup-safe logging hooks while
# preserving existing upload/processing/results workflow.
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)
_PI_DETECTOR = None
_PI_SEGMENTOR = None


# Dental hygiene quotes for the loading screen
DENTAL_QUOTES = [
    "A smile is the prettiest thing you can wear.",
    "Your teeth are like stars — they come out at night... just kidding! Keep them healthy!",
    "Brush your teeth twice a day, floss daily, and visit your dentist regularly.",
    "Good oral hygiene is the foundation of a beautiful smile.",
    "Healthy gums, healthy life — periodontal health matters.",
    "Prevention is always better than cure — especially in dentistry.",
    "Your mouth is the gateway to your overall health.",
    "Strong teeth come from good habits and regular dental care.",
    "A clean mouth speaks volumes about your health.",
    "Dental health is not a luxury — it's a necessity.",
    "Behind every great smile is healthy gingiva.",
    "Plaque is the enemy — brush it away every day.",
    "Your dental health today shapes your smile tomorrow.",
    "Invest in your smile — it's the first thing people notice.",
    "Oral hygiene indices help us measure what matters most.",
]


def _generate_username_from_name(name):
    base = ''.join(ch.lower() for ch in name if ch.isalnum())[:10] or 'patient'
    return f"{base[:6]}{random.randint(1000, 9999)}"


def _normalize_phone(phone_number):
    return ''.join(ch for ch in str(phone_number or '') if ch.isdigit())


def _generate_password(length=10):
    alphabet = string.ascii_letters + string.digits
    return ''.join(random.choice(alphabet) for _ in range(length))


def _can_access_report(user, report):
    if not user.is_authenticated:
        return False
    if user.is_role_admin:
        return True
    if user.is_role_dentist:
        return report.dentist_owner_id == user.id or report.created_by_id == user.id
    return report.patient_user_id == user.id


def _can_review_report(user, report):
    if not user.is_authenticated:
        return False
    if user.is_role_admin:
        return True
    return user.is_role_dentist and report.dentist_owner_id == user.id


def _get_status_banner(patient, viewer=None):
    viewer_is_patient = bool(viewer and getattr(viewer, 'is_role_patient', False))

    if patient.review_status == PatientAnalysis.REVIEW_UNREVIEWED:
        return {
            'bg': 'bg-orange-50 border-orange-700 text-orange-700',
            'text': 'Yet to be approved by the dentist',
        }
    if patient.review_status == PatientAnalysis.REVIEW_APPROVED:
        dentist_name = patient.reviewed_by.display_name if patient.reviewed_by else 'Dentist'
        return {
            'bg': 'bg-green-50 border-green-700 text-green-700',
            'text': f'Approved by Dr. {dentist_name}',
        }

    dentist_name = patient.reviewed_by.display_name if patient.reviewed_by else 'Dentist'
    rejected_text = f'Reviewed by Dr. {dentist_name}' if viewer_is_patient else f'Report by Dr. {dentist_name}'
    return {
        'bg': 'bg-green-50 border-green-700 text-green-700',
        'text': rejected_text,
    }


def _plaque_badge_style(score):
    """Return Tailwind badge classes for plaque score severity."""
    mapping = {
        0: 'bg-green-50 text-green-700 border-green-200',
        1: 'bg-green-50 text-green-700 border-green-200',
        2: 'bg-yellow-50 text-yellow-700 border-yellow-200',
        3: 'bg-orange-50 text-orange-700 border-orange-200',
        4: 'bg-orange-50 text-orange-700 border-orange-200',
        5: 'bg-red-50 text-red-700 border-red-200',
    }
    return mapping.get(score, 'bg-gray-50 text-gray-700 border-gray-200')


_PREDICTOR = None
_LAST_INFERENCE_LATENCY = 0.0

def get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        try:
            from inference.predict import OralHealthPredictor
            ensemble_path = os.environ.get('ENSEMBLE_CONFIG_PATH', os.path.join(settings.BASE_DIR, 'models', 'ensemble_config.json'))
            pi_calibration_path = os.environ.get('PI_CALIBRATION_PATH', os.path.join(settings.BASE_DIR, 'models', 'pi_calibration.json'))
            device = os.environ.get('DEVICE', 'cpu')
            _PREDICTOR = OralHealthPredictor(ensemble_path, pi_calibration_path, device)
        except Exception as e:
            logger.error(f"Failed to load OralHealthPredictor: {e}")
    return _PREDICTOR

def health_check_view(request):
    try:
        pred = get_predictor()
        status = "loaded" if pred else "failed"
    except Exception as e:
        status = f"error: {str(e)}"
    
    return JsonResponse({
        "model_status": status,
        "last_inference_latency_seconds": _LAST_INFERENCE_LATENCY
    })



def _dashboard_redirect_for_user(user):
    if user.is_role_admin:
        return 'analysis:admin_dashboard'
    if user.is_role_dentist:
        return 'analysis:dentist_dashboard'
    return 'analysis:patient_dashboard'


def home_view(request):
    if not request.user.is_authenticated:
        return redirect('analysis:login')
    return redirect(_dashboard_redirect_for_user(request.user))


@login_required
def upload_view(request):
    """Main upload page where users submit dental photographs."""
    if not (request.user.is_role_admin or request.user.is_role_dentist or request.user.is_role_patient):
        messages.error(request, 'Your account role is not allowed to upload reports.')
        return redirect('analysis:login')

    if request.method == 'POST':
        post_data = request.POST.copy()
        if request.user.is_role_patient:
            post_data['patient_name'] = request.user.display_name
        form = PatientUploadForm(post_data, request.FILES, actor=request.user)
        if form.is_valid():
            if 'frontal_image' not in request.FILES or 'left_lateral_image' not in request.FILES or 'right_lateral_image' not in request.FILES:
                from django.http import HttpResponseBadRequest
                return HttpResponseBadRequest("Exactly 3 images (frontal, left, right) must be provided.")
                
            patient = form.save(commit=False)
            selected_patient = form.cleaned_data.get('patient_user')

            patient.created_by = request.user
            if request.user.is_role_patient:
                patient.patient_user = request.user
                patient.dentist_owner = request.user.dentist_owner
                patient.patient_name = request.user.display_name
            elif selected_patient:
                patient.patient_user = selected_patient
                patient.dentist_owner = selected_patient.dentist_owner or (request.user if request.user.is_role_dentist else None)
                patient.patient_name = selected_patient.display_name
            else:
                patient.dentist_owner = request.user if request.user.is_role_dentist else None

            patient.review_status = PatientAnalysis.REVIEW_UNREVIEWED
            patient.reviewed_by = None
            patient.reviewed_at = None
            # Start background processing
            patient.save()
            _start_analysis(patient)
            return redirect('analysis:processing', pk=patient.pk)
    else:
        initial = {}
        if request.user.is_role_patient:
            initial['patient_name'] = request.user.display_name
        form = PatientUploadForm(actor=request.user, initial=initial)

    return render(request, 'analysis/upload.html', {'form': form})


@login_required
def processing_view(request, pk):
    """Loading screen with dental quotes while analysis runs."""
    patient = get_object_or_404(PatientAnalysis, pk=pk)
    if not _can_access_report(request.user, patient):
        return HttpResponse('Access denied.', status=403)
    
    # If already complete, redirect to results
    if patient.status == 'completed':
        return redirect('analysis:results', pk=pk)

    return render(request, 'analysis/processing.html', {
        'patient': patient,
        'quotes': json.dumps(DENTAL_QUOTES),
    })


@login_required
def results_view(request, pk):
    """Display analysis results with scores and Grad-CAM overlays."""
    patient = get_object_or_404(PatientAnalysis, pk=pk)
    if not _can_access_report(request.user, patient):
        return HttpResponse('Access denied.', status=403)

    if patient.status in ('pending', 'processing'):
        return redirect('analysis:processing', pk=pk)
    if patient.status == 'failed':
        return redirect('analysis:processing', pk=pk)

    # Score descriptions for display
    mgi_descriptions = {
        0: 'Absence of inflammation',
        1: 'Mild inflammation — slight change in color, little change in texture',
        2: 'Moderate inflammation — moderate glazing, redness, edema, hypertrophy',
        3: 'Severe inflammation — marked redness, edema, ulceration, tendency to bleed',
        4: 'Severe inflammation — marked redness, edema, ulceration, spontaneous bleeding',
    }

    ohi_descriptions = {
        0: 'Good — no debris or calculus',
        1: 'Fair — debris/calculus covering up to 1/3 of tooth surface',
        2: 'Poor — debris/calculus covering 1/3 to 2/3 of tooth surface',
        3: 'Very Poor — debris/calculus covering more than 2/3 of tooth surface',
    }

    gei_descriptions = {
        0: 'No enlargement',
        1: 'Mild enlargement — slight enlargement confined to interdental papilla',
        2: 'Moderate enlargement — enlargement of papilla and/or marginal gingiva',
        3: 'Severe enlargement — marked enlargement covering a significant portion of the clinical crown',
    }

    pi_descriptions = {
        0: 'No plaque',
        1: 'Separate flecks of plaque at the cervical margin',
        2: 'A thin continuous band of plaque (up to 1 mm) at the cervical margin',
        3: 'A band of plaque wider than 1 mm, but covering less than one-third of the crown',
        4: 'Plaque covering one-third to two-thirds of the crown',
        5: 'Plaque covering two-thirds or more of the crown',
    }

    can_view_revisions = request.user.is_role_dentist or request.user.is_role_admin
    revisions = patient.revisions.select_related('edited_by').all() if can_view_revisions else ReportRevision.objects.none()

    plaque_ratio = patient.plaque_ratio if patient.plaque_ratio is not None else patient.ai_plaque_ratio
    plaque_score = patient.plaque_score if patient.plaque_score is not None else patient.ai_plaque_score
    plaque_label = patient.plaque_label or patient.ai_plaque_label

    if plaque_label is None and plaque_score is not None:
        plaque_label = {0: 'None', 1: 'Low', 2: 'Medium', 3: 'High'}.get(plaque_score, 'Unknown')

    low_confidence_warning = False
    if patient.ai_mgi_confidence is not None and patient.ai_mgi_confidence < 50.0:
        low_confidence_warning = True
    if patient.ai_ohi_confidence is not None and patient.ai_ohi_confidence < 50.0:
        low_confidence_warning = True
    if patient.ai_gei_confidence is not None and patient.ai_gei_confidence < 50.0:
        low_confidence_warning = True

    context = {
        'patient': patient,
        'low_confidence_warning': low_confidence_warning,
        'mgi_desc': mgi_descriptions.get(patient.mgi_score, 'N/A'),
        'ohi_desc': ohi_descriptions.get(patient.ohi_score, 'N/A'),
        'gei_desc': gei_descriptions.get(patient.gei_score, 'N/A'),
        'pi_desc': pi_descriptions.get(plaque_score, 'N/A') if plaque_score is not None else 'N/A',
        'mgi_max': 4,
        'ohi_max': 3,
        'gei_max': 3,
        'status_banner': _get_status_banner(patient, viewer=request.user),
        'is_unreviewed': patient.review_status == PatientAnalysis.REVIEW_UNREVIEWED,
        'can_review': _can_review_report(request.user, patient),
        'review_form': ReviewReportForm(patient=patient),
        'show_old_and_new': request.user.is_role_dentist or request.user.is_role_admin,
        'show_revisions': can_view_revisions,
        'revisions': revisions,
        'plaque_ratio': plaque_ratio,
        'plaque_percent': plaque_ratio * 100.0 if plaque_ratio is not None else None,
        'plaque_score': plaque_score,
        'plaque_label': plaque_label,
        'plaque_badge_style': _plaque_badge_style(plaque_score),
    }

    return render(request, 'analysis/results.html', context)


@never_cache
@login_required
def check_status(request, pk):
    """AJAX endpoint to check processing status."""
    patient = get_object_or_404(PatientAnalysis, pk=pk)
    if not _can_access_report(request.user, patient):
        return JsonResponse({'error': 'Access denied'}, status=403)

    review_status_for_viewer = patient.review_status
    if request.user.is_role_patient and patient.review_status == PatientAnalysis.REVIEW_REJECTED:
        review_status_for_viewer = PatientAnalysis.REVIEW_APPROVED

    data = {
        'status': patient.status,
        'unique_code': patient.unique_code,
        'review_status': review_status_for_viewer,
        'status_banner_text': _get_status_banner(patient, viewer=request.user)['text'],
    }
    if patient.status == 'completed':
        data['results'] = {
            'mgi_score': patient.mgi_score,
            'ohi_score': patient.ohi_score,
            'gei_score': patient.gei_score,
            'plaque_score': patient.plaque_score,
            'plaque_ratio': patient.plaque_ratio,
            'plaque_label': patient.plaque_label,
        }
    elif patient.status == 'failed':
        data['error'] = patient.error_message
    return JsonResponse(data)


@login_required
def lookup_view(request):
    """Authenticated lookup for users with access to a report."""
    patient = None
    error = None

    if request.method == 'POST':
        code = request.POST.get('unique_code', '').strip()
        if code:
            try:
                patient = PatientAnalysis.objects.get(unique_code=code)
                if not _can_access_report(request.user, patient):
                    raise PatientAnalysis.DoesNotExist
                if patient.status == 'completed':
                    return redirect('analysis:results', pk=patient.pk)
                elif patient.status in ('pending', 'processing'):
                    return redirect('analysis:processing', pk=patient.pk)
                else:
                    error = f'Analysis failed: {patient.error_message}'
            except PatientAnalysis.DoesNotExist:
                error = 'No accessible patient found with that code.'

    return render(request, 'analysis/lookup.html', {'patient': patient, 'error': error})


@login_required
def dashboard_view(request):
    return redirect(_dashboard_redirect_for_user(request.user))


@login_required
def admin_dashboard_view(request):
    if not request.user.is_role_admin:
        return HttpResponse('Access denied.', status=403)

    reports = PatientAnalysis.objects.select_related('dentist_owner', 'patient_user', 'reviewed_by').all()[:120]
    users_qs = DentalUser.objects.select_related('dentist_owner').order_by('role', 'username')

    role_filter = request.GET.get('role', '').strip()
    search_query = request.GET.get('q', '').strip()

    if role_filter in {DentalUser.ROLE_ADMIN, DentalUser.ROLE_DENTIST, DentalUser.ROLE_PATIENT}:
        users_qs = users_qs.filter(role=role_filter)

    if search_query:
        users_qs = users_qs.filter(
            db_models.Q(username__icontains=search_query)
            | db_models.Q(first_name__icontains=search_query)
            | db_models.Q(last_name__icontains=search_query)
            | db_models.Q(phone_number__icontains=search_query)
            | db_models.Q(role__icontains=search_query)
        )

    paginator = Paginator(users_qs, 12)
    page_obj = paginator.get_page(request.GET.get('page'))

    query_params = {}
    if role_filter:
        query_params['role'] = role_filter
    if search_query:
        query_params['q'] = search_query
    query_string = urlencode(query_params)

    context = {
        'reports': reports,
        'users': page_obj.object_list,
        'page_obj': page_obj,
        'users_total': DentalUser.objects.count(),
        'dentists_total': DentalUser.objects.filter(role=DentalUser.ROLE_DENTIST).count(),
        'patients_total': DentalUser.objects.filter(role=DentalUser.ROLE_PATIENT).count(),
        'admin_user_form': AdminCreateUserForm(),
        'admin_password_form': AdminSetUserPasswordForm(),
        'search_query': search_query,
        'role_filter': role_filter,
        'query_string': query_string,
    }
    return render(request, 'analysis/admin_dashboard.html', context)


@login_required
def admin_create_user_view(request):
    if not request.user.is_role_admin:
        return HttpResponse('Access denied.', status=403)
    if request.method != 'POST':
        return redirect('analysis:admin_dashboard')

    form = AdminCreateUserForm(request.POST)
    if not form.is_valid():
        messages.error(request, 'Could not create user. Please correct the form errors.')
        return redirect('analysis:admin_dashboard')

    user = DentalUser.objects.create_user(
        username=form.cleaned_data['username'],
        password=form.cleaned_data['password'],
        role=form.cleaned_data['role'],
        first_name=form.cleaned_data.get('first_name', '').strip(),
        last_name=form.cleaned_data.get('last_name', '').strip(),
        phone_number=form.cleaned_data.get('phone_number') or None,
        dentist_owner=form.cleaned_data.get('dentist_owner'),
    )

    messages.success(request, f'User {user.username} was created successfully.')
    return redirect('analysis:admin_dashboard')


@login_required
def admin_delete_user_view(request, user_id):
    if not request.user.is_role_admin:
        return HttpResponse('Access denied.', status=403)
    if request.method != 'POST':
        return redirect('analysis:admin_dashboard')

    target = get_object_or_404(DentalUser, pk=user_id)

    if target.id == request.user.id:
        messages.error(request, 'You cannot delete your own admin account while logged in.')
        return redirect('analysis:admin_dashboard')

    if target.is_role_admin and DentalUser.objects.filter(role=DentalUser.ROLE_ADMIN).count() <= 1:
        messages.error(request, 'You cannot delete the last remaining admin account.')
        return redirect('analysis:admin_dashboard')

    username = target.username
    target.delete()
    messages.success(request, f'User {username} was removed successfully.')
    return redirect('analysis:admin_dashboard')


@login_required
def admin_set_user_password_view(request, user_id):
    if not request.user.is_role_admin:
        return HttpResponse('Access denied.', status=403)
    if request.method != 'POST':
        return redirect('analysis:admin_dashboard')

    target = get_object_or_404(DentalUser, pk=user_id)
    form = AdminSetUserPasswordForm(request.POST)
    if not form.is_valid():
        messages.error(request, f'Password update failed for {target.username}.')
        return redirect('analysis:admin_dashboard')

    target.set_password(form.cleaned_data['new_password'])
    target.save(update_fields=['password'])
    messages.success(request, f'Password was updated for {target.username}.')
    return redirect('analysis:admin_dashboard')


@login_required
def admin_reset_user_password_view(request, user_id):
    if not request.user.is_role_admin:
        return HttpResponse('Access denied.', status=403)
    if request.method != 'POST':
        return redirect('analysis:admin_dashboard')

    target = get_object_or_404(DentalUser, pk=user_id)
    temp_password = _generate_password()
    target.set_password(temp_password)
    target.save(update_fields=['password'])

    messages.success(
        request,
        f'Temporary password for {target.username}: {temp_password} (shown only once).',
    )
    return redirect('analysis:admin_dashboard')


@login_required
def dentist_dashboard_view(request):
    if not request.user.is_role_dentist and not request.user.is_role_admin:
        return HttpResponse('Access denied.', status=403)

    owner = request.user
    if request.user.is_role_admin:
        owner = None

    base_qs = PatientAnalysis.objects.select_related('patient_user', 'dentist_owner', 'reviewed_by')
    if owner:
        base_qs = base_qs.filter(dentist_owner=owner)

    unreviewed_reports = base_qs.filter(status='completed', review_status=PatientAnalysis.REVIEW_UNREVIEWED)
    approved_reports = base_qs.filter(status='completed').filter(
        db_models.Q(review_status=PatientAnalysis.REVIEW_APPROVED) | db_models.Q(revisions__isnull=False)
    ).distinct()
    rejected_reports = base_qs.filter(status='completed', review_status=PatientAnalysis.REVIEW_REJECTED)

    context = {
        'unreviewed_reports': unreviewed_reports,
        'approved_reports': approved_reports,
        'rejected_reports': rejected_reports,
        'patient_form': DentistCreatePatientForm(),
        'generated_patient_credentials': request.session.pop('generated_patient_credentials', None),
    }
    return render(request, 'analysis/dentist_dashboard.html', context)


@login_required
def patient_dashboard_view(request):
    if not request.user.is_role_patient and not request.user.is_role_admin:
        return HttpResponse('Access denied.', status=403)

    reports = PatientAnalysis.objects.select_related('reviewed_by').filter(patient_user=request.user).order_by('-created_at')
    if request.user.is_role_admin:
        reports = PatientAnalysis.objects.select_related('reviewed_by').all().order_by('-created_at')
    return render(request, 'analysis/patient_dashboard.html', {'reports': reports})


@login_required
def create_patient_account_view(request):
    if not request.user.is_role_dentist and not request.user.is_role_admin:
        return HttpResponse('Access denied.', status=403)

    if request.method != 'POST':
        return redirect('analysis:dentist_dashboard')

    form = DentistCreatePatientForm(request.POST)
    generated = None
    if form.is_valid():
        patient_name = form.cleaned_data['patient_name']
        phone_number = _normalize_phone(form.cleaned_data['phone_number'])

        if DentalUser.objects.filter(phone_number=phone_number).exists() or DentalUser.objects.filter(username=phone_number).exists():
            messages.error(request, 'This phone number is already registered.')
            return redirect('analysis:dentist_dashboard')

        username = phone_number
        password = _generate_password()

        user = DentalUser.objects.create_user(
            username=username,
            password=password,
            role=DentalUser.ROLE_PATIENT,
            first_name=patient_name,
            phone_number=phone_number,
            dentist_owner=request.user if request.user.is_role_dentist else None,
        )
        generated = {
            'username': username,
            'password': password,
            'patient_name': user.display_name,
            'phone_number': phone_number,
        }
        messages.success(request, f"Patient account created for {user.display_name}.")
    else:
        messages.error(request, 'Could not create patient account. Please check the entered name.')

    if generated:
        request.session['generated_patient_credentials'] = generated
    return redirect('analysis:dentist_dashboard')


@login_required
def review_report_view(request, pk):
    patient = get_object_or_404(PatientAnalysis, pk=pk)
    if not _can_review_report(request.user, patient):
        return HttpResponse('Access denied.', status=403)
    if patient.status != 'completed':
        messages.error(request, 'Only completed AI reports can be reviewed.')
        return redirect('analysis:processing', pk=pk)

    if request.method != 'POST':
        return redirect('analysis:results', pk=pk)

    form = ReviewReportForm(request.POST, patient=patient)
    if not form.is_valid():
        plaque_ratio = patient.plaque_ratio if patient.plaque_ratio is not None else patient.ai_plaque_ratio
        plaque_score = patient.plaque_score if patient.plaque_score is not None else patient.ai_plaque_score
        plaque_label = patient.plaque_label or patient.ai_plaque_label
        messages.error(request, 'Please correct the review form errors.')
        return render(request, 'analysis/results.html', {
            'patient': patient,
            'review_form': form,
            'mgi_max': 4,
            'ohi_max': 3,
            'gei_max': 3,
            'status_banner': _get_status_banner(patient, viewer=request.user),
            'is_unreviewed': patient.review_status == PatientAnalysis.REVIEW_UNREVIEWED,
            'can_review': True,
            'show_old_and_new': True,
            'show_revisions': True,
            'mgi_desc': 'N/A',
            'ohi_desc': 'N/A',
            'gei_desc': 'N/A',
            'plaque_ratio': plaque_ratio,
            'plaque_percent': plaque_ratio * 100.0 if plaque_ratio is not None else None,
            'plaque_score': plaque_score,
            'plaque_label': plaque_label,
            'plaque_badge_style': _plaque_badge_style(plaque_score),
        })

    action = form.cleaned_data['action']
    note = form.cleaned_data.get('reason', '').strip()
    patient.reviewed_by = request.user
    patient.reviewed_at = timezone.now()
    patient.dentist_note = note

    if action == ReviewReportForm.ACTION_APPROVE:
        patient.review_status = PatientAnalysis.REVIEW_APPROVED
    else:
        ReportRevision.objects.create(
            analysis=patient,
            edited_by=request.user,
            old_mgi_score=patient.mgi_score,
            old_ohi_score=patient.ohi_score,
            old_gei_score=patient.gei_score,
            new_mgi_score=form.cleaned_data['mgi_score'],
            new_ohi_score=form.cleaned_data['ohi_score'],
            new_gei_score=form.cleaned_data['gei_score'],
            reason=note,
        )
        patient.mgi_score = form.cleaned_data['mgi_score']
        patient.ohi_score = form.cleaned_data['ohi_score']
        patient.gei_score = form.cleaned_data['gei_score']
        patient.review_status = PatientAnalysis.REVIEW_REJECTED

    patient.save(update_fields=['mgi_score', 'ohi_score', 'gei_score', 'review_status', 'reviewed_by', 'reviewed_at', 'dentist_note', 'updated_at'])
    messages.success(request, 'Report review has been saved successfully.')
    return redirect('analysis:results', pk=pk)


@login_required
def download_report_pdf_view(request, pk):
    patient = get_object_or_404(PatientAnalysis, pk=pk)
    if not _can_access_report(request.user, patient):
        return HttpResponse('Access denied.', status=403)

    try:
        from textwrap import wrap
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception:
        return HttpResponse('PDF dependency is not installed. Add reportlab and retry.', status=500)

    def _score_text(value):
        return str(value) if value is not None else '-'

    def _confidence_text(value):
        if value is None:
            return 'N/A'
        return f"{float(value):.0f}%"

    def _severity_label(metric, value):
        if value is None:
            return 'Pending'
        if metric == 'MGI':
            if value <= 1:
                return 'Low'
            if value <= 2:
                return 'Moderate'
            return 'High'
        if metric == 'OHI':
            if value <= 1:
                return 'Good/Fair'
            if value == 2:
                return 'Poor'
            return 'Very Poor'
        if value == 0:
            return 'None'
        if value == 1:
            return 'Mild'
        return 'Moderate/Severe'

    mgi_descriptions = {
        0: 'Absence of inflammation',
        1: 'Mild inflammation',
        2: 'Moderate inflammation',
        3: 'Severe inflammation with bleeding tendency',
        4: 'Severe inflammation with spontaneous bleeding',
    }
    ohi_descriptions = {
        0: 'Good oral hygiene',
        1: 'Fair oral hygiene',
        2: 'Poor oral hygiene',
        3: 'Very poor oral hygiene',
    }
    gei_descriptions = {
        0: 'No gingival enlargement',
        1: 'Mild papillary enlargement',
        2: 'Moderate enlargement',
        3: 'Severe enlargement',
    }

    recommendations = []
    if patient.review_status == PatientAnalysis.REVIEW_UNREVIEWED:
        recommendations.append('Await dentist validation before making treatment decisions.')
    if patient.mgi_score is not None and patient.mgi_score >= 2:
        recommendations.append('Schedule periodontal evaluation for inflammation management.')
    if patient.ohi_score is not None and patient.ohi_score >= 2:
        recommendations.append('Reinforce oral hygiene protocol and plaque-control counseling.')
    if patient.gei_score is not None and patient.gei_score >= 1:
        recommendations.append('Assess gingival enlargement and review causative factors.')
    recommendations.extend([
        'Brush twice daily using soft bristles and fluoride toothpaste.',
        'Use interdental cleaning once daily (floss or interdental brush).',
        'Arrange follow-up review in 2 to 6 weeks based on clinical severity.',
    ])
    recommendations = recommendations[:6]

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="report_{patient.unique_code}.pdf"'

    c = canvas.Canvas(response, pagesize=A4)
    width, height = A4
    margin = 34
    content_w = width - (2 * margin)
    y = height - margin

    # Header banner
    header_h = 78
    c.setFillColor(colors.HexColor('#0f766e'))
    c.roundRect(margin, y - header_h, content_w, header_h, 10, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont('Helvetica-Bold', 19)
    c.drawString(margin + 16, y - 30, 'DentAI Clinical Oral Health Report')
    c.setFont('Helvetica', 10)
    c.drawString(margin + 16, y - 47, 'AI-assisted dental index assessment for clinical support')
    c.drawRightString(width - margin - 16, y - 30, f'Code: {patient.unique_code}')
    c.drawRightString(width - margin - 16, y - 47, timezone.localtime(patient.created_at).strftime('%d %b %Y %H:%M'))
    y -= (header_h + 14)

    # Patient and report metadata card
    meta_h = 88
    c.setFillColor(colors.HexColor('#f8fafc'))
    c.setStrokeColor(colors.HexColor('#d1d5db'))
    c.roundRect(margin, y - meta_h, content_w, meta_h, 8, stroke=1, fill=1)

    patient_phone = patient.patient_user.phone_number if patient.patient_user else '-'
    dentist_name = f"Dr. {patient.reviewed_by.display_name}" if patient.reviewed_by else 'Pending assignment'
    status = _get_status_banner(patient, viewer=request.user)['text']

    c.setFillColor(colors.HexColor('#111827'))
    c.setFont('Helvetica-Bold', 11)
    c.drawString(margin + 14, y - 22, 'Patient Information')
    c.drawString(margin + (content_w / 2), y - 22, 'Clinical Workflow Status')
    c.setFont('Helvetica', 9.6)
    c.drawString(margin + 14, y - 39, f'Patient Name: {patient.patient_name}')
    c.drawString(margin + 14, y - 54, f'Phone: {patient_phone}')
    c.drawString(margin + 14, y - 69, f'Uploaded By: {patient.created_by.display_name if patient.created_by else "-"}')

    c.drawString(margin + (content_w / 2), y - 39, f'Status: {status}')
    c.drawString(margin + (content_w / 2), y - 54, f'Reviewed By: {dentist_name}')
    c.drawString(
        margin + (content_w / 2),
        y - 69,
        f'Review Time: {timezone.localtime(patient.reviewed_at).strftime("%d %b %Y %H:%M") if patient.reviewed_at else "Pending"}',
    )
    y -= (meta_h + 12)

    # Score cards section
    card_h = 108
    card_gap = 10
    card_w = (content_w - (2 * card_gap)) / 3

    scores = [
        ('MGI', patient.mgi_score, 4, patient.mgi_confidence, mgi_descriptions.get(patient.mgi_score, 'N/A')),
        ('OHI', patient.ohi_score, 3, patient.ohi_confidence, ohi_descriptions.get(patient.ohi_score, 'N/A')),
        ('GEI', patient.gei_score, 2, patient.gei_confidence, gei_descriptions.get(patient.gei_score, 'N/A')),
    ]

    for idx, (label, value, max_value, confidence, desc) in enumerate(scores):
        x = margin + (idx * (card_w + card_gap))
        c.setFillColor(colors.HexColor('#ffffff'))
        c.setStrokeColor(colors.HexColor('#cbd5e1'))
        c.roundRect(x, y - card_h, card_w, card_h, 8, stroke=1, fill=1)

        c.setFillColor(colors.HexColor('#0f766e'))
        c.setFont('Helvetica-Bold', 11)
        c.drawString(x + 10, y - 18, label)

        c.setFillColor(colors.HexColor('#111827'))
        c.setFont('Helvetica-Bold', 20)
        c.drawString(x + 10, y - 43, f"{_score_text(value)} / {max_value}")

        c.setFont('Helvetica', 9)
        c.setFillColor(colors.HexColor('#374151'))
        c.drawString(x + 10, y - 59, f'Severity: {_severity_label(label, value)}')
        c.drawString(x + 10, y - 73, f'Confidence: {_confidence_text(confidence)}')

        desc_line = wrap(desc, 30)
        c.setFillColor(colors.HexColor('#4b5563'))
        c.setFont('Helvetica', 8.4)
        if desc_line:
            c.drawString(x + 10, y - 89, desc_line[0])
            if len(desc_line) > 1:
                c.drawString(x + 10, y - 101, desc_line[1])

    y -= (card_h + 12)

    # Interpretation summary
    summary_h = 96
    c.setFillColor(colors.HexColor('#ecfeff'))
    c.setStrokeColor(colors.HexColor('#99f6e4'))
    c.roundRect(margin, y - summary_h, content_w, summary_h, 8, stroke=1, fill=1)
    c.setFillColor(colors.HexColor('#0f172a'))
    c.setFont('Helvetica-Bold', 11)
    c.drawString(margin + 14, y - 20, 'Clinical Interpretation Summary')

    summary_lines = [
        f"MGI indicates {mgi_descriptions.get(patient.mgi_score, 'insufficient data')}.",
        f"OHI indicates {ohi_descriptions.get(patient.ohi_score, 'insufficient data')}.",
        f"GEI indicates {gei_descriptions.get(patient.gei_score, 'insufficient data')}.",
        'This AI-assisted report supports clinical decision-making and should be interpreted with examination findings.',
    ]

    c.setFont('Helvetica', 9.2)
    c.setFillColor(colors.HexColor('#334155'))
    text_y = y - 38
    for line in summary_lines:
        for segment in wrap(line, 110):
            c.drawString(margin + 14, text_y, segment)
            text_y -= 13

    y -= (summary_h + 12)

    # Recommendations and instructions section (two-column)
    col_h = 152
    col_gap = 10
    col_w = (content_w - col_gap) / 2

    c.setFillColor(colors.HexColor('#ffffff'))
    c.setStrokeColor(colors.HexColor('#d1d5db'))
    c.roundRect(margin, y - col_h, col_w, col_h, 8, stroke=1, fill=1)
    c.roundRect(margin + col_w + col_gap, y - col_h, col_w, col_h, 8, stroke=1, fill=1)

    c.setFillColor(colors.HexColor('#0f172a'))
    c.setFont('Helvetica-Bold', 11)
    c.drawString(margin + 12, y - 20, 'Recommended Follow-up Actions')
    c.drawString(margin + col_w + col_gap + 12, y - 20, 'Patient Home-Care Instructions')

    c.setFont('Helvetica', 8.8)
    c.setFillColor(colors.HexColor('#334155'))
    bullet_y = y - 38
    for rec in recommendations:
        lines = wrap(rec, 52)
        c.drawString(margin + 14, bullet_y, f'- {lines[0]}')
        bullet_y -= 12
        for cont in lines[1:]:
            c.drawString(margin + 23, bullet_y, cont)
            bullet_y -= 12
        bullet_y -= 2
        if bullet_y < (y - col_h + 12):
            break

    instruction_lines = [
        '1. Brush at least 2 minutes, twice daily.',
        '2. Clean interdental spaces once daily.',
        '3. Use alcohol-free antimicrobial mouth rinse.',
        '4. Limit sugary snacks and acidic beverages.',
        '5. Report bleeding, pain, or swelling early.',
        '6. Keep regular dental appointments.',
    ]

    instruction_y = y - 38
    for item in instruction_lines:
        c.drawString(margin + col_w + col_gap + 14, instruction_y, item)
        instruction_y -= 18

    y -= (col_h + 12)

    # Dentist note and audit box
    review_h = 74
    c.setFillColor(colors.HexColor('#f9fafb'))
    c.setStrokeColor(colors.HexColor('#d1d5db'))
    c.roundRect(margin, y - review_h, content_w, review_h, 8, stroke=1, fill=1)
    c.setFillColor(colors.HexColor('#111827'))
    c.setFont('Helvetica-Bold', 10.5)
    c.drawString(margin + 12, y - 18, 'Dentist Review Note and Audit Trace')

    note_text = patient.dentist_note.strip() if patient.dentist_note else 'No dentist note provided for this review cycle.'
    note_lines = wrap(note_text, 116)[:2]
    c.setFont('Helvetica', 8.9)
    c.setFillColor(colors.HexColor('#374151'))
    line_y = y - 35
    for line in note_lines:
        c.drawString(margin + 12, line_y, line)
        line_y -= 13

    c.drawRightString(width - margin - 12, y - 18, f'Revisions Recorded: {patient.revisions.count()}')
    c.drawRightString(width - margin - 12, y - 32, f'Generated by DentAI v1.0')
    y -= (review_h + 10)

    # Footer disclaimer
    foot_h = 46
    c.setFillColor(colors.HexColor('#fefce8'))
    c.setStrokeColor(colors.HexColor('#fde68a'))
    c.roundRect(margin, y - foot_h, content_w, foot_h, 8, stroke=1, fill=1)
    c.setFillColor(colors.HexColor('#713f12'))
    c.setFont('Helvetica-Bold', 9.5)
    c.drawString(margin + 12, y - 17, 'Clinical Disclaimer')
    c.setFont('Helvetica', 8.4)
    c.drawString(
        margin + 12,
        y - 31,
        'This report is AI-assisted and not a standalone diagnosis. Final treatment decisions must be made by a licensed dentist.',
    )

    c.showPage()
    c.save()
    return response


def _start_analysis(patient):
    """
    Start the ML analysis in a background thread.
    In production, this would be a Celery task.
    """
    patient.status = 'processing'
    patient.save()

    thread = threading.Thread(target=_run_analysis, args=(patient.pk,))
    thread.daemon = True
    thread.start()


def _run_analysis(patient_pk):
    """Run the ML inference pipeline."""
    import django
    django.setup()

    from analysis.models import PatientAnalysis

    try:
        patient = PatientAnalysis.objects.get(pk=patient_pk)

        import time
        start_time = time.time()
        
        predictor = get_predictor()
        if not predictor:
            raise ValueError("AI Predictor could not be initialized.")

        # Get image paths
        frontal_path = patient.frontal_image.path
        left_path = patient.left_lateral_image.path
        right_path = patient.right_lateral_image.path

        # Run prediction
        predictions = predictor.predict(frontal_path, left_path, right_path)
        
        # Latency tracking
        latency = time.time() - start_time
        global _LAST_INFERENCE_LATENCY
        _LAST_INFERENCE_LATENCY = latency

        # Compute plaque metrics using the new pi_estimator BEFORE logging
        import cv2
        from inference.pi_estimator import estimate_pi
        
        try:
            frontal_rgb = cv2.cvtColor(cv2.imread(frontal_path), cv2.COLOR_BGR2RGB)
            left_rgb    = cv2.cvtColor(cv2.imread(left_path),    cv2.COLOR_BGR2RGB)
            right_rgb   = cv2.cvtColor(cv2.imread(right_path),   cv2.COLOR_BGR2RGB)
            
            pi_result = estimate_pi(
                frontal_rgb=frontal_rgb,
                left_rgb=left_rgb,
                right_rgb=right_rgb,
                predicted_ohi=predictions['ohi']['score'],
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"pi_estimator failed: {e}")
            pi_result = {"pi_score": 0, "pi_raw": 0.0, "coverage_f": 0.0, "confidence": "low"}
            
        predictions['pi'] = pi_result  # Inject into the predictions dictionary

        # Log prediction to CSV
        log_file = os.path.join(settings.BASE_DIR, 'inference_log.csv')
        log_exists = os.path.exists(log_file)
        try:
            with open(log_file, 'a') as f:
                if not log_exists:
                    f.write("timestamp,patient_id,mgi,ohi,gei,pi\\n")
                pid = patient.patient_user.id if patient.patient_user else ""
                f.write(f"{timezone.now().isoformat()},{pid},{predictions['mgi']['score']},{predictions['ohi']['score']},{predictions['gei']['score']},{predictions['pi']['pi_score']}\\n")
        except Exception as e:
            logger.warning(f"Could not write to inference log: {e}")

        # Save results
        patient.ai_mgi_score = predictions['mgi']['score']
        patient.ai_ohi_score = predictions['ohi']['score']
        patient.ai_gei_score = predictions['gei']['score']

        patient.mgi_score = predictions['mgi']['score']
        patient.ohi_score = predictions['ohi']['score']
        patient.gei_score = predictions['gei']['score']

        # Convert to percentage
        patient.ai_mgi_confidence = predictions['mgi']['confidence'] * 100
        patient.ai_ohi_confidence = predictions['ohi']['confidence'] * 100
        patient.ai_gei_confidence = predictions['gei']['confidence'] * 100

        patient.mgi_confidence = predictions['mgi']['confidence'] * 100
        patient.ohi_confidence = predictions['ohi']['confidence'] * 100
        patient.gei_confidence = predictions['gei']['confidence'] * 100
        patient.review_status = PatientAnalysis.REVIEW_UNREVIEWED
        patient.reviewed_by = None
        patient.reviewed_at = None

        # Mapping confidence string to float for the model fields
        conf_map = {"high": 100.0, "medium": 60.0}
        conf_val = conf_map.get(pi_result.get("confidence", "low").split()[0], 30.0)

        patient.ai_plaque_score = pi_result['pi_score']
        patient.plaque_score = pi_result['pi_score']
        
        # We can store the weighted coverage in ratio fields if needed
        patient.ai_plaque_ratio = pi_result['pi_raw'] / 5.0  # Normalize back to [0,1] if raw is 0-5
        patient.plaque_ratio = pi_result['pi_raw'] / 5.0
        
        # Plaque label based on score
        labels = {0: "No plaque", 1: "Trace", 2: "Mild", 3: "Moderate", 4: "Heavy", 5: "Severe"}
        pl_label = labels.get(pi_result['pi_score'], "Unknown")
        patient.ai_plaque_label = pl_label
        patient.plaque_label = pl_label
        
        patient.ai_plaque_confidence = conf_val
        patient.plaque_confidence = conf_val

        # Save Grad-CAM images if available
        if predictions.get('gradcam'):
            gradcam_dir = os.path.join(settings.MEDIA_ROOT, 'gradcam')
            os.makedirs(gradcam_dir, exist_ok=True)

            for view_name, overlay in predictions['gradcam'].items():
                filename = f"{patient.unique_code}_{view_name}.jpg"
                filepath = os.path.join(gradcam_dir, filename)
                overlay.save(filepath, 'JPEG', quality=90)

                if view_name == 'frontal':
                    patient.gradcam_frontal = f'gradcam/{filename}'
                elif view_name == 'left_lateral':
                    patient.gradcam_left = f'gradcam/{filename}'
                else:
                    patient.gradcam_right = f'gradcam/{filename}'

        patient.status = 'completed'
        patient.save()

    except Exception as e:
        logger.exception('Background analysis failed for patient %s: %s', patient_pk, e)
        try:
            patient = PatientAnalysis.objects.get(pk=patient_pk)
            patient.status = 'failed'
            patient.error_message = str(e)
            patient.save()
        except Exception:
            pass

from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from .forms import UserProfileForm

@login_required
def profile_view(request):
    user = request.user
    if request.method == 'POST':
        if 'update_profile' in request.POST:
            profile_form = UserProfileForm(request.POST, instance=user)
            password_form = PasswordChangeForm(user)
            if profile_form.is_valid():
                profile_form.save()
                messages.success(request, 'Profile details updated successfully.')
                return redirect('analysis:profile')
        elif 'change_password' in request.POST:
            profile_form = UserProfileForm(instance=user)
            password_form = PasswordChangeForm(user, request.POST)
            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(request, user)
                messages.success(request, 'Password updated successfully.')
                return redirect('analysis:profile')
            else:
                messages.error(request, 'Please correct the errors below.')
    else:
        profile_form = UserProfileForm(instance=user)
        password_form = PasswordChangeForm(user)
    
    for field in password_form.fields.values():
        field.widget.attrs.update({'class': 'w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-dental-500 focus:border-dental-500 transition-all outline-none bg-gray-50'})

    return render(request, 'analysis/profile.html', {
        'profile_form': profile_form,
        'password_form': password_form
    })

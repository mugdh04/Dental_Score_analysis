import os
import uuid
import json
import threading
from pathlib import Path

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.conf import settings
from PIL import Image

from .models import PatientAnalysis
from .forms import PatientUploadForm


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


def upload_view(request):
    """Main upload page where users submit dental photographs."""
    if request.method == 'POST':
        form = PatientUploadForm(request.POST, request.FILES)
        if form.is_valid():
            patient = form.save()
            # Start background processing
            _start_analysis(patient)
            return redirect('analysis:processing', pk=patient.pk)
    else:
        form = PatientUploadForm()

    return render(request, 'analysis/upload.html', {'form': form})


def processing_view(request, pk):
    """Loading screen with dental quotes while analysis runs."""
    patient = get_object_or_404(PatientAnalysis, pk=pk)
    
    # If already complete, redirect to results
    if patient.status == 'completed':
        return redirect('analysis:results', pk=pk)

    return render(request, 'analysis/processing.html', {
        'patient': patient,
        'quotes': json.dumps(DENTAL_QUOTES),
    })


def results_view(request, pk):
    """Display analysis results with scores and Grad-CAM overlays."""
    patient = get_object_or_404(PatientAnalysis, pk=pk)

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
        2: 'Moderate to severe enlargement — enlargement of papilla and/or marginal gingiva',
    }

    context = {
        'patient': patient,
        'mgi_desc': mgi_descriptions.get(patient.mgi_score, 'N/A'),
        'ohi_desc': ohi_descriptions.get(patient.ohi_score, 'N/A'),
        'gei_desc': gei_descriptions.get(patient.gei_score, 'N/A'),
        'mgi_max': 4,
        'ohi_max': 3,
        'gei_max': 2,
    }

    return render(request, 'analysis/results.html', context)


def check_status(request, pk):
    """AJAX endpoint to check processing status."""
    patient = get_object_or_404(PatientAnalysis, pk=pk)
    data = {
        'status': patient.status,
        'unique_code': patient.unique_code,
    }
    if patient.status == 'completed':
        data['results'] = {
            'mgi_score': patient.mgi_score,
            'ohi_score': patient.ohi_score,
            'gei_score': patient.gei_score,
        }
    elif patient.status == 'failed':
        data['error'] = patient.error_message
    return JsonResponse(data)


def lookup_view(request):
    """Look up patient results by unique code."""
    patient = None
    error = None

    if request.method == 'POST':
        code = request.POST.get('unique_code', '').strip()
        if code:
            try:
                patient = PatientAnalysis.objects.get(unique_code=code)
                if patient.status == 'completed':
                    return redirect('analysis:results', pk=patient.pk)
                elif patient.status in ('pending', 'processing'):
                    return redirect('analysis:processing', pk=patient.pk)
                else:
                    error = f'Analysis failed: {patient.error_message}'
            except PatientAnalysis.DoesNotExist:
                error = 'No patient found with that code. Please check and try again.'

    return render(request, 'analysis/lookup.html', {'patient': patient, 'error': error})


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

        # Import ML inference
        from ml.inference import predict_from_images

        # Get image paths
        frontal_path = patient.frontal_image.path
        left_path = patient.left_lateral_image.path
        right_path = patient.right_lateral_image.path

        # Run prediction
        predictions = predict_from_images(frontal_path, left_path, right_path)

        # Save results
        patient.mgi_score = predictions['mgi']['score']
        patient.ohi_score = predictions['ohi']['score']
        patient.gei_score = predictions['gei']['score']
        patient.mgi_confidence = predictions['mgi']['confidence']
        patient.ohi_confidence = predictions['ohi']['confidence']
        patient.gei_confidence = predictions['gei']['confidence']

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
        import traceback
        traceback.print_exc()
        try:
            patient = PatientAnalysis.objects.get(pk=patient_pk)
            patient.status = 'failed'
            patient.error_message = str(e)
            patient.save()
        except Exception:
            pass

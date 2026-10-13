# Normal photographs become optional; disclosing-agent photographs are added
# so PI can be scored from stained images and MGI/OHI/GEI from normal ones.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0010_dentist_image_request'),
    ]

    operations = [
        migrations.AlterField(
            model_name='patientanalysis',
            name='frontal_image',
            field=models.ImageField(blank=True, null=True, upload_to='uploads/frontal/'),
        ),
        migrations.AlterField(
            model_name='patientanalysis',
            name='left_lateral_image',
            field=models.ImageField(blank=True, null=True, upload_to='uploads/left_lateral/'),
        ),
        migrations.AlterField(
            model_name='patientanalysis',
            name='right_lateral_image',
            field=models.ImageField(blank=True, null=True, upload_to='uploads/right_lateral/'),
        ),
        migrations.AddField(
            model_name='patientanalysis',
            name='disclosed_frontal_image',
            field=models.ImageField(blank=True, null=True, upload_to='uploads/disclosed/frontal/'),
        ),
        migrations.AddField(
            model_name='patientanalysis',
            name='disclosed_left_lateral_image',
            field=models.ImageField(blank=True, null=True, upload_to='uploads/disclosed/left_lateral/'),
        ),
        migrations.AddField(
            model_name='patientanalysis',
            name='disclosed_right_lateral_image',
            field=models.ImageField(blank=True, null=True, upload_to='uploads/disclosed/right_lateral/'),
        ),
    ]

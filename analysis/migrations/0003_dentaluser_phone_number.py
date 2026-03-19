from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0002_rename_analysis_pat_dentist_8ec29f_idx_analysis_pa_dentist_1ca651_idx_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='dentaluser',
            name='phone_number',
            field=models.CharField(blank=True, db_index=True, max_length=20, null=True, unique=True),
        ),
    ]

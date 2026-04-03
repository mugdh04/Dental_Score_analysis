from django.db import migrations, models


def _ensure_plaque_ratio_and_label_columns(apps, schema_editor):
    PatientAnalysis = apps.get_model('analysis', 'PatientAnalysis')
    table = PatientAnalysis._meta.db_table

    with schema_editor.connection.cursor() as cursor:
        description = schema_editor.connection.introspection.get_table_description(cursor, table)
        existing_columns = {col.name for col in description}

    desired_fields = {
        'plaque_ratio': models.FloatField(null=True, blank=True),
        'plaque_label': models.CharField(max_length=20, null=True, blank=True),
        'ai_plaque_ratio': models.FloatField(null=True, blank=True),
        'ai_plaque_label': models.CharField(max_length=20, null=True, blank=True),
    }

    for field_name, field in desired_fields.items():
        if field_name in existing_columns:
            continue
        field.set_attributes_from_name(field_name)
        schema_editor.add_field(PatientAnalysis, field)


def _noop_reverse(apps, schema_editor):
    # Keep columns on reverse to avoid data loss in local environments.
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0005_patientanalysis_clinical_recommendations'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunPython(_ensure_plaque_ratio_and_label_columns, _noop_reverse),
            ],
            state_operations=[
                migrations.AddField(
                    model_name='patientanalysis',
                    name='plaque_ratio',
                    field=models.FloatField(blank=True, null=True),
                ),
                migrations.AddField(
                    model_name='patientanalysis',
                    name='plaque_label',
                    field=models.CharField(blank=True, max_length=20, null=True),
                ),
                migrations.AddField(
                    model_name='patientanalysis',
                    name='ai_plaque_ratio',
                    field=models.FloatField(blank=True, null=True),
                ),
                migrations.AddField(
                    model_name='patientanalysis',
                    name='ai_plaque_label',
                    field=models.CharField(blank=True, max_length=20, null=True),
                ),
            ],
        ),
    ]

from django.db import migrations, models


def _ensure_clinical_recommendations_column(apps, schema_editor):
    PatientAnalysis = apps.get_model('analysis', 'PatientAnalysis')
    table = PatientAnalysis._meta.db_table

    with schema_editor.connection.cursor() as cursor:
        description = schema_editor.connection.introspection.get_table_description(cursor, table)
        existing_columns = {col.name for col in description}

    if 'clinical_recommendations' in existing_columns:
        return

    field = models.JSONField(blank=True, default=list)
    field.set_attributes_from_name('clinical_recommendations')
    schema_editor.add_field(PatientAnalysis, field)


def _noop_reverse(apps, schema_editor):
    # Intentionally keep the column in reverse migrations to avoid data loss.
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0004_patientanalysis_ai_plaque_confidence_and_more'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunPython(_ensure_clinical_recommendations_column, _noop_reverse),
            ],
            state_operations=[
                migrations.AddField(
                    model_name='patientanalysis',
                    name='clinical_recommendations',
                    field=models.JSONField(blank=True, default=list),
                ),
            ],
        ),
    ]

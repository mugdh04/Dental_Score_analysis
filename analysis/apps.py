import logging
import os
import sys
import threading

from django.conf import settings
from django.apps import AppConfig


# -----------------------------------------------------------------------------
# Change Note (2026-04-03)
# Added safe model warm-up on app startup to reduce first-inference latency while
# avoiding duplicate warmups during autoreload and management commands.
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)


class AnalysisConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'analysis'

    def ready(self):
        """Warm up ML model cache at startup when enabled by settings."""
        if not getattr(settings, 'WARMUP_MODEL_ON_STARTUP', False):
            return

        skip_commands = {'migrate', 'makemigrations', 'collectstatic', 'shell', 'test'}
        if any(cmd in sys.argv for cmd in skip_commands):
            return

        # In runserver autoreload, warm up only in child process.
        if 'runserver' in sys.argv and os.environ.get('RUN_MAIN') != 'true':
            return

        thread = threading.Thread(target=self._warmup_models, daemon=True)
        thread.start()

    @staticmethod
    def _warmup_models():
        """Best-effort model warm-up so first request is not cold."""
        try:
            from analysis.views import get_predictor
            get_predictor()
            logger.info('ML model warm-up completed successfully.')
        except Exception as exc:
            logger.warning('ML model warm-up skipped due to error: %s', exc)

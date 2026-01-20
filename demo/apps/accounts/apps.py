from django.apps import AppConfig

class AccountsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.accounts'

    def ready(self):
        # PATCH FOR VERCEL READ-ONLY DEPLOYMENT
        # This disconnects the 'last_login' update that normally happens on login.
        # Without this, the app crashes because the DB is read-only.
        from django.contrib.auth.models import update_last_login
        from django.contrib.auth.signals import user_logged_in
        user_logged_in.disconnect(update_last_login, dispatch_uid='update_last_login')

from django.urls import path
from .views import diagnosisView

urlpatterns = [
    path("diag/", diagnosisView, name="diagnosis"),
]

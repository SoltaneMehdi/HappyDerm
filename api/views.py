from django.http import HttpRequest, HttpResponse, JsonResponse
from rest_framework import generics
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .predictor import predict
from PIL import Image
from io import BytesIO


# Create your views here.
@csrf_exempt
def diagnosisView(request):
    image_data = request.FILES["image"].read()
    image = Image.open(BytesIO(image_data))
    prediction = predict(image)
    prediction = {key: str(value) for key, value in prediction.items()}
    return JsonResponse(prediction)

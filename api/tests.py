from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from diagnosis.models import Lesion


# Create your tests here.
class APITests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.Lesion = Lesion.objects.create(
            image="/home/mehdi/Desktop/pfe master 2/HappyDerm/images/original/IMG_20190306_205757_413.jpg",
            malignant=False,
            type="hottie",
        )

    def test_api_listview(self):
        response = self.client.get(reverse("run_diagnosis"))  # pass name of url
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(Lesion.objects.count(), 1)
        self.assertContains(response, self.Lesion)

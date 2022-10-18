from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from base.models import Classification
import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '../../src/'))
import textmood as tm

@api_view(['GET'])
def getRoutes(request):
    routes = [
        'POST /api/classify/'
    ]
    return Response(routes)

@api_view(['POST'])
def getClassification(request):
    data = request.data
    sentence = data['sentence']
    try:
        my_model = tm.TextMoodModel("model", True)
        my_model.initialize_model()
        category = my_model.predict(sentence)
        category = 1
        return Response({'status':"succes", 'data': {'sentence':data['sentence'], 'classification':category}})
    except:
        return Response({'status':'fail'})
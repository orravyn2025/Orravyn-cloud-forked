from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

try:
    from .views import UserRegistrationAPIView, login_api_view, UserProfileAPIView
except ImportError:
    from rest_framework.views import APIView
    from rest_framework.response import Response
    
    class UserRegistrationAPIView(APIView):
        def post(self, request):
            return Response({'message': 'API not implemented yet'})
    
    class UserProfileAPIView(APIView):
        def get(self, request):
            return Response({'message': 'API not implemented yet'})
    
    def login_api_view(request):
        return Response({'message': 'API not implemented yet'})

urlpatterns = [
    path('auth/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    
    path('auth/register/', UserRegistrationAPIView.as_view(), name='api-register'),
    path('auth/profile/', UserProfileAPIView.as_view(), name='api-profile'),
]

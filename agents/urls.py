from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('runners/', views.runners, name='runners'),
    path('analytics/', views.analytics, name='analytics'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('settings/', views.settings, name='settings'),
    path('agent/<int:pk>/', views.AgentDetailView.as_view(), name='agent_detail'),
    
    # Chatbot API endpoints
    path('api/chatbot/message/', views.chatbot_message, name='chatbot_message'),
    path('api/conversation/<int:conversation_id>/', views.conversation_detail, name='conversation_detail'),
    path('api/conversation/<int:conversation_id>/delete/', views.delete_conversation, name='delete_conversation'),
]

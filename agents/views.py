from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import DetailView
from django.db.models import Q, Count
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import update_session_auth_hash
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import asyncio
from .models import ChatBot, Conversation, Message
from .chatbot_service import ChatBotManager, DataEngineerChatBot
from agents.forms import AgentForm
from .models import Agent, Runner
CHATBOT_AVAILABLE = True

def home(request):
    # Get search query
    query = request.GET.get('q')
    
    # Get all active agents
    agents = Agent.objects.filter(is_active=True)
    
    if query:
        agents = agents.filter(
            Q(name__icontains=query) | 
            Q(description__icontains=query) | 
            Q(skills__icontains=query)
        )
    
    agents = agents.order_by('-created_at')
    
    # Get user's runners if authenticated
    user_runners = []
    if request.user.is_authenticated:
        user_runners = Runner.objects.filter(owner=request.user, is_active=True)
    
    # Stats
    total_agents = Agent.objects.filter(is_active=True).count()
    user_runners_count = len(user_runners)
    
    context = {
        'query': query,
        'agents': agents,
        'user_runners': user_runners,
        'total_agents': total_agents,
        'user_runners_count': user_runners_count,
    }
    return render(request, 'agents/home.html', context)

@login_required
def dashboard(request):
    user_agents = Agent.objects.filter(owner=request.user).order_by('-created_at')
    user_runners = Runner.objects.filter(owner=request.user, is_active=True)
    
    context = {
        'user_agents': user_agents,
        'user_runners': user_runners,
    }
    return render(request, 'agents/dashboard.html', context)

class AgentDetailView(DetailView):
    model = Agent
    template_name = 'agents/agent_detail.html'
    context_object_name = 'agent'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.user.is_authenticated:
            context['user_runners'] = Runner.objects.filter(
                owner=self.request.user, 
                is_active=True
            )
        return context
    form_class = AgentForm
    template_name = 'agents/create_agent.html'
    
    def form_valid(self, form):
        form.instance.owner = self.request.user
        messages.success(self.request, 'Agent created successfully!')
        return super().form_valid(form)

class AgentDetailView(DetailView):
    model = Agent
    template_name = 'agents/agent_detail.html'
    context_object_name = 'agent'

@login_required
def runners(request):
    user_runners = Runner.objects.filter(owner=request.user)
    
    # Mock data for demonstration
    runner_stats = {
        'total_runners': user_runners.count(),
        'active_runners': user_runners.filter(is_active=True).count(),
        'total_runtime': '142h 30m',
        'avg_cpu_usage': '23%'
    }
    
    context = {
        'user_runners': user_runners,
        'runner_stats': runner_stats,
    }
    return render(request, 'agents/runners.html', context)

@login_required
def settings(request):
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'update_profile':
            # Handle profile update
            first_name = request.POST.get('first_name', '')
            last_name = request.POST.get('last_name', '')
            email = request.POST.get('email', '')
            
            request.user.first_name = first_name
            request.user.last_name = last_name
            request.user.email = email
            request.user.save()
            
            messages.success(request, 'Profile updated successfully!')
            
        elif action == 'change_password':
            # Handle password change
            current_password = request.POST.get('current_password')
            new_password = request.POST.get('new_password')
            confirm_password = request.POST.get('confirm_password')
            
            if request.user.check_password(current_password):
                if new_password == confirm_password and len(new_password) >= 8:
                    request.user.set_password(new_password)
                    request.user.save()
                    update_session_auth_hash(request, request.user)
                    messages.success(request, 'Password changed successfully!')
                else:
                    messages.error(request, 'New passwords do not match or are too short.')
            else:
                messages.error(request, 'Current password is incorrect.')
                
        elif action == 'delete_account':
            # Handle account deletion
            password = request.POST.get('delete_password')
            if request.user.check_password(password):
                username = request.user.username
                request.user.delete()
                messages.success(request, f'Account {username} has been deleted successfully.')
                return redirect('home')
            else:
                messages.error(request, 'Password is incorrect. Account not deleted.')
    
    context = {
        'user': request.user,
    }
    return render(request, 'agents/settings.html', context)

@login_required
def analytics(request):
    # Get user's agents and runners
    user_agents = Agent.objects.filter(owner=request.user)
    user_runners = Runner.objects.filter(owner=request.user)
    
    # Agent statistics
    total_agents = Agent.objects.filter(is_active=True).count()
    user_agent_count = user_agents.count()
    active_user_agents = user_agents.filter(is_active=True).count()
    
    # Most popular agents (mock data for now)
    popular_agents = Agent.objects.filter(is_active=True).order_by('?')[:5]
    
    # Agent categories/skills analysis
    all_agents = Agent.objects.filter(is_active=True)
    skill_data = {}
    for agent in all_agents:
        skills = agent.get_skills_list()
        for skill in skills:
            skill_data[skill] = skill_data.get(skill, 0) + 1
    
    # Sort skills by popularity and calculate widths
    top_skills = []
    sorted_skills = sorted(skill_data.items(), key=lambda x: x[1], reverse=True)[:10]
    max_count = max([count for skill, count in sorted_skills]) if sorted_skills else 1
    
    for skill, count in sorted_skills:
        width = min((count / max_count) * 100, 100)
        top_skills.append((skill, count, width))
    
    # Mock usage data (replace with real data from your tracking system)
    usage_stats = {
        'total_executions': 1247,
        'this_month_executions': 184,
        'avg_execution_time': '2.3 minutes',
        'success_rate': 94.2,
        'total_runtime': '342 hours',
        'data_processed': '1.2 TB'
    }
    
    # Recent activity (mock data)
    recent_activities = [
        {
            'agent_name': 'Data Scientist',
            'action': 'executed',
            'runner': 'Cloud Runner #1',
            'duration': '2h 15m',
            'status': 'success',
            'timestamp': '2 hours ago'
        },
        {
            'agent_name': 'Code Copilot', 
            'action': 'executed',
            'runner': 'Our Site',
            'duration': '45m',
            'status': 'running',
            'timestamp': '4 hours ago'
        },
        {
            'agent_name': 'Content Creator',
            'action': 'executed', 
            'runner': 'Local Runner',
            'duration': '1h 30m',
            'status': 'success',
            'timestamp': '1 day ago'
        }
    ]
    
    # Performance trends (mock data for charts)
    performance_data = {
        'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'executions': [45, 67, 123, 89, 156, 184],
        'success_rates': [91, 93, 95, 92, 96, 94],
        'avg_times': [2.8, 2.5, 2.3, 2.1, 2.4, 2.3]
    }
    
    context = {
        'user_agents': user_agents,
        'user_runners': user_runners,
        'total_agents': total_agents,
        'user_agent_count': user_agent_count,
        'active_user_agents': active_user_agents,
        'popular_agents': popular_agents,
        'top_skills': top_skills,
        'usage_stats': usage_stats,
        'recent_activities': recent_activities,
        'performance_data': performance_data,
    }
    return render(request, 'agents/analytics.html', context)

@login_required
def chatbot(request):
    """Main chatbot interface."""
    if not CHATBOT_AVAILABLE:
        return render(request, 'agents/chatbot_unavailable.html')
    
    try:
        chatbots = ChatBot.objects.filter(is_active=True)
        user_conversations = Conversation.objects.filter(user=request.user).order_by('-updated_at')[:10]
        
        context = {
            'chatbots': chatbots,
            'conversations': user_conversations,
        }
        return render(request, 'agents/chatbot.html', context)
    except Exception as e:
        # If tables don't exist yet, show setup page
        return render(request, 'agents/chatbot_setup.html', {'error': str(e)})

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def chatbot_message(request):
    """Handle chatbot message API with full DataEngineerChatBot integration."""
    if not CHATBOT_AVAILABLE:
        return JsonResponse({'error': 'Chatbot not available'}, status=503)
    
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        conversation_id = data.get('conversation_id')
        chatbot_id = data.get('chatbot_id', 1)
        
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
        
        # Check if chatbot exists
        try:
            chatbot = ChatBot.objects.get(id=chatbot_id, is_active=True)
        except ChatBot.DoesNotExist:
            return JsonResponse({'error': 'Chatbot not found'}, status=404)
        
        # Get or create conversation
        if not conversation_id:
            conversation = Conversation.objects.create(
                user=request.user,
                chatbot=chatbot,
                title=message[:50] + "..." if len(message) > 50 else message
            )
            conversation_id = conversation.id
        
        # Initialize the DataEngineerChatBot service
        try:
            chatbot_service = DataEngineerChatBot(chatbot_id)
            
            # Get the response from the chatbot service (using sync method)
            response_data = chatbot_service.get_response_sync(
                user_message=message,
                conversation_id=conversation_id,
                user_id=request.user.id
            )
            
            if response_data['success']:
                return JsonResponse({
                    'success': True,
                    'response': response_data['response'],
                    'metadata': response_data.get('metadata', {})
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': response_data.get('error', 'Unknown error'),
                    'response': response_data.get('response', 'Sorry, I encountered an error. Please try again.')
                }, status=500)
            
        except Exception as e:
            # Fallback to simple response if chatbot service fails
            conversation = Conversation.objects.get(id=conversation_id, user=request.user)
            
            # Save user message
            Message.objects.create(
                conversation=conversation,
                content=message,
                message_type='user'
            )
            
            # Generate fallback response
            fallback_response = f"""I'm experiencing some technical difficulties, but I can still help you with: "{message}"

As your Data Engineering Assistant, I can provide guidance on:
• Data pipeline design and optimization
• ETL/ELT processes and best practices  
• Database design and query optimization
• Big data technologies (Spark, Kafka, Airflow)
• Cloud data platforms and architecture
• Performance tuning and troubleshooting

Could you please provide more specific details about your challenge? I'll do my best to assist you!

Error details: {str(e)}"""
            
            # Save assistant response
            Message.objects.create(
                conversation=conversation,
                content=fallback_response,
                message_type='assistant',
                metadata={'model_used': 'fallback', 'error': str(e)}
            )
            
            return JsonResponse({
                'success': True,
                'response': fallback_response,
                'metadata': {
                    'conversation_id': conversation_id,
                    'model_used': 'fallback',
                    'tools_available': ['mcp_server', 'knowledge_base_search', 'web_search'],
                    'agent_mode': False
                }
            })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'response': 'Sorry, I encountered an error. Please try again.'
        }, status=500)

@login_required
def conversation_detail(request, conversation_id):
    """Get conversation details and messages."""
    if not CHATBOT_AVAILABLE:
        return JsonResponse({'error': 'Chatbot not available'}, status=503)
    
    try:
        conversation = Conversation.objects.get(id=conversation_id, user=request.user)
        messages = conversation.messages.all()
        
        messages_data = [{
            'id': msg.id,
            'content': msg.content,
            'type': msg.message_type,
            'timestamp': msg.created_at.isoformat(),
            'metadata': msg.metadata
        } for msg in messages]
        
        return JsonResponse({
            'success': True,
            'conversation': {
                'id': conversation.id,
                'title': conversation.title,
                'chatbot': conversation.chatbot.name,
                'created_at': conversation.created_at.isoformat()
            },
            'messages': messages_data
        })
        
    except Conversation.DoesNotExist:
        return JsonResponse({'error': 'Conversation not found'}, status=404)

@login_required
@csrf_exempt
@require_http_methods(["DELETE"])
def delete_conversation(request, conversation_id):
    """Delete a conversation using the chatbot service."""
    if not CHATBOT_AVAILABLE:
        return JsonResponse({'error': 'Chatbot not available'}, status=503)
    
    try:
        # Get chatbot_id from request or use default
        chatbot_id = request.GET.get('chatbot_id', 1)
        
        # Initialize the DataEngineerChatBot service
        chatbot_service = DataEngineerChatBot(chatbot_id)
        
        # Use the chatbot service to delete the conversation
        result = chatbot_service.delete_conversation(conversation_id, request.user.id)
        
        if result['success']:
            return JsonResponse(result)
        else:
            return JsonResponse(result, status=404)
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to delete conversation: {str(e)}'
        }, status=500)

@login_required
@csrf_exempt
@require_http_methods(["DELETE"])
def delete_all_conversations(request):
    """Delete all conversations for the current user."""
    if not CHATBOT_AVAILABLE:
        return JsonResponse({'error': 'Chatbot not available'}, status=503)
    
    try:
        # Get chatbot_id from request or use default
        chatbot_id = request.GET.get('chatbot_id', 1)
        
        # Initialize the DataEngineerChatBot service
        chatbot_service = DataEngineerChatBot(chatbot_id)
        
        # Use the chatbot service to delete all conversations
        result = chatbot_service.delete_all_conversations(request.user.id)
        
        return JsonResponse(result)
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to delete all conversations: {str(e)}'
        }, status=500)

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def clear_conversation_messages(request, conversation_id):
    """Clear all messages in a conversation but keep the conversation."""
    if not CHATBOT_AVAILABLE:
        return JsonResponse({'error': 'Chatbot not available'}, status=503)
    
    try:
        # Get chatbot_id from request or use default
        chatbot_id = request.GET.get('chatbot_id', 1)
        
        # Initialize the DataEngineerChatBot service
        chatbot_service = DataEngineerChatBot(chatbot_id)
        
        # Use the chatbot service to clear conversation messages
        result = chatbot_service.clear_conversation_messages(conversation_id, request.user.id)
        
        if result['success']:
            return JsonResponse(result)
        else:
            return JsonResponse(result, status=404)
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to clear conversation messages: {str(e)}'
        }, status=500)

@login_required
def get_user_conversations(request):
    """Get all conversations for the current user."""
    if not CHATBOT_AVAILABLE:
        return JsonResponse({'error': 'Chatbot not available'}, status=503)
    
    try:
        # Get chatbot_id from request or use default
        chatbot_id = request.GET.get('chatbot_id', 1)
        
        # Initialize the DataEngineerChatBot service
        chatbot_service = DataEngineerChatBot(chatbot_id)
        
        # Get user conversations
        conversations = chatbot_service.get_user_conversations(request.user.id)
        
        return JsonResponse({
            'success': True,
            'conversations': conversations
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to get conversations: {str(e)}'
        }, status=500)

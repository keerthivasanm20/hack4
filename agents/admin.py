from django.contrib import admin
from .models import Agent, Runner, ChatBot, Conversation, Message

@admin.register(Agent)
class AgentAdmin(admin.ModelAdmin):
    list_display = ['name', 'owner', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['name', 'description']

@admin.register(Runner)
class RunnerAdmin(admin.ModelAdmin):
    list_display = ['name', 'owner', 'runner_type', 'is_active', 'created_at']
    list_filter = ['runner_type', 'is_active', 'created_at']
    search_fields = ['name']

@admin.register(ChatBot)
class ChatBotAdmin(admin.ModelAdmin):
    list_display = ['name', 'chatbot_type', 'is_active', 'web_search_enabled', 'created_at']
    list_filter = ['chatbot_type', 'is_active', 'web_search_enabled']
    search_fields = ['name', 'description']

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'chatbot', 'created_at', 'updated_at']
    list_filter = ['chatbot', 'created_at']
    search_fields = ['title', 'user__username']

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ['conversation', 'message_type', 'content_preview', 'created_at']
    list_filter = ['message_type', 'created_at']
    search_fields = ['content']
    
    def content_preview(self, obj):
        return obj.content[:50] + "..." if len(obj.content) > 50 else obj.content
    content_preview.short_description = 'Content Preview'

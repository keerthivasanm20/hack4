from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

class Agent(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='agents')
    skills = models.CharField(max_length=500, help_text="Comma-separated skills")
    image = models.ImageField(upload_to='agents/', blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse('agent_detail', kwargs={'pk': self.pk})

    def get_skills_list(self):
        return [skill.strip() for skill in self.skills.split(',') if skill.strip()]

class Runner(models.Model):
    RUNNER_TYPES = [
        ('local', 'Local Runner'),
        ('cloud', 'Cloud Runner'),
        ('gpu', 'GPU Runner'),
        ('edge', 'Edge Runner'),
    ]
    
    name = models.CharField(max_length=200)
    runner_type = models.CharField(max_length=20, choices=RUNNER_TYPES, default='local')
    description = models.TextField(blank=True)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='runners')
    is_active = models.BooleanField(default=True)
    endpoint_url = models.URLField(blank=True, null=True)
    api_key = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.get_runner_type_display()})"

    class Meta:
        ordering = ['-created_at']

class ChatBot(models.Model):
    CHATBOT_TYPES = [
        ('data_engineer', 'Data Engineer Assistant'),
        ('architect', 'System Architect Assistant'),
        ('analyst', 'Data Analyst Assistant'),
        ('general', 'General AI Assistant'),
    ]
    
    name = models.CharField(max_length=100)
    chatbot_type = models.CharField(max_length=20, choices=CHATBOT_TYPES, default='data_engineer')
    description = models.TextField()
    system_prompt = models.TextField()
    is_active = models.BooleanField(default=True)
    knowledge_base_path = models.CharField(max_length=500, blank=True)
    web_search_enabled = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} ({self.get_chatbot_type_display()})"

class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    chatbot = models.ForeignKey(ChatBot, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.title}"
    
    class Meta:
        ordering = ['-updated_at']

class Message(models.Model):
    MESSAGE_TYPES = [
        ('user', 'User Message'),
        ('assistant', 'Assistant Message'),
        ('system', 'System Message'),
    ]
    
    conversation = models.ForeignKey(Conversation, related_name='messages', on_delete=models.CASCADE)
    content = models.TextField()
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES)
    metadata = models.JSONField(default=dict, blank=True)  # For storing sources, confidence, etc.
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.get_message_type_display()}: {self.content[:50]}..."
    
    class Meta:
        ordering = ['created_at']

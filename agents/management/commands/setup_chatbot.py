from django.core.management.base import BaseCommand
from agents.models import ChatBot

class Command(BaseCommand):
    help = 'Set up initial chatbot for data engineering'

    def handle(self, *args, **options):
        # Create Data Engineer ChatBot
        chatbot, created = ChatBot.objects.get_or_create(
            name='Data Engineer Assistant',
            chatbot_type='data_engineer',
            defaults={
                'description': 'AI assistant specialized in data engineering tasks, pipeline optimization, and technical guidance.',
                'system_prompt': '''You are an expert Data Engineer Assistant designed to help data engineers with their technical challenges.''',
                'is_active': True,
                'web_search_enabled': True,
            }
        )
        
        if created:
            self.stdout.write(
                self.style.SUCCESS(f'Successfully created chatbot: {chatbot.name}')
            )
        else:
            self.stdout.write(
                self.style.WARNING(f'Chatbot already exists: {chatbot.name}')
            )

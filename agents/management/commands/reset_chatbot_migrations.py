from django.core.management.base import BaseCommand
from django.db import connection
from agents.models import ChatBot

class Command(BaseCommand):
    help = 'Reset chatbot migrations and create initial data'

    def handle(self, *args, **options):
        # Check if tables exist
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE 'agents_%';
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
        self.stdout.write(f"Found tables: {tables}")
        
        # Create default chatbot if it doesn't exist
        try:
            chatbot, created = ChatBot.objects.get_or_create(
                name='Data Engineer Assistant',
                defaults={
                    'chatbot_type': 'data_engineer',
                    'description': 'AI assistant specialized in data engineering tasks, pipeline optimization, and technical guidance.',
                    'system_prompt': 'You are an expert Data Engineer Assistant...',
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
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error creating chatbot: {str(e)}')
            )

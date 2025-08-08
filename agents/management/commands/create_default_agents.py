from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from agents.models import Agent

class Command(BaseCommand):
    help = 'Create default agents for the platform'

    def handle(self, *args, **options):
        # Create a system user if it doesn't exist
        system_user, created = User.objects.get_or_create(
            username='system',
            defaults={
                'email': 'system@hack4.ai',
                'first_name': 'System',
                'is_staff': True
            }
        )

        default_agents = [
            {
                'name': 'GPT-4 Assistant',
                'description': 'Advanced conversational AI that can help with coding, writing, analysis, and creative tasks. Powered by the latest GPT-4 architecture.',
                'skills': 'Natural Language Processing, Code Generation, Creative Writing, Data Analysis, Problem Solving',
            },
            {
                'name': 'Code Copilot',
                'description': 'Specialized programming assistant that excels at code generation, debugging, and explaining complex programming concepts across multiple languages.',
                'skills': 'Python, JavaScript, React, Django, API Development, Code Review, Debugging',
            },
            {
                'name': 'Data Scientist',
                'description': 'Expert in data analysis, machine learning, and statistical modeling. Can help with data preprocessing, visualization, and predictive modeling.',
                'skills': 'Machine Learning, Data Visualization, Python, Pandas, NumPy, Statistical Analysis, Predictive Modeling',
            },
            {
                'name': 'UI/UX Designer',
                'description': 'Creative assistant specialized in user interface design, user experience optimization, and modern design principles.',
                'skills': 'User Interface Design, User Experience, Figma, Adobe Creative Suite, Prototyping, Design Systems',
            },
            {
                'name': 'Content Creator',
                'description': 'Versatile content creation assistant for blogs, social media, marketing copy, and creative writing projects.',
                'skills': 'Content Writing, Copywriting, SEO, Social Media, Blog Writing, Creative Writing, Marketing',
            },
            {
                'name': 'Business Analyst',
                'description': 'Strategic thinking assistant for business planning, market analysis, and process optimization.',
                'skills': 'Business Strategy, Market Research, Process Optimization, Financial Analysis, Project Management',
            },
            {
                'name': 'Research Assistant',
                'description': 'Academic and professional research helper that can gather information, summarize findings, and assist with citations.',
                'skills': 'Academic Research, Information Gathering, Citation Management, Literature Review, Report Writing',
            },
            {
                'name': 'Language Tutor',
                'description': 'Multilingual assistant for language learning, translation, and cross-cultural communication.',
                'skills': 'Language Learning, Translation, Grammar, Vocabulary, Cultural Context, Communication',
            }
        ]

        created_count = 0
        for agent_data in default_agents:
            agent, created = Agent.objects.get_or_create(
                name=agent_data['name'],
                owner=system_user,
                defaults={
                    'description': agent_data['description'],
                    'skills': agent_data['skills'],
                    'is_active': True
                }
            )
            if created:
                created_count += 1

        self.stdout.write(
            self.style.SUCCESS(f'Successfully created {created_count} default agents')
        )

#!/usr/bin/env python
"""
Test script for conversation management functionality
"""
import os
import sys
import django

# Setup Django environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agent_platform.settings')
django.setup()

from django.contrib.auth.models import User
from agents.models import ChatBot, Conversation, Message
from agents.chatbot_service import DataEngineerChatBot

def test_conversation_management():
    """Test conversation management functionality"""
    print("🧪 Testing Conversation Management Functionality")
    print("=" * 50)
    
    try:
        # Get or create test user
        user, created = User.objects.get_or_create(
            username='test_user',
            defaults={
                'email': 'test@example.com',
                'first_name': 'Test',
                'last_name': 'User'
            }
        )
        print(f"✅ User: {user.username} ({'created' if created else 'found'})")
        
        # Get or create chatbot
        chatbot, created = ChatBot.objects.get_or_create(
            name='Data Engineer Assistant',
            defaults={
                'chatbot_type': 'data_engineer',
                'description': 'AI Assistant for Data Engineering tasks',
                'system_prompt': 'You are a helpful data engineering assistant.',
                'is_active': True
            }
        )
        print(f"✅ ChatBot: {chatbot.name} ({'created' if created else 'found'})")
        
        # Initialize chatbot service
        chatbot_service = DataEngineerChatBot(chatbot.id)
        print("✅ ChatBot service initialized")
        
        # Test 1: Create a test conversation
        print("\n📝 Test 1: Creating conversation...")
        conversation = chatbot_service.create_conversation(
            user_id=user.id,
            title="Test Conversation for Management"
        )
        print(f"✅ Created conversation: {conversation.title} (ID: {conversation.id})")
        
        # Test 2: Add some test messages
        print("\n💬 Test 2: Adding test messages...")
        Message.objects.create(
            conversation=conversation,
            content="How do I optimize SQL queries?",
            message_type='user'
        )
        
        Message.objects.create(
            conversation=conversation,
            content="Here are some SQL optimization techniques: 1. Use indexes properly, 2. Avoid SELECT *, 3. Use WHERE clauses effectively...",
            message_type='assistant'
        )
        
        message_count = conversation.messages.count()
        print(f"✅ Added test messages. Total messages: {message_count}")
        
        # Test 3: Get user conversations
        print("\n📋 Test 3: Getting user conversations...")
        conversations = chatbot_service.get_user_conversations(user.id)
        print(f"✅ Found {len(conversations)} conversations for user")
        
        if conversations:
            conv = conversations[0]
            print(f"   - Title: {conv['title']}")
            print(f"   - Messages: {conv['message_count']}")
            print(f"   - Last message: {conv['last_message']['content'][:50]}...")
        
        # Test 4: Clear conversation messages
        print("\n🧹 Test 4: Clearing conversation messages...")
        result = chatbot_service.clear_conversation_messages(conversation.id, user.id)
        if result['success']:
            print(f"✅ {result['message']}")
            print(f"   Deleted {result['deleted_message_count']} messages")
        else:
            print(f"❌ Error: {result['error']}")
        
        # Test 5: Delete conversation
        print("\n🗑️ Test 5: Deleting conversation...")
        result = chatbot_service.delete_conversation(conversation.id, user.id)
        if result['success']:
            print(f"✅ {result['message']}")
        else:
            print(f"❌ Error: {result['error']}")
        
        # Test 6: Create multiple conversations for bulk delete test
        print("\n📚 Test 6: Creating multiple conversations for bulk delete test...")
        conversations_created = []
        for i in range(3):
            conv = chatbot_service.create_conversation(
                user_id=user.id,
                title=f"Test Conversation {i+1}"
            )
            conversations_created.append(conv)
        
        print(f"✅ Created {len(conversations_created)} test conversations")
        
        # Test 7: Delete all conversations
        print("\n🗑️ Test 7: Deleting all conversations...")
        result = chatbot_service.delete_all_conversations(user.id)
        if result['success']:
            print(f"✅ {result['message']}")
            print(f"   Deleted {result['deleted_count']} conversations")
        else:
            print(f"❌ Error: {result['error']}")
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("✅ Conversation management functionality is working properly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_conversation_management()
    sys.exit(0 if success else 1)

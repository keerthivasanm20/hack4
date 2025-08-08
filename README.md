# hack4
# 🧠 Data Engineer Assistant – Agent Listing & Integration Platform

A **Django-powered platform** for listing AI agents, integrating them into your own applications, and accessing a **Data Engineer Assistant** that uses **RAG (Retrieval-Augmented Generation)** and **Web Search** to provide accurate, up-to-date answers.

---

## 🚀 Features

- **Agent Listing** – Browse, search, and manage available AI agents.
- **Agent Integration** – Easily embed selected agents into your apps via API.
- **AI Data Engineer Assistant** – Get help with:
  - Data pipeline design
  - SQL optimization
  - ETL troubleshooting
  - Cloud data architecture
- **Powered by RAG** – Fetch relevant, context-aware information from your documents.
- **Web Search Integration** – Get real-time answers from the latest online sources.
- **Admin Dashboard** – Manage agents, integrations, and AI capabilities.

---

## 🛠️ Tech Stack

- **Backend** – [Django](https://www.djangoproject.com/) (Python 3.x)
- **Database** – PostgreSQL (default) or any Django-supported DB
- **AI Integration** – LLM API + RAG pipeline
- **Search** – Web Search API integration
- **Frontend** – Django Templates / Optional React/Vue integration

---

## 📦 Installation

# 1️⃣ **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/agent-platform.git](https://github.com/keerthivasanm20/hack4.git)
# 2️⃣ Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Set up environment variables
# Create a `.env` file in the project root with:
# ---------------------------------------------
# DEBUG=True
# SECRET_KEY=your_secret_key
# DATABASE_URL=postgres://user:pass@localhost:5432/dbname
# OPENAI_API_KEY=your_openai_api_key
# WEB_SEARCH_API_KEY=your_web_search_key
# ---------------------------------------------

# 5️⃣ Run database migrations
python manage.py migrate

# 6️⃣ Create an admin user
python manage.py createsuperuser

# 7️⃣ Start the development server
python manage.py runserver
   


from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from uuid import uuid4
from rapidfuzz import fuzz

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = str(uuid4())  # Secure session key
CORS(app, supports_credentials=True)  # Enable CORS for frontend integration

# Configure logging
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Expanded FAQ data covering all services, portfolio, and new categories
faq_data = [

# Greetings
{
    "question": "hi|hello|hey|greetings",
    "answer": "Hello! Welcome to StellSync Solutions. How can I assist you today?",
    "category": "greeting"
},
{
    "question": "good morning|morning",
    "answer": "Good morning! How can I help you with your tech needs?",
    "category": "greeting"
},
{
    "question": "good evening|evening",
    "answer": "Good evening! Ready to explore our services?",
    "category": "greeting"
},
{
    "question": "thank you|thanks",
    "answer": "You're welcome! Anything else I can help with?",
    "category": "polite"
},
{
    "question": "bye|goodbye",
    "answer": "Goodbye! Feel free to reach out anytime.",
    "category": "polite"
},

# Company Info
{
    "question": "about stellsync|tell me about your company|who are you",
    "answer": "StellSync Solutions, founded in 2022 in Sri Lanka, is a premier software development and data science firm. With a team of 10 experts, we've completed over 20 projects across retail, energy, education, and government sectors.",
    "category": "company"
},
{
    "question": "where are you located|address|location",
    "answer": "We are based at No. 123, Tech Street, Colombo 05, Sri Lanka.",
    "category": "company"
},
{
    "question": "vision|what is your vision",
    "answer": "Our vision is to be Sri Lanka's most innovative technology partner, delivering intelligent solutions that transform businesses across South Asia.",
    "category": "company"
},
{
    "question": "mission|what is your mission",
    "answer": "Our mission is to empower organizations with cutting-edge software and data solutions that solve real-world challenges and enhance efficiency.",
    "category": "company"
},
{
    "question": "how long in business|when founded|years of operation",
    "answer": "StellSync Solutions was founded in 2022, with 3 years of experience delivering innovative solutions.",
    "category": "company"
},
{
    "question": "company size|how many employees|team size",
    "answer": "We have a dedicated team of 10 highly skilled engineers and developers specializing in software and data science.",
    "category": "company"
},
{
    "question": "client types|who are your clients",
    "answer": "Our clients include startups, SMEs, and large enterprises in retail, energy, education, and government sectors.",
    "category": "company"
},
{
    "question": "certifications|awards|accreditations",
    "answer": "We are ISO 9001:2015 certified for quality management and have received the Sri Lanka Innovation Award in 2024.",
    "category": "company"
},
{
    "question": "industries served|sectors you work in",
    "answer": "We serve retail, energy, education, government, healthcare, and logistics industries with tailored solutions.",
    "category": "company"
},
{
    "question": "office hours|working hours|when are you open|what time are you open|opening hours|business hours|what time do you open|when do you open|when are you available",
    "answer": "Our office hours are Monday to Friday, 9 AM to 5 PM (Sri Lanka Time). Support is available 24/7 via email.",
    "category": "contact"
},

# Contact
{
    "question": "contact|how to reach you|how to contact",
    "answer": "Reach us at hello@stellsync.com, support@stellsync.com, or call +94 71 987 6543 (Sales) or +94 76 543 2109 (Support).",
    "category": "contact"
},
{
    "question": "email address|email",
    "answer": "You can email us at hello@stellsync.com or support@stellsync.com.",
    "category": "contact"
},
{
    "question": "phone number|call you|contact number",
    "answer": "Call us at +94 71 987 6543 for sales or +94 76 543 2109 for support.",
    "category": "contact"
},
{
    "question": "visit office|can i visit",
    "answer": "Yes, visit us at No. 123, Tech Street, Colombo 05, Sri Lanka. Please schedule an appointment via hello@stellsync.com.",
    "category": "contact"
},

# Services
{
    "question": "services|what do you offer|what services",
    "answer": "We offer custom software development, AI/ML solutions, data science and analytics, mobile app development, cloud solutions, IoT, big data processing, DevOps, database administration, and system integration.",
    "category": "services"
},
{
    "question": "software development|custom software|app development",
    "answer": "We build tailored applications using React, Angular, Node.js, .NET Core, Django, and Spring Boot, designed for performance and scalability.",
    "category": "services"
},
{
    "question": "mobile apps|mobile app development|build apps",
    "answer": "We develop cross-platform apps using Flutter and React Native, as well as native apps for iOS (Swift) and Android (Kotlin).",
    "category": "services"
},
{
    "question": "data science|analytics|data analysis",
    "answer": "Our data science services include predictive modeling, business intelligence dashboards, and analytics using Python, R, SQL, Tableau, and Power BI.",
    "category": "services"
},
{
    "question": "machine learning|ai|artificial intelligence",
    "answer": "We provide AI solutions including NLP, computer vision, predictive analytics, recommendation systems, and chatbots using BERT, LLAMA, TensorFlow, and PyTorch.",
    "category": "services"
},
{
    "question": "cloud|cloud computing|cloud solutions",
    "answer": "We offer cloud architecture, migration, and optimization on AWS, Azure, and Google Cloud, including serverless and containerized solutions.",
    "category": "services"
},
{
    "question": "iot|internet of things|iot projects",
    "answer": "We integrate IoT devices with big data platforms for real-time monitoring and analytics, using MQTT, ESP32, and AWS IoT Core.",
    "category": "services"
},
{
    "question": "data engineering|data warehouse|data lakehouse",
    "answer": "We provide data engineering solutions with Azure Data Factory, Snowflake, Microsoft Fabric, and lakehouses for efficient data pipelines.",
    "category": "services"
},
{
    "question": "devops|ci/cd|continuous integration",
    "answer": "Our DevOps services include CI/CD pipelines, containerization with Docker and Kubernetes, and infrastructure as code with Terraform.",
    "category": "services"
},
{
    "question": "big data|big data processing",
    "answer": "We handle large-scale data processing with Hadoop, Spark, and Kafka for real-time and batch analytics.",
    "category": "services"
},
{
    "question": "database|oracle|sql|database administration",
    "answer": "We offer database administration and optimization for Oracle, MySQL, PostgreSQL, MongoDB, and cloud-native databases.",
    "category": "services"
},
{
    "question": "chatbot|build chatbots|chatbot development",
    "answer": "We create rule-based and AI-powered chatbots using frameworks like Rasa, Dialogflow, and custom LLAMA models.",
    "category": "services"
},
{
    "question": "optimization|genetic algorithm|reinforcement learning|pso",
    "answer": "We develop optimization solutions using Genetic Algorithms, Reinforcement Learning, Particle Swarm Optimization, and simulated annealing.",
    "category": "services"
},
{
    "question": "time series|forecasting|time series forecasting",
    "answer": "Our forecasting solutions use LSTM, ARIMA, and Prophet for accurate predictions in demand and resource planning.",
    "category": "services"
},
{
    "question": "nlp|natural language processing",
    "answer": "We offer NLP solutions for text analysis, sentiment detection, entity recognition, and chatbot development.",
    "category": "services"
},
{
    "question": "recommendation system|recommender system",
    "answer": "We build recommendation systems using collaborative filtering, content-based filtering, and deep learning models.",
    "category": "services"
},
{
    "question": "web development|website development",
    "answer": "We develop responsive websites with React, Angular, Vue.js, Django, Flask, and Node.js.",
    "category": "services"
},
{
    "question": "testing|quality assurance|qa",
    "answer": "We provide QA services including automated testing with Selenium, unit testing, and performance testing.",
    "category": "services"
},
{
    "question": "security|cybersecurity|data protection",
    "answer": "We implement robust security measures including encryption, secure APIs, and compliance with GDPR and ISO standards.",
    "category": "services"
},
{
    "question": "maintenance|support|post-development",
    "answer": "We offer 24/7 maintenance and support, including bug fixes, updates, and monitoring.",
    "category": "services"
},

# Projects and Solutions
{
    "question": "projects|solutions|case studies|work samples|featured work",
    "answer": (
        "We have delivered impactful solutions:\n"
        "- Smart Solar Maintenance (94% fault detection accuracy)\n"
        "- Retail Demand Forecasting (23% improved accuracy)\n"
        "- AquaCare IoT Monitoring\n"
        "- EduAir Environmental Dashboard\n"
        "- QuickMart Grocery App\n"
        "- Supply Chain Optimization\n"
        "Let me know if you’d like details about any of these."
    ),
    "category": "projects"
},
{
    "question": "client feedback|testimonials|reviews",
    "answer": "Our clients appreciate our focus on results and collaboration. For example: 'StellSync's forecasting platform transformed our inventory management and improved profitability.'",
    "category": "projects"
},

# HR / Careers
{
    "question": "jobs|careers|vacancies|job openings|positions available|open positions|hiring now|how to find a job|how can i find a job|find a job|job opertunities at stellsync",

    "answer": "We’re always looking for talented individuals. View openings at www.stellsync.com/careers or email hr@stellsync.com",
    "category": "hr"
},
{
    "question": "apply for a job|how can i apply|submit application|send my cv|start a job|how can i start a job|join your team|get hired",
    "answer": "To apply, please email your resume to hr@stellsync.com with a short introduction about yourself.",
    "category": "hr"
},
{
    "question": "salary|compensation|remuneration|wages",
    "answer": "Salary varies by role and experience. We offer competitive compensation aligned with industry standards.",
    "category": "hr"
},
{
    "question": "working hours|work schedule|work time|office hours|shift timings",
    "answer": "Our regular working hours are Monday to Friday, 9 AM to 5 PM. Flexible and hybrid options are available.",
    "category": "hr"
},
{
    "question": "internship|graduate programs|trainee positions|student opportunities",
    "answer": "Yes, we offer internships and graduate programs. Email hr@stellsync.com for details.",
    "category": "hr"
},
{
    "question": "work culture|company culture|team atmosphere",
    "answer": "We foster a collaborative, inclusive environment that encourages innovation and work-life balance.",
    "category": "hr"
},
{
    "question": "benefits|perks|employee benefits|incentives",
    "answer": "Benefits include health insurance, development programs, bonuses, and flexible work.",
    "category": "hr"
},
{
    "question": "areas hiring|departments hiring|roles available",
    "answer": "We frequently hire in software development, data science, project management, and DevOps.",
    "category": "hr"
},

# Technical
{
    "question": "architecture|software architecture|design patterns",
    "answer": "We use microservices, MVC, and event-driven architectures, with patterns like Singleton and Factory.",
    "category": "technical"
},
{
    "question": "performance|optimization|scalability",
    "answer": "We optimize performance with load balancing, caching, and database indexing.",
    "category": "technical"
},
{
    "question": "tech stack|technologies used|tools",
    "answer": "Our tech stack includes Python, JavaScript, Java, C#, React, Angular, Node.js, Django, Flask, .NET Core, AWS, Azure, Docker, Kubernetes, and more.",
    "category": "technical"
},
{
    "question": "agile|methodology|development process",
    "answer": "We follow Agile (Scrum/Kanban), with iterative development and client collaboration.",
    "category": "technical"
},
{
    "question": "apis|api development|integration",
    "answer": "We develop RESTful and GraphQL APIs and integrate with third-party services.",
    "category": "technical"
},
{
    "question": "deployment|hosting|servers",
    "answer": "We deploy on AWS, Azure, and GCP with CI/CD pipelines.",
    "category": "technical"
},
{
    "question": "machine learning models|ml models|ai models",
    "answer": "We use LLAMA, BERT, TensorFlow, PyTorch for NLP, computer vision, and analytics.",
    "category": "technical"
},
{
    "question": "data pipelines|etl|data processing",
    "answer": "We build ETL pipelines with Apache Airflow, Azure Data Factory, and Talend.",
    "category": "technical"
},




# Payment and Billing
{
    "question": "discounts|special offers|promotions|deals",
    "answer": (
        "Yes! We periodically offer special discounts and promotions for new and returning clients. "
        "Please email sales@stellsync.com or visit our website for current offers and eligibility."
    ),
    "category": "billing"
},
{
    "question": (
        "installments|instalments|payment plan|pay in parts|split payment|pay in instalments|"
        "can I pay in instalments|can I pay by instalments|pay over time|flexible payment"
    ),
    "answer": (
        "We understand flexibility is important. Depending on the project size and duration, "
        "we can arrange milestone-based payments or instalment plans. Contact sales@stellsync.com to discuss options."
    ),
    "category": "billing"
},
{
    "question": (
        "payment methods|how to pay|card or cash|credit card|debit card|bank transfer|cash payment|"
        "payment options|i want to pay by card|i will pay by card|i like to pay by card|"
        "i like to pay in card|i prefer card payment|can i pay with card|pay using card|pay by card"
    ),
    "answer": (
        "We accept payments via bank transfer, credit/debit cards, and online payment gateways. "
        "For larger projects, milestone payments are recommended. Let us know your preferred method!"
    ),
    "category": "billing"
},
{
    "question": (
        "how to make a payment|how do i pay for a project|how to do a payment for a project|"
        "how to finalize payment|how do i complete payment|payment process|pay my invoice"
    ),
    "answer": (
        "To make a payment:\n"
        "1️⃣ Finalize your project details and agreement with our development team.\n"
        "2️⃣ We will issue an invoice with the payment amount and bank details.\n"
        "3️⃣ Make your advance or milestone payment via bank transfer, credit/debit card, or online gateway.\n"
        "4️⃣ Send the payment slip or confirmation to billing@stellsync.com or WhatsApp it to +94 76 543 2109.\n"
        "5️⃣ We will confirm receipt and proceed with development.\n\n"
        "If you have any questions, feel free to contact our billing team!"
    ),
    "category": "billing"
},
{
    "question": "invoices|billing|invoice details|billing information|invoice copy|get invoice|request invoice",
    "answer": (
        "Invoices are issued for each project milestone or monthly services. "
        "If you need a copy or have questions about billing, please email billing@stellsync.com."
    ),
    "category": "billing"
},
{
    "question": "refunds|cancellation|money back|cancel contract|cancel my project|get refund",
    "answer": (
        "Our contracts include terms for cancellation and refunds, depending on the project stage and work completed. "
        "Contact us to review your agreement details."
    ),
    "category": "billing"
},
{
    "question": "payment terms|due dates|payment deadline|when do I pay|when is payment due",
    "answer": (
        "Payment terms are usually milestone-based or monthly, with due dates specified in your agreement. "
        "We’ll always provide clear timelines and reminders before payments are due."
    ),
    "category": "billing"
},
{
    "question": "receipt|payment confirmation|proof of payment",
    "answer": (
        "After each payment, we issue a receipt and confirmation. If you need another copy, email billing@stellsync.com."
    ),
    "category": "billing"
}
,


# Client and Project Process
{
    "question": "start a project|how to start|new project",
    "answer": "Contact hello@stellsync.com to discuss your requirements. We'll provide a proposal and timeline.",
    "category": "process"
},
{
    "question": "nda|non-disclosure|confidentiality",
    "answer": "Yes, we sign NDAs to protect your confidentiality.",
    "category": "process"
},
{
    "question": "development time|how long|project timeline",
    "answer": "Project timelines vary. Small projects take 1-3 months, complex ones 6-12 months.",
    "category": "process"
},
{
    "question": "cost|pricing|how much",
    "answer": "Costs depend on complexity. Email hello@stellsync.com for a customized quote.",
    "category": "process"
},
{
    "question": "project management|how you manage projects",
    "answer": "We use Agile management with Jira/Trello and provide regular updates.",
    "category": "process"
},
{
    "question": "client collaboration|how you work with clients",
    "answer": "We collaborate closely via meetings, reports, and feedback sessions.",
    "category": "process"
},
{
    "question": "post-delivery|after project|support",
    "answer": "We offer post-delivery support, maintenance, updates, and 24/7 assistance.",
    "category": "process"
},
{
    "question": "projects and services|services and projects|what are you doing|what services you are giving|what services you provide|what projects you are doing|current projects|what project are you doing|tell me about your projects and services",
    "answer": (
        "We offer a wide range of services including:\n"
        "- Custom Software Development\n"
        "- AI/ML Solutions\n"
        "- Data Science & Analytics\n"
        "- Mobile App Development\n"
        "- Cloud Solutions\n"
        "- IoT & Big Data\n"
        "- DevOps & System Integration\n\n"
        "We have also delivered impactful solutions such as:\n"
        "- Smart Solar Maintenance System (94% fault detection accuracy)\n"
        "- Retail Demand Forecasting Engine (23% improved accuracy)\n"
        "- AquaCare IoT Monitoring\n"
        "- EduAir Environmental Dashboard\n"
        "- QuickMart Grocery App\n"
        "- Supply Chain Optimization Software\n\n"
        "Let me know if you’d like more details about any of these."
    ),
    "category": "combined"
},

]


# Preprocess text for better matching
stop_words = set(stopwords.words('english'))

from spellchecker import SpellChecker

spell = SpellChecker()

def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    corrected = [spell.correction(word) for word in tokens]
    # Replace None with the original word
    safe_corrected = [c if c is not None else w for c, w in zip(corrected, tokens)]
    tokens = [t for t in safe_corrected if t not in stop_words]
    return " ".join(tokens)



# Prepare questions for vectorization
questions = [item["question"] for item in faq_data]
processed_questions = [preprocess_text(q.split('|')[0]) for q in questions]  # Process primary question

# Initialize TF-IDF vectorizer with n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
question_vectors = vectorizer.fit_transform(processed_questions)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            logger.warning("Invalid request: Missing 'message' field")
            return jsonify({"response": "Invalid request. Please send a JSON with a 'message' field."}), 400

        user_message = data.get("message", "").strip()
        if not user_message:
            logger.info("Empty message received")
            return jsonify({"response": "Please enter a message so I can assist you."}), 400

        # Preprocess user message
        processed_user_message = preprocess_text(user_message)

        best_match = None
        best_score = 0
        best_item = None

        for item in faq_data:
            patterns = item["question"].split('|')
            for pattern in patterns:
                similarity = fuzz.partial_ratio(pattern.strip(), user_message.lower())
                if similarity > best_score:
                    best_score = similarity
                    best_item = item

        if best_score >= 80:
            logger.info(f"Fuzzy best match: '{best_item['question']}' ({best_score}%)")
            return jsonify({"response": best_item["answer"]})


        # Fallback to TF-IDF similarity
        user_vector = vectorizer.transform([processed_user_message])
        similarities = cosine_similarity(user_vector, question_vectors).flatten()
        max_index = np.argmax(similarities)
        max_score = similarities[max_index]

        threshold = 0.3  # Lowered threshold for broader matching

        if max_score >= threshold:
            response = faq_data[max_index]["answer"]
            session['last_category'] = faq_data[max_index].get('category', '')
            logger.info(f"Similarity match for '{user_message}': {response} (score: {max_score})")
        else:
            # Check context for follow-up questions
            last_category = session.get('last_category', '')
            if last_category == "greeting":
                response = "It seems you're just saying hi! Want to know about our services or projects?"
            elif last_category == "company":
                response = "I couldn't find a match. Want to learn more about StellSync's history, team, or achievements?"
            elif last_category == "contact":
                response = "Not sure about your question. Would you like our email, phone, or office details?"
            elif last_category == "services":
                response = "I didn't catch that. Are you asking about specific services like AI, web development, or cloud solutions?"
            elif last_category == "portfolio":
                response = "No match found. Interested in details about projects like QuickMart or AquaCare?"
            elif last_category == "technical":
                response = "That’s a bit unclear. Want to dive deeper into our tech stack or development methods?"
            elif last_category == "hr":
                response = "Not sure I got that. Are you asking about jobs, our team, or work culture?"
            elif last_category == "process":
                response = "I didn’t understand. Want to know more about starting a project or our process?"
            else:
                response = "I'm not sure I understand. Could you rephrase or ask about our services, like AI, software development, or our portfolio?"
            logger.info(f"No match for '{user_message}' (score: {max_score})")

        return jsonify({"response": response})

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"response": "An error occurred. Please try again later."}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
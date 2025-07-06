import streamlit as st
import mysql.connector
import os
from openai import OpenAI
import json
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import plotly.graph_objects as go
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from io import BytesIO
import time

# Configure API Key
os.environ["DEEPSEEK_API_KEY"] = "sk-79e23726e1f040dbbde0925cd398b098"

class DeepSeekAPI:
    """DeepSeek API调用类，支持将响应日志保存为JSON文件"""
    
    def __init__(self, api_key=None, base_url="https://api.deepseek.com/v1", log_file="api_responses.json"):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=base_url
        )
        self.log_file = log_file

    def invoke(self, prompt, model="deepseek-chat", temperature=0.2):
        """调用DeepSeek API并返回响应内容"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            
            # 提取响应内容
            content = response.choices[0].message.content
            
            # 记录日志
            self._log_response(prompt, content)
            
            return content
        except Exception as e:
            st.error(f"API调用错误: {str(e)}")
            return None

    def _log_response(self, prompt, response):
        """记录API调用日志到JSON文件"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response
        }
        
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, "r") as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            with open(self.log_file, "w") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            st.warning(f"日志记录失败: {str(e)}")

# 初始化DeepSeek LLM
llm = DeepSeekAPI()

# Database configuration
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="123456",  # 替换为你的密码
    database="learning_assistant"
)
cursor = db.cursor(dictionary=True)

# Database initialization
def init_db():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE,
            password VARCHAR(255),
            interests TEXT,
            learning_style TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_paths (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            subject VARCHAR(100),
            progress FLOAT,
            difficulty_level VARCHAR(20),
            content JSON,
            target_completion_date DATE,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            subject VARCHAR(100),
            topic VARCHAR(100),
            score FLOAT,
            feedback TEXT,
            taken_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS certifications (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            learning_path_id INT,
            completion_date DATE,
            certificate_number VARCHAR(50) UNIQUE,
            recipient_name VARCHAR(100),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (learning_path_id) REFERENCES learning_paths(id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS path_assessments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            learning_path_id INT,
            user_id INT,
            question TEXT,
            user_answer TEXT,
            score FLOAT,
            feedback TEXT,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (learning_path_id) REFERENCES learning_paths(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    db.commit()


def generate_certificate(name, course, completion_date, certificate_number):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Certificate border
    c.setStrokeColorRGB(0.2, 0.5, 0.7)
    c.setLineWidth(3)
    c.rect(1*inch, 1*inch, width-2*inch, height-2*inch)

    # Title
    c.setFont("Helvetica-Bold", 36)
    c.drawCentredString(width/2, height-3*inch, "Certificate of Completion")

    # Logo/Company Name
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height-2*inch, "LearnPro Elite")

    # Certificate text
    c.setFont("Helvetica", 16)
    c.drawCentredString(width/2, height-4*inch, "This is to certify that")
    
    # Recipient name
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height-4.75*inch, name)
    
    # Course completion text
    c.setFont("Helvetica", 16)
    c.drawCentredString(width/2, height-5.5*inch, 
                       f"has successfully completed the course")
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width/2, height-6*inch, course)

    # Date and certificate number
    c.setFont("Helvetica", 12)
    c.drawString(2*inch, 2*inch, f"Date: {completion_date}")
    c.drawString(width-4*inch, 2*inch, f"Certificate #: {certificate_number}")

    c.save()
    buffer.seek(0)
    return buffer

def authenticate_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
    return cursor.fetchone()

def generate_assessment_questions(subject, topic):
    """Generate 5 questions for a given topic"""
    prompt = f"""
    Create 5 assessment questions for the subject '{subject}', topic '{topic}'
    
    Format as JSON:
    {{
        "questions": [
            {{
                "question": "<question text>",
                "type": "open_ended",
                "expected_concepts": ["concept1", "concept2"]
            }}
        ]
    }}
    """
    
    response = llm.invoke(prompt)
    try:
        if response:  # 检查响应是否存在
            content_str = response.split("```json")[-1].split("```")[0].strip()
            return json.loads(content_str)
    except:
        # Fallback questions if generation fails
        return {
            "questions": [
                {
                    "question": f"Question {i+1}: Explain {topic} concept {i+1}",
                    "type": "open_ended",
                    "expected_concepts": ["understanding"]
                } for i in range(5)
            ]
        }

def check_certificate_eligibility(path_id, user_id):
    """Check if user is eligible for certificate"""
    # Check progress
    cursor.execute("""
        SELECT progress FROM learning_paths 
        WHERE id = %s AND user_id = %s
    """, (path_id, user_id))
    path = cursor.fetchone()
    
    if not path or path['progress'] < 1.0:
        return False, "Complete the course progress to 100%"
        
    # Check number of completed assessments
    cursor.execute("""
        SELECT COUNT(*) as count 
        FROM path_assessments 
        WHERE learning_path_id = %s AND user_id = %s
    """, (path_id, user_id))
    assessment_count = cursor.fetchone()['count']
    
    if assessment_count < 5:
        return False, f"Complete all 5 assessments ({assessment_count}/5 completed)"
        
    return True, "Eligible for certificate"

def evaluate_answer(topic, question, user_answer):
    prompt = f"""
    Evaluate this answer for the topic '{topic}':
    
    Question: {question}
    User's Answer: {user_answer}
    
    Provide evaluation in JSON format:
    {{
        "score": <float between 0 and 1>,
        "feedback": "<detailed feedback>",
        "correct_answer": "<explanation of the correct approach>"
    }}
    """
    
    response = llm.invoke(prompt)
    try:
        if response:  # 检查响应是否存在
            content_str = response.split("```json")[-1].split("```")[0].strip()
            return json.loads(content_str)
    except:
        return {
            "score": 0.5,
            "feedback": "Unable to evaluate answer",
            "correct_answer": "Please try again"
        }

def generate_assessment(topic):
    prompt = f"""
    Create an assessment question for the topic: {topic}
    
    Format as JSON:
    {{
        "question": "<question text>",
        "type": "open_ended",
        "expected_concepts": ["concept1", "concept2"]
    }}
    """
    
    response = llm.invoke(prompt)
    try:
        if response:  # 检查响应是否存在
            content_str = response.split("```json")[-1].split("```")[0].strip()
            return json.loads(content_str)
    except:
        return {
            "question": f"Explain the key concepts of {topic}",
            "type": "open_ended",
            "expected_concepts": ["basic understanding", "application"]
        }

def generate_learning_path(subject, user_interests, learning_style, target_days):
    prompt = f"""
    Create a detailed {target_days}-day learning path for {subject} considering:
    - User interests: {user_interests}
    - Learning style: {learning_style}
    
    Generate a learning path with educational resources. Each resource must have a title, 
    type (video/article/exercise/course), and description. URLs and platforms are optional.
    
    Format as JSON:
    {{
        "topics": [
            {{
                "name": "<topic name>",
                "description": "<detailed topic description>",
                "duration_days": <number>,
                "resources": [
                    {{
                        "title": "<specific course/resource title>",
                        "type": "<video/article/exercise/course>",
                        "description": "<detailed description>",
                        "url": "<resource URL or null>",
                        "platform": "<platform name or null>"
                    }}
                ],
                "practice_exercises": [
                    {{
                        "description": "<specific exercise description>",
                        "difficulty": "<beginner/intermediate/advanced>",
                        "url": "<exercise URL or null>"
                    }}
                ]
            }}
        ],
        "milestones": [
            {{
                "name": "<specific milestone name>",
                "expected_completion_day": <day_number>,
                "assessment_criteria": "<detailed criteria>"
            }}
        ]
    }}
    """
    
    response = llm.invoke(prompt)
    try:
        if response:  # 检查响应是否存在
            content_str = response.split("```json")[-1].split("```")[0].strip()
            path_content = json.loads(content_str)
            
            # Ensure all required fields exist and provide defaults if missing
            for topic in path_content['topics']:
                for resource in topic['resources']:
                    if 'url' not in resource:
                        resource['url'] = '#'
                    if 'platform' not in resource:
                        resource['platform'] = 'General'
                        
                for exercise in topic['practice_exercises']:
                    if 'url' not in exercise:
                        exercise['url'] = '#'
            
            return path_content
    except Exception as e:
        st.error(f"Failed to generate learning path: {str(e)}")
        return None


def extract_and_parse_json(response_str):
    try:
        json_start = response_str.find("```json") + len("```json")
        json_end = response_str.rfind("```")
        json_content = response_str[json_start:json_end].strip()
        return json.loads(json_content)
    
    except (ValueError, json.JSONDecodeError) as e:
        st.error(f"fail to extract and parse JSON: {str(e)}")
        return None
    except Exception as e:
        st.error(f"fail to extract and parse JSON: {str(e)}")
        return None


def display_learning_path(path):
    if not path:
        return
        
    st.subheader(f"{path['subject']} - {path['difficulty_level']}")
    
    # Progress and timeline
    col1, col2 = st.columns(2)
    with col1:
        st.progress(path['progress'])
        st.write(f"Progress: {path['progress']*100:.1f}%")
    with col2:
        days_remaining = (path['target_completion_date'] - datetime.now().date()).days
        st.write(f"Target completion: {path['target_completion_date']}")
        st.write(f"Days remaining: {days_remaining}")
    
    try:
        content = json.loads(path['content'])
        content = extract_and_parse_json(content.get('response', []))
        topics = content.get('topics', [])
        if topics:
            st.subheader("📚 Learning Topics")
            # Generate topic tabs
            topic_tabs = st.tabs([f"📌 {topic.get('name', 'Unnamed Topic')}" for topic in topics])
            
            for i, topic in enumerate(topics):
                with topic_tabs[i]:
                    # Topic basic information
                    st.write(f"**Duration:** {topic.get('duration_days', 'N/A')} days")
                    st.markdown(f"**Description:** {topic.get('description', 'No description available')}")
                    
                    # Display resources (matching your resource display logic)
                    st.subheader("📚 Resources")
                    resources = topic.get('resources', [])
                    if resources:
                        for resource in resources:
                            if not isinstance(resource, dict):
                                continue
                            with st.expander(f"{resource.get('title', 'Untitled')} ({resource.get('type', 'Resource')})"):
                                st.markdown(f"**Description:** {resource.get('description', 'No description')}")
                                st.markdown(f"**Platform:** {resource.get('platform', 'N/A')}")
                                if resource.get('url') and resource['url'] not in [None, '#']:
                                    st.link_button("Open Resource", resource['url'])
                    else:
                        st.info("No resources available")
                    
                    # Display practice exercises (matching your exercise display logic)
                    st.subheader("💪 Practice Exercises")
                    exercises = topic.get('practice_exercises', [])
                    if exercises:
                        for exercise in exercises:
                            if not isinstance(exercise, dict):
                                continue
                            with st.expander(f"{exercise.get('difficulty', 'General').title()} Level Exercise"):
                                st.markdown(f"**Description:** {exercise.get('description', 'No description')}")
                                if exercise.get('url') and exercise['url'] not in [None, '#']:
                                    st.link_button("Start Exercise", exercise['url'])
                    else:
                        st.info("No practice exercises available")
                    
                    # Assessment button (matching your assessment logic)
                    if st.button(f"Take Assessment: {topic.get('name', 'Unnamed Topic')}",
                            key=f"assess_{i}"):
                        st.session_state.current_assessment = {
                            'topic': topic.get('name', 'General Topic'),
                            'path_id': i  # Replace with actual path_id in production
                        }
                        st.success("Assessment started!")

        # Milestones section
        milestones = content.get('milestones', [])
        if milestones:
            st.subheader("🎯 Milestones")
            for milestone in milestones:
                # Ensure milestone is a dictionary
                if not isinstance(milestone, dict):
                    continue
                    
                with st.expander(
                    f"Day {milestone.get('expected_completion_day', 'N/A')}: {milestone.get('name', 'Unnamed Milestone')}"
                ):
                    st.markdown(
                        f"**Assessment Criteria:** {milestone.get('assessment_criteria', 'No criteria specified')}"
                    )
                    
        # Progress update section
        st.subheader("📈 Update Progress")
        new_progress = st.slider(
            "Update your progress",
            0.0,
            1.0,
            path['progress'],
            key=f"slider_{path['id']}"
        )
        
        if st.button("Save Progress", key=f"save_{path['id']}"):
            cursor.execute("""
                UPDATE learning_paths 
                SET progress = %s, last_updated = NOW()
                WHERE id = %s
            """, (new_progress, path['id']))
            db.commit()
            st.success("Progress updated successfully!")
        
        st.subheader("📝 Assessments")
    
    # Get completed assessments count
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM path_assessments 
            WHERE learning_path_id = %s AND user_id = %s
        """, (path['id'], st.session_state.user['id']))
        completed_assessments = cursor.fetchone()['count']
        
        st.write(f"Completed Assessments: {completed_assessments}/5")
        
        try:
            content = json.loads(path['content'])
            if completed_assessments < 5:
                topic = content['topics'][0]['name']  # Get first topic or you can let user select
                if st.button("Take Assessment", key=f"take_assessment_{path['id']}"):
                    questions = generate_assessment_questions(path['subject'], topic)
                    st.session_state.current_assessment = {
                        'path_id': path['id'],
                        'subject': path['subject'],
                        'topic': topic,
                        'questions': questions,
                        'current_question': 0
                    }
                    st.rerun()
        except Exception as e:
            st.error(f"Error loading assessment: {str(e)}")
        
        # Certificate section
        if path['progress'] >= 1.0:
            eligible, message = check_certificate_eligibility(path['id'], st.session_state.user['id'])
            
            if eligible:
                st.success("🎉 Congratulations! You've completed the course and all assessments!")
                cursor.execute("""
                SELECT * FROM certifications 
                WHERE user_id = %s AND learning_path_id = %s
            """, (st.session_state.user['id'], path['id']))
                existing_cert = cursor.fetchone()
                
                if existing_cert:
                    st.success("Certificate already generated!")
                    cert_buffer = generate_certificate(
                        existing_cert['recipient_name'],
                        path['subject'],
                        existing_cert['completion_date'].strftime('%B %d, %Y'),
                        existing_cert['certificate_number']
                    )
                    st.download_button(
                        label="Download Certificate",
                        data=cert_buffer,
                        file_name=f"certificate_{path['subject'].lower().replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
                else:
                # Container for certificate generation
                    cert_container = st.container()
                    
                    # Form for certificate details
                    with st.form(key=f"cert_form_{path['id']}"):
                        recipient_name = st.text_input("Enter your full name as it should appear on the certificate")
                        completion_date = st.date_input("Completion Date", datetime.now().date())
                        submitted = st.form_submit_button("Generate Certificate")
                        
                        if submitted and recipient_name:
                            # Generate unique certificate number
                            cert_number = f"ALA-{path['id']}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                            
                            # Save certification record
                            cursor.execute("""
                                INSERT INTO certifications 
                                (user_id, learning_path_id, completion_date, certificate_number, recipient_name)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (
                                st.session_state.user['id'],
                                path['id'],
                                completion_date,
                                cert_number,
                                recipient_name
                            ))
                            db.commit()
                            
                            # Store certificate data for download
                            st.session_state[f"cert_data_{path['id']}"] = {
                                "name": recipient_name,
                                "course": path['subject'],
                                "date": completion_date.strftime('%B %d, %Y'),
                                "number": cert_number
                            }
                            st.rerun()
                    
                    # Show download button if certificate was just generated
                    cert_data_key = f"cert_data_{path['id']}"
                    if cert_data_key in st.session_state:
                        cert_data = st.session_state[cert_data_key]
                        cert_buffer = generate_certificate(
                            cert_data["name"],
                            cert_data["course"],
                            cert_data["date"],
                            cert_data["number"]
                        )
                        
                        st.success("Certificate generated successfully!")
                        st.download_button(
                            label="Download Certificate",
                            data=cert_buffer,
                            file_name=f"certificate_{path['subject'].lower().replace(' ', '_')}.pdf",
                            mime="application/pdf"
                        )
                        # Clean up session state
                        del st.session_state[cert_data_key]
            else:
                st.warning(message)
    except Exception as e:
        st.error(f"Error displaying learning path: {str(e)}")
        # Print the full error for debugging
        import traceback
        st.error(traceback.format_exc())

def display_assessment_tab():
    if 'current_assessment' in st.session_state and st.session_state.current_assessment:
        assessment = st.session_state.current_assessment
        
        # Get subject from learning path if not present in assessment
        if 'subject' not in assessment:
            cursor.execute("""
                SELECT subject FROM learning_paths 
                WHERE id = %s
            """, (assessment['path_id'],))
            result = cursor.fetchone()
            if result:
                assessment['subject'] = result['subject']
            else:
                st.error("Could not find associated learning path")
                st.session_state.current_assessment = None
                return
        
        # Validate questions data structure
        if 'questions' not in assessment or not assessment['questions'].get('questions'):
            try:
                questions_data = generate_assessment_questions(assessment['subject'], assessment['topic'])
                if not questions_data or 'questions' not in questions_data:
                    raise ValueError("Invalid question data structure")
                assessment['questions'] = questions_data
                assessment['current_question'] = 0
                st.session_state.current_assessment = assessment
            except Exception as e:
                st.error(f"Error generating questions: {str(e)}")
                st.session_state.current_assessment = None
                return
        
        # Access questions safely
        questions = assessment['questions'].get('questions', [])
        if not questions:
            st.error("No questions available")
            st.session_state.current_assessment = None
            return
            
        current_q = assessment.get('current_question', 0)
        if current_q >= len(questions):
            st.error("Assessment completed")
            st.session_state.current_assessment = None
            return
        
        st.subheader(f"Assessment for: {assessment['subject']} - {assessment['topic']}")
        st.write(f"Question {current_q + 1}/{len(questions)}")
        
        # Safely access question data
        current_question = questions[current_q]
        if isinstance(current_question, dict) and 'question' in current_question:
            st.write(current_question['question'])
        else:
            st.error("Invalid question format")
            st.session_state.current_assessment = None
            return
        
        # Add unique key for the text area
        answer_key = f"assessment_answer_{assessment['path_id']}_{current_q}"
        user_answer = st.text_area("Your Answer", key=answer_key)
        
        # Add unique key for the submit button
        submit_key = f"assessment_submit_{assessment['path_id']}_{current_q}"
        if st.button("Submit Answer", key=submit_key):
            try:
                evaluation = evaluate_answer(
                    assessment['topic'],
                    current_question['question'],
                    user_answer
                )
                
                cursor.execute("""
                    INSERT INTO path_assessments 
                    (learning_path_id, user_id, question, user_answer, score, feedback)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    assessment['path_id'],
                    st.session_state.user['id'],
                    current_question['question'],
                    user_answer,
                    evaluation['score'],
                    evaluation['feedback']
                ))
                db.commit()
                
                st.write("Score:", f"{evaluation['score']*100:.1f}%")
                st.write("Feedback:", evaluation['feedback'])
                
                if current_q < len(questions) - 1:
                    st.session_state.current_assessment['current_question'] = current_q + 1
                    time.sleep(2)
                    st.rerun()
                else:
                    st.success("Assessment completed!")
                    st.session_state.current_assessment = None
                    time.sleep(2)
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing answer: {str(e)}")
                
    # Display past assessments if no current assessment
    else:
        cursor.execute("""
            SELECT pa.*, lp.subject 
            FROM path_assessments pa
            JOIN learning_paths lp ON pa.learning_path_id = lp.id
            WHERE pa.user_id = %s 
            ORDER BY pa.completed_at DESC
        """, (st.session_state.user['id'],))
        past_assessments = cursor.fetchall()
        
        if past_assessments:
            st.subheader("Past Assessments")
            for assessment in past_assessments:
                with st.expander(f"{assessment['subject']} - {assessment['completed_at']}"):
                    st.write("Question:", assessment['question'])
                    st.write("Your Answer:", assessment['user_answer'])
                    st.write(f"Score: {assessment['score']*100:.1f}%")
                    st.write("Feedback:", assessment['feedback'])
        else:
            st.info("No assessments taken yet. Start a course to take assessments!")

def display_task_assessment_tab():
    st.header("Task Assessments")
    
    if st.session_state.current_assessment1:
        assessment = st.session_state.current_assessment1
        st.subheader(f"Assessment for: {assessment['topic']}")
        st.write(assessment['question'])
        
        # Add unique key for the task assessment text area
        task_answer_key = f"task_assessment_answer_{assessment['path_id']}"
        user_answer = st.text_area("Your Answer", key=task_answer_key)
        
        # Add unique key for the task assessment submit button
        task_submit_key = f"task_assessment_submit_{assessment['path_id']}"
        if st.button("Submit Assessment", key=task_submit_key):
            evaluation = evaluate_answer(assessment['topic'], 
                                      assessment['question'], 
                                      user_answer)
            
            cursor.execute("""
                INSERT INTO assessments 
                (user_id, subject, topic, score, feedback)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                st.session_state.user['id'],
                assessment['topic'],
                assessment['topic'],
                evaluation['score'],
                evaluation['feedback']
            ))
            db.commit()
            
            st.write("Score:", f"{evaluation['score']*100:.1f}%")
            st.write("Feedback:", evaluation['feedback'])
            st.write("Correct Approach:", evaluation['correct_answer'])
            
            # Add unique key for the clear assessment button
            clear_key = f"clear_assessment_{assessment['path_id']}"
            if st.button("Clear Assessment", key=clear_key):
                st.session_state.current_assessment1 = None
                st.rerun()
    
    # Display past assessments
    cursor.execute("""
        SELECT * FROM assessments 
        WHERE user_id = %s 
        ORDER BY taken_at DESC
    """, (st.session_state.user['id'],))
    past_assessments = cursor.fetchall()
    
    if past_assessments:
        st.subheader("Past Assessments")
        for idx, assessment in enumerate(past_assessments):
            # Add unique key for each expander
            expander_key = f"past_assessment_{assessment['id']}_{idx}"
            with st.expander(f"{assessment['subject']} - {assessment['taken_at']}", 
                           key=expander_key):
                st.write(f"Score: {assessment['score']*100:.1f}%")
                st.write("Feedback:", assessment['feedback'])
                
def display_analytics(progress_data, user_id):
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(progress_data)
    
    # Overall Progress Section
    st.subheader("📊 Overall Learning Progress")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Subjects", len(df))
    with col2:
        avg_progress = df['progress'].mean() * 100
        st.metric("Average Progress", f"{avg_progress:.1f}%")
    with col3:
        if not df.empty:
            top_subject = df.loc[df['progress'].idxmax(), 'subject']
            top_progress = df['progress'].max() * 100
            st.metric("Best Performing Subject", f"{top_subject} ({top_progress:.1f}%)")

    # Progress Heatmap
    if not df.empty:
        st.subheader("🔥 Progress Intensity")
        df['day_of_week'] = pd.to_datetime(df['last_updated']).dt.strftime('%A')
        df['hour_of_day'] = pd.to_datetime(df['last_updated']).dt.hour
        
        fig_heatmap = px.density_heatmap(
            df,
            x='day_of_week',
            y='hour_of_day',
            title="Learning Activity Patterns",
            labels={'day_of_week': 'Day of Week', 'hour_of_day': 'Hour of Day'}
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # Progress Over Time
    st.subheader("📈 Progress Timeline")
    # Get assessment data
    cursor.execute("""
        SELECT subject, score, taken_at 
        FROM assessments 
        WHERE user_id = %s
        ORDER BY taken_at
    """, (user_id,))
    assessment_data = cursor.fetchall()
    assessment_df = pd.DataFrame(assessment_data)

    if not df.empty:
        # Create timeline of progress updates
        fig_timeline = px.line(
            df,
            x='last_updated',
            y='progress',
            color='subject',
            title="Learning Progress Over Time",
            labels={
                'last_updated': 'Date',
                'progress': 'Progress (%)',
                'subject': 'Subject'
            }
        )
        
        # Update progress values to percentage
        fig_timeline.update_traces(y=df['progress'] * 100)
        
        # Add assessment scores if available
        if not assessment_df.empty:
            assessment_scatter = px.scatter(
                assessment_df,
                x='taken_at',
                y='score',
                title="Assessment Scores",
                labels={
                    'taken_at': 'Date',
                    'score': 'Score (%)'
                }
            )
            # Update score values to percentage
            assessment_scatter.update_traces(y=assessment_df['score'] * 100, mode='markers', 
                                          marker=dict(symbol='star', size=12))
            
            for trace in assessment_scatter.data:
                fig_timeline.add_trace(trace)

        fig_timeline.update_layout(
            hovermode='x unified',
            yaxis_title="Progress/Score (%)"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    # Learning Pace Analysis
    st.subheader("⚡ Learning Pace Analysis")
    if not df.empty:
        df['days_since_start'] = (pd.to_datetime(df['last_updated']) - 
                                 pd.to_datetime(df['last_updated']).min()).dt.days
        
        pace_fig = px.scatter(
            df,
            x='days_since_start',
            y='progress',
            color='subject',
            size='progress',
            trendline="ols",
            title="Learning Pace by Subject",
            labels={
                'days_since_start': 'Days Since Starting',
                'progress': 'Progress (%)',
                'subject': 'Subject'
            }
        )
        # Update progress values to percentage
        pace_fig.update_traces(y=df['progress'] * 100)
        pace_fig.update_layout(yaxis_title="Progress (%)")
        st.plotly_chart(pace_fig, use_container_width=True)

    # Assessment Performance
    if not assessment_df.empty:
        st.subheader("📝 Assessment Performance")
        
        # Average scores by subject
        avg_scores = assessment_df.groupby('subject')['score'].agg(['mean', 'count']).reset_index()
        avg_scores['mean'] = avg_scores['mean'] * 100
        
        fig_scores = px.bar(
            avg_scores,
            x='subject',
            y='mean',
            color='count',
            title="Average Assessment Scores by Subject",
            labels={
                'subject': 'Subject',
                'mean': 'Average Score (%)',
                'count': 'Number of Assessments'
            }
        )
        st.plotly_chart(fig_scores, use_container_width=True)

    # Time Management
    st.subheader("⏰ Time Management Insights")
    if not df.empty:
        df['hour'] = pd.to_datetime(df['last_updated']).dt.hour
        df['day'] = pd.to_datetime(df['last_updated']).dt.day_name()
        
        col1, col2 = st.columns(2)
        with col1:
            # Most productive hours
            productive_hours = df.groupby('hour')['progress'].mean().sort_values(ascending=False)
            st.write("Most Productive Hours:")
            for hour, progress in productive_hours.head(3).items():
                st.write(f"• {hour:02d}:00 - {progress*100:.1f}% average progress")
        
        with col2:
            # Most active days
            active_days = df.groupby('day')['progress'].mean().sort_values(ascending=False)
            st.write("Most Active Days:")
            for day, progress in active_days.head(3).items():
                st.write(f"• {day} - {progress*100:.1f}% average progress")

    # Recommendations
    st.subheader("💡 Personalized Recommendations")
    if not df.empty:
        # Generate recommendations based on analysis
        low_progress_subjects = df[df['progress'] < 0.5]['subject'].tolist()
        inactive_subjects = df[pd.to_datetime(df['last_updated']) < 
                             (datetime.now() - timedelta(days=7))]['subject'].tolist()
        
        if low_progress_subjects:
            st.write("Subjects needing attention:")
            for subject in low_progress_subjects:
                st.write(f"• Focus more on {subject}")
        
        if inactive_subjects:
            st.write("Subjects to revisit:")
            for subject in inactive_subjects:
                st.write(f"• Resume learning {subject}")

        # Best performing times
        if not productive_hours.empty and not active_days.empty:
            best_hour = productive_hours.index[0]
            best_day = active_days.index[0]
            st.write(f"💪 You perform best on {best_day}s at {best_hour:02d}:00")

    st.subheader("🎓 Certifications Earned")
    cursor.execute("""
        SELECT c.*, lp.subject 
        FROM certifications c
        JOIN learning_paths lp ON c.learning_path_id = lp.id
        WHERE c.user_id = %s
        ORDER BY c.completion_date DESC
    """, (user_id,))
    certifications = cursor.fetchall()
    
    if certifications:
        cert_df = pd.DataFrame(certifications)
        st.metric("Total Certifications", len(certifications))
        
        # Display certificates table
        st.dataframe(
            cert_df[['subject', 'completion_date', 'certificate_number', 'recipient_name']]
                .rename(columns={
                    'subject': 'Course',
                    'completion_date': 'Completion Date',
                    'certificate_number': 'Certificate Number',
                    'recipient_name': 'Recipient Name'
                })
        )
    else:
        st.info("Complete courses to earn certificates!")

def main():
    st.set_page_config(page_title="AI Learning Assistant", layout="wide")
    
    # Initialize session states
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'current_assessment' not in st.session_state:
        st.session_state.current_assessment = None
    if 'current_assessment1' not in st.session_state:
        st.session_state.current_assessment1 = None
    
    # Sidebar
    with st.sidebar:
        st.title("🎓 AI Learning Assistant")
        if st.session_state.user:
            st.write(f"Welcome, {st.session_state.user['username']}!")
            if st.button("Logout"):
                st.session_state.user = None
                st.rerun()
    
    # Login/Register page
    if not st.session_state.user:
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                user = authenticate_user(username, password)
                if user:
                    st.session_state.user = user
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        with col2:
            st.header("Register")
            new_username = st.text_input("Choose Username", key="reg_username")
            new_password = st.text_input("Choose Password", type="password", key="reg_password")
            interests = st.text_area("Your Interests (comma-separated)")
            learning_style = st.selectbox("Preferred Learning Style", 
                                        ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"])
            
            if st.button("Register"):
                try:
                    cursor.execute("""
                        INSERT INTO users (username, password, interests, learning_style)
                        VALUES (%s, %s, %s, %s)
                    """, (new_username, new_password, interests, learning_style))
                    db.commit()
                    st.success("Registration successful! Please login.")
                except mysql.connector.Error as err:
                    st.error(f"Registration failed: {err}")
    
    # Main application interface
    else:
        tabs = st.tabs(["Learning Dashboard", "Create New Path", "Progress Analytics", "Assessments", "Task Assessments"])
        
        # Dashboard Tab
        with tabs[0]:
            st.header("Your Learning Dashboard")
            
            cursor.execute("""
                SELECT * FROM learning_paths 
                WHERE user_id = %s
                ORDER BY last_updated DESC
            """, (st.session_state.user['id'],))
            paths = cursor.fetchall()
            
            if paths:
                for path in paths:
                    display_learning_path(path)
            else:
                st.info("No learning paths yet. Create one in the 'Create New Path' tab!")
        
        # Create New Path Tab
        with tabs[1]:
            st.header("Create New Learning Path")
            
            col1, col2 = st.columns(2)
            with col1:
                subject = st.text_input("Subject you want to learn")
                difficulty = st.select_slider("Difficulty Level", 
                                           options=["Beginner", "Intermediate", "Advanced"])
                target_days = st.number_input("Target completion (days)", min_value=1, value=30)
            
            with col2:
                st.write("Current Settings:")
                st.write(f"Learning Style: {st.session_state.user['learning_style']}")
                st.write(f"Interests: {st.session_state.user['interests']}")
            
            if st.button("Generate Learning Path"):
                with st.spinner("Generating personalized learning path..."):
                    target_date = datetime.now().date() + timedelta(days=target_days)
                    path_content = generate_learning_path(
                        subject,
                        st.session_state.user['interests'],
                        st.session_state.user['learning_style'],
                        target_days
                    )
                    
                    cursor.execute("""
                        INSERT INTO learning_paths 
                        (user_id, subject, progress, difficulty_level, content, target_completion_date)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        st.session_state.user['id'],
                        subject,
                        0.0,
                        difficulty,
                        json.dumps(path_content),
                        target_date
                    ))
                    db.commit()
                    
                    st.success("Learning path created successfully!")
                    st.rerun()
        
        with tabs[2]:
            st.header("Progress Analytics")
            
            cursor.execute("""
                SELECT subject, progress, last_updated 
                FROM learning_paths 
                WHERE user_id = %s
            """, (st.session_state.user['id'],))
            progress_data = cursor.fetchall()
            
            if progress_data:
                display_analytics(progress_data, st.session_state.user['id'])
            else:
                st.info("Start learning to see your progress analytics!")
        # Assessment Tab
        with tabs[3]:
            st.header("Assessments")
            
            if st.session_state.get('current_assessment'):
                display_assessment_tab()
            else:
                # Display past assessments
                cursor.execute("""
                    SELECT pa.*, lp.subject 
                    FROM path_assessments pa
                    JOIN learning_paths lp ON pa.learning_path_id = lp.id
                    WHERE pa.user_id = %s 
                    ORDER BY pa.completed_at DESC
                """, (st.session_state.user['id'],))
                past_assessments = cursor.fetchall()
                
                if past_assessments:
                    st.subheader("Past Assessments")
                    for assessment in past_assessments:
                        with st.expander(f"{assessment['subject']} - {assessment['completed_at']}"):
                            st.write("Question:", assessment['question'])
                            st.write("Your Answer:", assessment['user_answer'])
                            st.write(f"Score: {assessment['score']*100:.1f}%")
                            st.write("Feedback:", assessment['feedback'])
                else:
                    st.info("No assessments taken yet. Start a course to take assessments!")

        with tabs[4]:
            display_task_assessment_tab()

if __name__ == "__main__":
    init_db()
    main()
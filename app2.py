import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import sqlite3
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
from typing import Dict, List, Tuple, Optional
import base64
from io import BytesIO
from urllib.parse import urlencode
import os

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays"""
    def default(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return super().default(obj)
    
# =============================================================================
# PRODUCTION-READY CONFIGURATION
# =============================================================================

def get_api_key(key_name: str) -> str:
    """Get API key from Streamlit secrets or environment variables"""
    try:
        # Try Streamlit secrets first (for Streamlit Cloud)
        return st.secrets[key_name]
    except (KeyError, FileNotFoundError):
        # Fallback to environment variables (for local development)
        return os.getenv(key_name, "")    

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# API Configuration
FREESOUND_API_KEY = get_api_key("FREESOUND_API_KEY")
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")       
ANTHROPIC_API_KEY = get_api_key("ANTHROPIC_API_KEY")       

# Hearing test configuration
STANDARD_FREQUENCIES = [250, 500, 1000, 2000, 4000, 8000]  # Hz
FREQUENCY_NAMES = ["250Hz", "500Hz", "1kHz", "2kHz", "4kHz", "8kHz"]
VOLUME_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80]  # dB HL
TONE_DURATION = 2.0  # seconds

# Gamification settings
POINTS_PER_CORRECT = 10
BONUS_MULTIPLIER = 1.5
ACHIEVEMENT_THRESHOLDS = {
    'first_test': 1,
    'consistent_user': 3,
    'hearing_champion': 10,
    'perfect_score': 1
}

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'user_profile': {},
        'current_test': {},
        'test_history': [],
        'current_page': 'onboarding',
        'test_results': [],
        'hearing_data': np.zeros((6, 8)),  # 6 frequencies x 8 volume levels
        'current_test_step': 0,
        'test_started': False,
        'current_frequency_idx': 0,
        'current_volume_idx': 0,
        'current_ear': 'left',
        'gamification_data': {
            'total_points': 0,
            'achievements': [],
            'streak': 0,
            'level': 1
        },
        'adaptive_challenge': {
            'difficulty': 1,
            'score': 0,
            'lives': 3
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =============================================================================
# AUDIO GENERATION AND FREESOUND INTEGRATION
# =============================================================================

def generate_tone(frequency: float, duration: float = 2.0, volume: float = 0.5, 
                 sample_rate: int = 44100) -> str:
    """
    Generate a pure tone and return as base64 audio data
    
    Args:
        frequency: Tone frequency in Hz
        duration: Duration in seconds
        volume: Volume level (0.0 to 1.0)
        sample_rate: Audio sample rate
    
    Returns:
        Base64 encoded WAV audio data
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Generate sine wave with fade in/out to avoid clicks
    tone = np.sin(2 * np.pi * frequency * t) * volume
    
    # Apply fade in/out (10ms each)
    fade_samples = int(0.01 * sample_rate)
    if len(tone) > 2 * fade_samples:
        tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
        tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # Convert to 16-bit PCM
    audio_data = (tone * 32767).astype(np.int16)
    
    # Create WAV file in memory
    buffer = BytesIO()
    # WAV header
    buffer.write(b'RIFF')
    buffer.write((36 + len(audio_data) * 2).to_bytes(4, 'little'))
    buffer.write(b'WAVE')
    buffer.write(b'fmt ')
    buffer.write((16).to_bytes(4, 'little'))
    buffer.write((1).to_bytes(2, 'little'))  # PCM format
    buffer.write((1).to_bytes(2, 'little'))  # Mono
    buffer.write(sample_rate.to_bytes(4, 'little'))
    buffer.write((sample_rate * 2).to_bytes(4, 'little'))
    buffer.write((2).to_bytes(2, 'little'))
    buffer.write((16).to_bytes(2, 'little'))
    buffer.write(b'data')
    buffer.write((len(audio_data) * 2).to_bytes(4, 'little'))
    buffer.write(audio_data.tobytes())
    
    # Encode as base64
    audio_b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:audio/wav;base64,{audio_b64}"

class AudioTestManager:
    """Enhanced AudioTestManager with proper FreeSound integration"""
    
    def __init__(self):
        self.frequencies = STANDARD_FREQUENCIES
        self.volume_levels = VOLUME_LEVELS
        self.frequency_names = FREQUENCY_NAMES
        self.freesound_api_key = st.secrets.get("FREESOUND_API_KEY", "")
        
    def get_test_audio(self, frequency: int, volume: int) -> str:
        """Generate test audio for given frequency and volume"""
        # Convert dB to linear scale (simplified)
        volume_linear = min(1.0, volume / 80.0)
        return generate_tone(frequency, TONE_DURATION, volume_linear)
    
    def fetch_freesound_audio(self, query: str, duration: int = 2) -> Optional[Dict]:
        """
        Fetch audio from FreeSound API with enhanced error handling
        
        Args:
            query: Search query for sounds
            duration: Maximum duration in seconds
            
        Returns:
            Dictionary with audio info or None if failed
        """
        if not self.freesound_api_key:
            st.warning("🔑 FreeSound API key not configured. Using generated tones instead.")
            return None
        
        try:
            # Search for sounds
            search_url = "https://freesound.org/apiv2/search/text/"
            headers = {"Authorization": f"Token {self.freesound_api_key}"}
            params = {
                "query": query,
                "filter": f"duration:[0.5 TO {duration}] AND type:wav",
                "fields": "id,name,url,previews,duration,samplerate",
                "page_size": 10,
                "sort": "score"
            }
            
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    # Return the best match
                    best_sound = data["results"][0]
                    return {
                        "id": best_sound["id"],
                        "name": best_sound["name"],
                        "preview_url": best_sound["previews"]["preview-lq-mp3"],
                        "duration": best_sound.get("duration", duration),
                        "sample_rate": best_sound.get("samplerate", 44100)
                    }
                else:
                    st.info(f"No sounds found for query: {query}")
                    return None
            elif response.status_code == 401:
                st.error("🔑 Invalid FreeSound API key. Please check your configuration.")
                return None
            elif response.status_code == 429:
                st.warning("⏳ FreeSound API rate limit reached. Please try again later.")
                return None
            else:
                st.warning(f"FreeSound API error: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            st.warning("⏰ FreeSound API request timed out. Using generated audio.")
            return None
        except requests.exceptions.ConnectionError:
            st.warning("🌐 Unable to connect to FreeSound API. Check your internet connection.")
            return None
        except Exception as e:
            st.error(f"Error fetching audio from FreeSound: {str(e)}")
            return None
    def get_calibration_audio(self) -> str:
        """Get calibration audio, preferring FreeSound if available"""
        freesound_audio = self.fetch_freesound_audio("1000hz tone calibration", 3)
        
        if freesound_audio:
            st.success(f"🎵 Using FreeSound audio: {freesound_audio['name']}")
            return freesound_audio["preview_url"]
        else:
            # Fallback to generated tone
            return generate_tone(1000, 3.0, 0.4)
    
    def get_challenge_audio(self, frequency: int, volume: float) -> str:
        """Get audio for challenges, with FreeSound integration for variety"""
        # Try to get interesting sounds from FreeSound for challenges
        sound_queries = [
            f"{frequency}hz sine wave",
            f"pure tone {frequency}",
            f"test tone {frequency}hz",
            "hearing test tone"
        ]
        
        for query in sound_queries:
            freesound_audio = self.fetch_freesound_audio(query, 2)
            if freesound_audio:
                return freesound_audio["preview_url"]
        
        # Fallback to generated tone
        return generate_tone(frequency, 2.0, volume)    
# =============================================================================
# AI INTEGRATION MODULE
# =============================================================================

class AIIntegration:
    """Handles AI API integrations for personalized feedback"""
    
    def call_openai_api(self, prompt: str) -> str:
        """Call OpenAI ChatGPT API for hearing advice"""
        # Check if API key is properly configured
        if not OPENAI_API_KEY or OPENAI_API_KEY == "" or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
            return self.generate_mock_advice(prompt)
        
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful audiologist assistant providing hearing health advice."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                   headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                st.warning(f"OpenAI API error: {response.status_code}")
                return self.generate_mock_advice(prompt)
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
            return self.generate_mock_advice(prompt)
    
    def call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic Claude API for hearing analysis"""
        # Check if API key is properly configured
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "" or ANTHROPIC_API_KEY == "YOUR_ANTHROPIC_API_KEY_HERE":
            return self.generate_mock_analysis(prompt)
        
        try:
            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 300,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = requests.post("https://api.anthropic.com/v1/messages", 
                                   headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                return response.json()["content"][0]["text"]
            else:
                st.warning(f"Anthropic API error: {response.status_code}")
                return self.generate_mock_analysis(prompt)
        except Exception as e:
            st.error(f"Error calling Anthropic API: {e}")
            return self.generate_mock_analysis(prompt)
        
    def generate_mock_advice(self, prompt: str) -> str:
        """Generate mock advice when API is not available"""
        advice_templates = [
            "Based on your hearing test results, I recommend limiting exposure to loud environments and using ear protection when necessary. Consider the 60/60 rule for headphone use.",
            "Your hearing profile suggests maintaining good ear hygiene and taking regular breaks from noisy environments. Annual hearing checks are recommended.",
            "The results indicate normal hearing thresholds with some variation in higher frequencies. Continue protecting your hearing with proper ear protection in loud settings.",
            "Your test shows good hearing sensitivity overall. To maintain this, avoid prolonged exposure to sounds above 85 decibels and use hearing protection when needed.",
            "Regular hearing monitoring is beneficial. If you notice any changes in your hearing, consult with an audiologist for professional evaluation."
        ]
        return random.choice(advice_templates)
    
    def generate_mock_analysis(self, prompt: str) -> str:
        """Generate mock analysis when API is not available"""
        analysis_templates = [
            "Your hearing test demonstrates good sensitivity across most frequency ranges. The slight variations observed are within normal limits for your age group. Continue practicing hearing conservation.",
            "The assessment reveals healthy auditory function with normal thresholds. Your responses indicate good hearing acuity across the tested frequencies. Maintain protective habits.",
            "Results show typical hearing patterns with adequate sensitivity across speech-important frequencies. This suggests your auditory system is functioning well. Regular monitoring recommended.",
            "Your hearing profile indicates normal thresholds with good consistency across frequencies. The test results reflect healthy hearing function for your demographic.",
            "The evaluation demonstrates satisfactory hearing levels across the frequency spectrum. Your auditory processing appears intact with no significant concerns noted."
        ]
        return random.choice(analysis_templates)

# =============================================================================  
# ENHANCED DATABASE MANAGEMENT
# =============================================================================

class DatabaseManager:
    """Enhanced DatabaseManager with error handling for deployment"""
    
    def __init__(self):
        # Use a more specific database path for deployment
        self.db_path = "hearing_test_data.db"
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables with IF NOT EXISTS
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    test_date TEXT NOT NULL,
                    age INTEGER,
                    noise_exposure TEXT,
                    hearing_data TEXT,
                    ai_feedback TEXT,
                    recommendations TEXT,
                    game_score INTEGER DEFAULT 0,
                    test_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    name TEXT,
                    age INTEGER,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            st.error(f"Database initialization error: {e}")
            # Create an in-memory fallback
            self.db_path = ":memory:"
    
    def save_test_results(self, user_data: Dict):
        """Save test results to database with dynamic column handling"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
    
        try:
            # Get current table structure
            cursor.execute("PRAGMA table_info(test_results)")
            existing_columns = [row[1] for row in cursor.fetchall()]
        
            # Define the data we want to save
            data_to_save = {
                'user_id': user_data.get('user_id', 'anonymous'),
                'test_date': datetime.now().isoformat(),
                'age': user_data.get('age'),
                'noise_exposure': user_data.get('noise_exposure'),
                'hearing_data': json.dumps(user_data.get('hearing_data', [])),
                'ai_feedback': user_data.get('ai_feedback', ''),
                'recommendations': user_data.get('recommendations', ''),
                'game_score': user_data.get('game_score', 0),
                'test_summary': json.dumps(user_data.get("test_summary", {}), cls=NumpyJSONEncoder)
            }
        
            # Only include columns that exist in the table
            columns_to_insert = []
            values_to_insert = []
            placeholders = []
        
            for column, value in data_to_save.items():
                if column in existing_columns:
                    columns_to_insert.append(column)
                    values_to_insert.append(value)
                    placeholders.append('?')
        
            # Build dynamic INSERT query
            columns_str = ', '.join(columns_to_insert)
            placeholders_str = ', '.join(placeholders)
        
            query = f'''
                INSERT INTO test_results ({columns_str})
                VALUES ({placeholders_str})
            '''
        
            cursor.execute(query, values_to_insert)
            conn.commit()
        
            print(f"✅ Saved test results with columns: {columns_to_insert}")
        
        except sqlite3.Error as e:
            print(f"❌ Database error: {e}")
            print(f"Available columns: {existing_columns}")
            raise e
        finally:
            conn.close()
        
    def get_latest_test_result(self, user_id: str):
        """Get the most recent test result for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM test_results 
            WHERE user_id = ? 
            ORDER BY test_date DESC 
            LIMIT 1
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        return result
    
    def get_user_history(self, user_id: str = 'anonymous') -> pd.DataFrame:
        """Retrieve user's test history"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM test_results 
            WHERE user_id = ? 
            ORDER BY test_date DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()
        
        return df
# =============================================================================
# HEALTH CHECK FOR DEPLOYMENT MONITORING
# =============================================================================

def render_health_check():
    """Health check for deployment monitoring"""
    if st.sidebar.button("🔍 System Status"):
        st.sidebar.write("**System Status:**")
        
        # Check database
        try:
            db = DatabaseManager()
            st.sidebar.success("✅ Database: OK")
        except Exception as e:
            st.sidebar.error(f"❌ Database: {str(e)[:50]}...")
        
        # Check API keys
        if FREESOUND_API_KEY and FREESOUND_API_KEY!="":
            st.sidebar.success("✅ FreeSound API: Configured")
        else:
            st.sidebar.warning("⚠️ FreeSound API: Not configured")
            
        if OPENAI_API_KEY and OPENAI_API_KEY!="":
            st.sidebar.success("✅ OpenAI API: Configured")
        else:
            st.sidebar.info("ℹ️ OpenAI API: Optional")
            
        if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY!="" :
            st.sidebar.success("✅ Anthropic API: Configured")
        else:
            st.sidebar.info("ℹ️ Anthropic API: Optional")    

# =============================================================================
# GAMIFICATION MODULE
# =============================================================================

class HearingGame:
    """Implements gamification features with adaptive difficulty"""
    
    def __init__(self):
        self.difficulty_levels = {
            1: {"volume_diff": 15, "freq_diff": 1000, "num_options": 3},
            2: {"volume_diff": 10, "freq_diff": 500, "num_options": 4},
            3: {"volume_diff": 7, "freq_diff": 250, "num_options": 4},
            4: {"volume_diff": 5, "freq_diff": 125, "num_options": 5},
            5: {"volume_diff": 3, "freq_diff": 62, "num_options": 6}
        }
    
    def generate_sound_sequence_game(self, difficulty: int = 1) -> Dict:
        """Generate a sound identification game"""
        params = self.difficulty_levels.get(difficulty, self.difficulty_levels[1])
        
        # Generate sequence of sounds, one different
        base_freq = random.choice([500, 1000, 2000])
        base_volume = random.randint(30, 60)
        
        # Create the different sound
        freq_change = random.choice([-1, 1]) * params["freq_diff"]
        different_freq = max(250, min(8000, base_freq + freq_change))
        
        sequence_length = 4
        sequence = []
        different_position = random.randint(0, sequence_length - 1)
        
        for i in range(sequence_length):
            if i == different_position:
                sequence.append({"freq": different_freq, "volume": base_volume})
            else:
                sequence.append({"freq": base_freq, "volume": base_volume})
        
        return {
            "sequence": sequence,
            "different_position": different_position,
            "difficulty": difficulty,
            "base_freq": base_freq,
            "different_freq": different_freq
        }
    
    def adjust_difficulty(self, correct: bool, current_difficulty: int) -> int:
        """Adaptive difficulty adjustment"""
        if correct and current_difficulty < 5:
            return current_difficulty + 1
        elif not correct and current_difficulty > 1:
            return current_difficulty - 1
        return current_difficulty

def update_gamification_data(results_df: pd.DataFrame):
    """Update user's gamification data based on test results"""
    if results_df.empty:
        return
        
    # Calculate points based on performance
    total_tests = len(results_df)
    heard_count = len(results_df[results_df['heard'] == True])
    accuracy = heard_count / total_tests if total_tests > 0 else 0
    
    points = int(accuracy * 100 * POINTS_PER_CORRECT)
    
    # Bonus for perfect score
    if accuracy == 1.0:
        points = int(points * BONUS_MULTIPLIER)
        if 'perfect_score' not in st.session_state.gamification_data['achievements']:
            st.session_state.gamification_data['achievements'].append('perfect_score')
    
    # Update total points and level
    st.session_state.gamification_data['total_points'] += points
    st.session_state.gamification_data['level'] = (
        st.session_state.gamification_data['total_points'] // 100
    ) + 1
    
    # Achievement checks
    test_count = len(st.session_state.test_history)
    achievements = st.session_state.gamification_data['achievements']
    
    if test_count >= 1 and 'first_test' not in achievements:
        achievements.append('first_test')
    if test_count >= 3 and 'consistent_user' not in achievements:
        achievements.append('consistent_user')
    if test_count >= 10 and 'hearing_champion' not in achievements:
        achievements.append('hearing_champion')

# =============================================================================
# USER INTERFACE MODULES
# =============================================================================

def render_onboarding():
    """Render user onboarding interface"""
    st.title("🔊 ListenWell AI")
    st.markdown("### Welcome to your comprehensive hearing evaluation")
    
    with st.form("user_onboarding"):
        st.subheader("👤 Personal Information")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name*", key="name")
            age = st.number_input("Age*", min_value=1, max_value=120, value=25)
            gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
        
        with col2:
            email = st.text_input("Email (optional)", key="email")
            occupation = st.text_input("Occupation (optional)", key="occupation")
        
        st.subheader("🔊 Hearing & Noise Exposure")
        
        col3, col4 = st.columns(2)
        with col3:
            noise_exposure = st.selectbox(
                "How often are you exposed to loud environments?*",
                ["Rarely", "Sometimes", "Often", "Very Often", "Daily"]
            )
            
            headphone_usage = st.selectbox(
                "Daily headphone/earphone usage*",
                ["Less than 1 hour", "1-3 hours", "3-6 hours", "6+ hours"]
            )
        
        with col4:
            hearing_concerns = st.multiselect(
                "Current hearing concerns",
                ["None", "Difficulty in conversations", "Ringing in ears", 
                 "Ear pain", "Recent hearing changes", "Noise sensitivity"]
            )
            
            previous_tests = st.selectbox(
                "Previous professional hearing tests",
                ["Never", "Within last year", "1-3 years ago", "More than 3 years ago"]
            )
        
        st.subheader("🎧 Test Environment Setup")
        
        environment = st.selectbox(
            "Current testing environment*",
            ["Quiet room with headphones", "Quiet room with speakers", 
             "Somewhat noisy", "Very noisy environment"]
        )
        
        device_type = st.selectbox(
            "Audio device being used*",
            ["Over-ear headphones", "In-ear headphones/earbuds", 
             "Computer speakers", "Phone speakers"]
        )
        
        st.info("""
        **📋 Before You Begin:**
        - Find a quiet environment for accurate results
        - Use headphones or earphones for best results  
        - Adjust volume to a comfortable listening level
        - This screening tool is for educational purposes only
        """)
        
        consent = st.checkbox("I understand this is a screening tool and consent to the hearing assessment*")
        
        submitted = st.form_submit_button("🚀 Begin Assessment", use_container_width=True, type="primary")
        
        if submitted:
            if not name or not consent:
                st.error("⚠️ Please fill in required fields (*) and provide consent.")
            else:
                # Save user profile
                st.session_state.user_profile = {
                    'name': name,
                    'age': age,
                    'gender': gender,
                    'email': email,
                    'occupation': occupation,
                    'noise_exposure': noise_exposure,
                    'headphone_usage': headphone_usage,
                    'hearing_concerns': hearing_concerns,
                    'previous_tests': previous_tests,
                    'environment': environment,
                    'device_type': device_type,
                    'registration_date': datetime.now().isoformat()
                }
                st.session_state.current_page = 'calibration'
                st.success("✅ Profile saved! Proceeding to audio calibration...")
                time.sleep(1)
                st.rerun()

# Enhanced calibration with FreeSound integration
def render_calibration():
    """Enhanced calibration interface with FreeSound integration"""
    st.title("🎵 Advanced Audio Calibration")
    st.markdown("### Optimizing your audio settings with professional-grade sounds")
    
    audio_manager = AudioTestManager()
    
    st.info("""
    **🔧 Enhanced Calibration Process:**
    1. We'll use high-quality audio from FreeSound.org when available
    2. Put on your headphones or position near speakers
    3. Play the calibration tone below  
    4. Adjust your device volume until the tone is clearly audible but comfortable
    5. The tone should not cause discomfort or be too quiet
    """)
    
    # Audio source indicator
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get calibration audio (FreeSound or generated)
        with st.spinner("🎵 Loading calibration audio..."):
            calibration_audio = audio_manager.get_calibration_audio()
        
        st.subheader("🎯 Professional Calibration Tone (1000 Hz)")
        
        if isinstance(calibration_audio, str) and calibration_audio.startswith("http"):
            st.audio(calibration_audio)
            st.success("✨ Using professional audio from FreeSound.org")
        else:
            st.audio(calibration_audio)
            st.info("🔊 Using generated calibration tone")
        
        if st.button("🔄 Load New Calibration Tone", use_container_width=True):
            st.rerun()
    
    with col2:
        st.subheader("📊 Audio Info")
        st.write("**Frequency:** 1000 Hz")
        st.write("**Duration:** 3 seconds")
        st.write("**Purpose:** Volume calibration")
        
        # API status indicator
        if audio_manager.freesound_api_key:
            st.success("🔑 FreeSound API: Connected")
        else:
            st.warning("🔑 FreeSound API: Not configured")
        
    # Rest of calibration interface remains the same...
    st.markdown("**Volume Guidelines:**")
    st.write("• Too Quiet: You can barely hear the tone")
    st.write("• Just Right: Clear and comfortable to listen to")  
    st.write("• Too Loud: Uncomfortable or causes distortion")
    
    volume_check = st.radio(
        "How does the calibration tone sound?",
        ["Too quiet - barely audible",
         "Perfect - clear and comfortable", 
         "Too loud - uncomfortable"]
    )
    
    if volume_check == "Perfect - clear and comfortable":
        st.success("🎉 Audio calibrated successfully!")
        
        if st.button("Continue to Hearing Test →", use_container_width=True, type="primary"):
            st.session_state.current_page = 'hearing_test'
            st.rerun()
    else:
        st.warning("⚠️ Please adjust your device volume and try the calibration tone again.")

def render_hearing_test():
    """Render comprehensive hearing test interface"""
    st.title("🔊 Comprehensive Hearing Assessment")
    
    # Initialize test if not started
    if not st.session_state.test_started:
        st.session_state.test_started = True
        st.session_state.current_test_step = 0
        st.session_state.test_results = []
        st.session_state.current_ear = 'left'
    
    audio_manager = AudioTestManager()
    
    # Calculate progress
    total_tests = len(STANDARD_FREQUENCIES) * len(VOLUME_LEVELS) * 2  # Both ears
    current_progress = st.session_state.current_test_step
    
    # Progress display
    progress_percent = min(current_progress / total_tests, 1.0) if total_tests > 0 else 0
    st.progress(progress_percent)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Progress", f"{current_progress}/{total_tests}")
    with col2:
        st.metric("Current Ear", st.session_state.current_ear.title())
    with col3:
        st.metric("Completion", f"{progress_percent:.0%}")
    
    if current_progress < total_tests:
        render_single_tone_test(audio_manager)
    else:
        render_test_completion()

def render_single_tone_test(audio_manager):
    """Render interface for individual tone testing"""
    current_progress = st.session_state.current_test_step  
    # Calculate current test parameters
    tests_per_ear = len(STANDARD_FREQUENCIES) * len(VOLUME_LEVELS)
    
    if st.session_state.current_ear == 'left':
        test_in_ear = st.session_state.current_test_step
    else:
        test_in_ear = st.session_state.current_test_step - tests_per_ear
    
    freq_idx = test_in_ear // len(VOLUME_LEVELS)
    vol_idx = test_in_ear % len(VOLUME_LEVELS)
    
    if freq_idx >= len(STANDARD_FREQUENCIES):
        # Switch to right ear or complete
        if st.session_state.current_ear == 'left':
            st.session_state.current_ear = 'right'
            st.session_state.current_test_step = tests_per_ear
            st.rerun()
        else:
            render_test_completion()
            return
    
    frequency = STANDARD_FREQUENCIES[freq_idx]
    volume = VOLUME_LEVELS[vol_idx] 
    frequency_name = FREQUENCY_NAMES[freq_idx]
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"🎯 Testing: {frequency_name} at {volume}dB - {st.session_state.current_ear.title()} Ear")
        
        # Generate test audio
        test_audio = audio_manager.get_test_audio(frequency, volume)
        
        st.info(f"""
        **📢 Listen Carefully:**
        - A tone will play at {frequency_name} frequency
        - Volume level: {volume}dB
        - Focus on your {st.session_state.current_ear} ear
        - Respond honestly based on what you hear
        """)
        
        # Audio player
        st.audio(test_audio)
        
        st.markdown("**🎯 Did you hear the tone clearly?**")
        
        # Response buttons
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("✅ Heard Clearly", key=f"clear_{current_progress}", use_container_width=True):
                record_response(frequency, volume, st.session_state.current_ear, "clear")
                advance_to_next_test()
                
        with col_b:
            if st.button("🤔 Heard Faintly", key=f"faint_{current_progress}", use_container_width=True):
                record_response(frequency, volume, st.session_state.current_ear, "faint")
                advance_to_next_test()
                
        with col_c:
            if st.button("❌ Not Heard", key=f"none_{current_progress}", use_container_width=True):
                record_response(frequency, volume, st.session_state.current_ear, "none")
                advance_to_next_test()
    
    with col2:
        st.subheader("📊 Test Info")
        st.write(f"**Frequency:** {frequency_name}")
        st.write(f"**Volume:** {volume}dB")
        st.write(f"**Ear:** {st.session_state.current_ear.title()}")
        st.write(f"**Step:** {test_in_ear + 1}")
        
        # Quick stats
        if st.session_state.test_results:
            heard_count = sum(1 for r in st.session_state.test_results if r['response'] != 'none')
            st.metric("Heard So Far", f"{heard_count}/{len(st.session_state.test_results)}")

def record_response(frequency: int, volume: int, ear: str, response: str):
    """Record user response to hearing test"""
    result = {
        'timestamp': datetime.now().isoformat(),
        'frequency': frequency,
        'volume': volume,
        'ear': ear,
        'response': response,
        'heard': response != 'none'
    }
    st.session_state.test_results.append(result)
    
    # Update hearing data matrix
    freq_idx = STANDARD_FREQUENCIES.index(frequency)
    vol_idx = VOLUME_LEVELS.index(volume)
    
    response_values = {'none': 0, 'faint': 0.5, 'clear': 1}
    st.session_state.hearing_data[freq_idx][vol_idx] = response_values[response]

def advance_to_next_test():
    """Move to next test step"""
    st.session_state.current_test_step += 1
    time.sleep(0.5)  # Brief pause for user experience
    st.rerun()

def render_test_completion():
    """Render test completion interface"""
    st.success("🎉 Hearing Assessment Complete!")
    
    # Generate test summary
    results_df = pd.DataFrame(st.session_state.test_results)
    test_summary = generate_test_summary(results_df)
    
    # Save to history
    test_session = {
        'date': datetime.now().isoformat(),
        'user': st.session_state.user_profile.get('name', 'Anonymous'),
        'results': st.session_state.test_results,
        'summary': test_summary,
        'hearing_data': st.session_state.hearing_data.tolist()
    }
    st.session_state.test_history.append(test_session)
    
    # Update gamification
    update_gamification_data(results_df)
    
    # Display completion stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tests", len(results_df))
    with col2:
        heard_count = len(results_df[results_df['heard'] == True])
        st.metric("Sounds Heard", heard_count)
    with col3:
        avg_threshold = test_summary.get('average_threshold', 0)
        st.metric("Avg Threshold", f"{avg_threshold:.1f}dB")
    with col4:
        points_earned = st.session_state.gamification_data.get('total_points', 0)
        st.metric("Points Earned", points_earned)
    
    st.markdown("---")
    
    # Next steps
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        if st.button("📊 View Detailed Results", use_container_width=True, type="primary"):
            st.session_state.current_page = 'results'
            st.rerun()
    
    with col_b:
        if st.button("🎮 Try Audio Challenge", use_container_width=True):
            st.session_state.current_page = 'challenge'
            st.rerun()
    
    with col_c:
        if st.button("💡 Get AI Recommendations", use_container_width=True):
            st.session_state.current_page = 'feedback'
            st.rerun()

def render_results():
    """Render detailed results with comprehensive analysis"""
    st.title("📊 Your Hearing Assessment Results")
    
    if not st.session_state.test_results:
        st.warning("⚠️ No test results available. Please complete a hearing test first.")
        if st.button("Take Hearing Test", type="primary"):
            st.session_state.current_page = 'hearing_test'
            st.rerun()
        return
    
    results_df = pd.DataFrame(st.session_state.test_results)
    
    # Summary statistics
    render_results_summary(results_df)
    
    # Visual analysis
    render_audiogram_visualization(results_df)
    
    # AI-powered analysis
    render_ai_analysis(results_df)
    
    # Progress tracking
    render_progress_tracking()

def render_results_summary(results_df: pd.DataFrame):
    """Render results summary section"""
    st.subheader("📈 Test Summary")
    
    # Calculate key metrics
    total_tests = len(results_df)
    heard_clearly = len(results_df[results_df['response'] == 'clear'])
    heard_faintly = len(results_df[results_df['response'] == 'faint'])
    not_heard = len(results_df[results_df['response'] == 'none'])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Tests", total_tests)
    with col2:
        st.metric("Heard Clearly", heard_clearly, f"{heard_clearly/total_tests:.0%}")
    with col3:
        st.metric("Heard Faintly", heard_faintly, f"{heard_faintly/total_tests:.0%}")
    with col4:
        st.metric("Not Heard", not_heard, f"{not_heard/total_tests:.0%}")
    with col5:
        overall_score = (heard_clearly * 2 + heard_faintly) / (total_tests * 2) * 100
        st.metric("Overall Score", f"{overall_score:.0f}%")

def render_audiogram_visualization(results_df: pd.DataFrame):
    """Render audiogram and hearing threshold visualizations"""
    st.subheader("📈 Hearing Threshold Analysis")
    
    # Calculate thresholds for each frequency and ear
    thresholds_data = []
    
    for ear in ['left', 'right']:
        for freq_idx, frequency in enumerate(STANDARD_FREQUENCIES):
            ear_freq_data = results_df[
                (results_df['ear'] == ear) & 
                (results_df['frequency'] == frequency)
            ]
            
            if not ear_freq_data.empty:
                # Find threshold (minimum volume where sound was heard)
                heard_data = ear_freq_data[ear_freq_data['heard'] == True]
                if not heard_data.empty:
                    threshold = heard_data['volume'].min()
                else:
                    threshold = 85  # Above test range
                
                thresholds_data.append({
                    'ear': ear,
                    'frequency': frequency,
                    'frequency_name': FREQUENCY_NAMES[freq_idx],
                    'threshold_db': threshold
                })
    
    if thresholds_data:
        threshold_df = pd.DataFrame(thresholds_data)
        
        # Create audiogram plot
        fig = go.Figure()
        
        colors = {'left': '#1f77b4', 'right': '#ff7f0e'}
        symbols = {'left': 'circle', 'right': 'x'}
        
        for ear in ['left', 'right']:
            ear_data = threshold_df[threshold_df['ear'] == ear]
            if not ear_data.empty:
                fig.add_trace(go.Scatter(
                    x=ear_data['frequency'],
                    y=ear_data['threshold_db'],
                    mode='lines+markers',
                    name=f'{ear.title()} Ear',
                    line=dict(color=colors[ear], width=3),
                    marker=dict(symbol=symbols[ear], size=12)
                ))
        
        fig.update_layout(
            title="Audiogram - Hearing Thresholds",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Hearing Threshold (dB HL)",
            yaxis=dict(autorange='reversed', range=[0, 90]),
            xaxis=dict(type='log'),
            height=500,
            showlegend=True
        )
        
        # Add hearing level zones
        fig.add_hrect(y0=0, y1=25, fillcolor="green", opacity=0.2, 
                      annotation_text="Normal", annotation_position="top left")
        fig.add_hrect(y0=25, y1=40, fillcolor="yellow", opacity=0.2,
                      annotation_text="Mild Loss", annotation_position="top left")
        fig.add_hrect(y0=40, y1=70, fillcolor="orange", opacity=0.2,
                      annotation_text="Moderate Loss", annotation_position="top left")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hearing data heatmap
        st.subheader("🔥 Response Intensity Heatmap")
        
        # Create heatmap data
        left_ear_data = np.zeros((len(STANDARD_FREQUENCIES), len(VOLUME_LEVELS)))
        right_ear_data = np.zeros((len(STANDARD_FREQUENCIES), len(VOLUME_LEVELS)))
        
        for _, row in results_df.iterrows():
            freq_idx = STANDARD_FREQUENCIES.index(row['frequency'])
            vol_idx = VOLUME_LEVELS.index(row['volume'])
            response_val = {'none': 0, 'faint': 0.5, 'clear': 1}[row['response']]
            
            if row['ear'] == 'left':
                left_ear_data[freq_idx][vol_idx] = response_val
            else:
                right_ear_data[freq_idx][vol_idx] = response_val
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_left = px.imshow(
                left_ear_data,
                x=[f"{vol}dB" for vol in VOLUME_LEVELS],
                y=FREQUENCY_NAMES,
                color_continuous_scale="RdYlGn",
                title="Left Ear Response Map",
                aspect="auto"
            )
            st.plotly_chart(fig_left, use_container_width=True)
        
        with col2:
            fig_right = px.imshow(
                right_ear_data,
                x=[f"{vol}dB" for vol in VOLUME_LEVELS],
                y=FREQUENCY_NAMES,
                color_continuous_scale="RdYlGn",
                title="Right Ear Response Map",
                aspect="auto"
            )
            st.plotly_chart(fig_right, use_container_width=True)

def render_ai_analysis(results_df: pd.DataFrame):
    """Render AI-powered analysis and recommendations"""
    st.subheader("🤖 AI-Powered Analysis & Recommendations")
    
    ai_integration = AIIntegration()
    
    # Prepare analysis data
    user_profile = st.session_state.user_profile
    test_summary = generate_test_summary(results_df)
    
    analysis_prompt = f"""
    Analyze this hearing test data and provide professional insights:
    
    User Profile:
    - Age: {user_profile.get('age')}
    - Noise Exposure: {user_profile.get('noise_exposure')}
    - Headphone Usage: {user_profile.get('headphone_usage')}
    - Hearing Concerns: {user_profile.get('hearing_concerns')}
    - Previous Tests: {user_profile.get('previous_tests')}
    
    Test Results Summary:
    - Total Tests: {len(results_df)}
    - Average Threshold: {test_summary.get('average_threshold', 'N/A')}dB
    - Left Ear Average: {test_summary.get('left_ear_avg', 'N/A')}dB
    - Right Ear Average: {test_summary.get('right_ear_avg', 'N/A')}dB
    - Completion Rate: {test_summary.get('completion_rate', 'N/A')}%
    
    Please provide:
    1. Overall hearing assessment interpretation
    2. Identified risk factors
    3. Personalized recommendations for hearing health
    4. Whether professional follow-up is recommended
    """
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🧠 Claude Analysis**")
        with st.spinner("Analyzing your results with Claude AI..."):
            claude_analysis = ai_integration.call_anthropic_api(analysis_prompt)
        
        st.markdown(claude_analysis)
    
    with col2:
        st.markdown("**💬 ChatGPT Recommendations**")
        with st.spinner("Generating personalized advice with ChatGPT..."):
            gpt_recommendations = ai_integration.call_openai_api(analysis_prompt)
        
        st.markdown(gpt_recommendations)
    
    # Save results with AI feedback
    if st.button("💾 Save Results with AI Analysis", use_container_width=True):
        db_manager = DatabaseManager()
        
        save_data = {
            **user_profile,
            "hearing_data": st.session_state.hearing_data.tolist(),
            "ai_feedback": f"Claude Analysis:\n{claude_analysis}\n\nChatGPT Recommendations:\n{gpt_recommendations}",
            "game_score": st.session_state.gamification_data.get('total_points', 0),
            "test_summary": test_summary
        }
        
        db_manager.save_test_results(save_data)
        st.success("✅ Results and AI analysis saved successfully!")

def render_progress_tracking():
    """Render progress tracking over multiple sessions"""
    st.subheader("📈 Progress Tracking")
    
    if len(st.session_state.test_history) < 2:
        st.info("💡 Complete more hearing assessments to track your progress over time!")
        return
    
    # Prepare progress data
    progress_data = []
    for i, session in enumerate(st.session_state.test_history):
        summary = session.get('summary', {})
        progress_data.append({
            'session': i + 1,
            'date': session['date'][:10],
            'average_threshold': summary.get('average_threshold', 0),
            'completion_rate': summary.get('completion_rate', 0),
            'left_ear_avg': summary.get('left_ear_avg', 0),
            'right_ear_avg': summary.get('right_ear_avg', 0)
        })
    
    progress_df = pd.DataFrame(progress_data)
    
    # Progress visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig_threshold = px.line(
            progress_df, 
            x='session', 
            y='average_threshold',
            title='Average Hearing Threshold Trend',
            markers=True
        )
        fig_threshold.update_layout(
            xaxis_title="Test Session",
            yaxis_title="Average Threshold (dB)",
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig_threshold, use_container_width=True)
    
    with col2:
        fig_completion = px.line(
            progress_df,
            x='session',
            y='completion_rate', 
            title='Test Completion Rate Trend',
            markers=True
        )
        fig_completion.update_layout(
            xaxis_title="Test Session",
            yaxis_title="Completion Rate (%)"
        )
        st.plotly_chart(fig_completion, use_container_width=True)

def render_adaptive_challenge():
    """Render gamified adaptive audio challenge"""
    st.title("🎮 Adaptive Audio Challenge")
    st.markdown("### Test your hearing discrimination skills!")
    
    challenge_data = st.session_state.adaptive_challenge
    
    # Game status display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Difficulty", f"Level {challenge_data['difficulty']}")
    with col2:
        st.metric("Score", challenge_data['score'])
    with col3:
        st.metric("Lives", f"❤️ × {challenge_data['lives']}")
    with col4:
        level = st.session_state.gamification_data.get('level', 1)
        st.metric("Player Level", level)
    
    # Check game over condition
    if challenge_data['lives'] <= 0:
        render_challenge_game_over()
        return
    
    # Initialize new challenge if needed
    if 'current_challenge' not in st.session_state:
        generate_new_challenge()
    
    render_current_challenge()

def generate_new_challenge():
    """Generate new adaptive audio challenge"""
    game = HearingGame()
    difficulty = st.session_state.adaptive_challenge['difficulty']
    
    challenge = game.generate_sound_sequence_game(difficulty)
    st.session_state.current_challenge = challenge

def render_current_challenge():
    """Render current challenge interface"""
    challenge = st.session_state.current_challenge
    
    st.markdown("---")
    st.subheader(f"🎯 Challenge Round {st.session_state.adaptive_challenge['score'] + 1}")
    
    st.info(f"""
    **🎵 Audio Discrimination Challenge**
    
    **Instructions:**
    - Listen to the 4 audio sequences below
    - One sound is different from the others
    - Click the button under the different sound
    - Difficulty Level: {challenge['difficulty']} (Higher = More Challenging)
    """)
    
    # Display audio sequence
    st.markdown("**🔊 Listen to each sound:**")
    
    cols = st.columns(4)
    
    for i, sound_data in enumerate(challenge['sequence']):
        with cols[i]:
            st.write(f"**Sound {i+1}**")
            
            # Generate audio for this sound
            audio_data = generate_tone(
                sound_data['freq'], 
                1.5, 
                sound_data['volume'] / 100.0
            )
            
            st.audio(audio_data)
            
            # Response button
            if st.button(
                f"🎯 This One!", 
                key=f"challenge_btn_{i}",
                use_container_width=True
            ):
                process_challenge_response(i)

def process_challenge_response(selected_index: int):
    """Process user response to challenge"""
    challenge = st.session_state.current_challenge
    correct_index = challenge['different_position']
    
    if selected_index == correct_index:
        # Correct answer
        st.session_state.adaptive_challenge['score'] += 1
        st.success(f"🎉 Correct! Sound {selected_index + 1} was different!")
        
        # Increase difficulty every 3 correct answers
        if st.session_state.adaptive_challenge['score'] % 3 == 0:
            game = HearingGame()
            st.session_state.adaptive_challenge['difficulty'] = game.adjust_difficulty(
                True, st.session_state.adaptive_challenge['difficulty']
            )
            st.info(f"🔥 Difficulty increased to Level {st.session_state.adaptive_challenge['difficulty']}!")
        
        # Add points to gamification
        points = st.session_state.adaptive_challenge['difficulty'] * 5
        st.session_state.gamification_data['total_points'] += points
        
    else:
        # Incorrect answer
        st.session_state.adaptive_challenge['lives'] -= 1
        st.error(f"❌ Incorrect! Sound {correct_index + 1} was different, you selected Sound {selected_index + 1}")
        
        # Decrease difficulty
        if st.session_state.adaptive_challenge['score'] > 0:
            game = HearingGame()
            st.session_state.adaptive_challenge['difficulty'] = game.adjust_difficulty(
                False, st.session_state.adaptive_challenge['difficulty']
            )
    
    # Generate new challenge if lives remaining
    if st.session_state.adaptive_challenge['lives'] > 0:
        generate_new_challenge()
    
    time.sleep(2)
    st.rerun()

def render_challenge_game_over():
    """Render challenge game over screen"""
    score = st.session_state.adaptive_challenge['score']
    max_difficulty = st.session_state.adaptive_challenge['difficulty']
    
    st.subheader("🎮 Challenge Complete!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Final Score", score)
    with col2:
        st.metric("Max Difficulty", f"Level {max_difficulty}")
    with col3:
        points_earned = score * 5
        st.metric("Points Earned", points_earned)
    
    # Performance evaluation
    if score >= 15:
        st.success("🏆 Excellent performance! Your hearing discrimination skills are outstanding!")
    elif score >= 10:
        st.info("👍 Good performance! You have solid hearing discrimination abilities.")
    elif score >= 5:
        st.warning("📈 Fair performance. Consider practicing more to improve your skills.")
    else:
        st.error("💪 Keep practicing! Hearing discrimination improves with training.")
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Play Again", use_container_width=True):
            # Reset challenge
            st.session_state.adaptive_challenge = {
                'difficulty': 1,
                'score': 0, 
                'lives': 3
            }
            if 'current_challenge' in st.session_state:
                del st.session_state.current_challenge
            st.rerun()
    
    with col2:
        if st.button("📊 View Results", use_container_width=True):
            st.session_state.current_page = 'results'
            st.rerun()
    
    with col3:
        if st.button("🏠 Main Menu", use_container_width=True):
            st.session_state.current_page = 'onboarding'
            st.rerun()

def render_personalized_feedback():
    """Render personalized feedback and recommendations"""
    st.title("💡 Personalized Hearing Health Recommendations")
    
    if not st.session_state.test_results:
        st.warning("⚠️ Complete a hearing assessment to receive personalized recommendations.")
        if st.button("Take Hearing Test", type="primary"):
            st.session_state.current_page = 'hearing_test'
            st.rerun()
        return
    
    results_df = pd.DataFrame(st.session_state.test_results)
    user_profile = st.session_state.user_profile
    
    # Generate comprehensive recommendations
    recommendations = generate_personalized_recommendations(results_df, user_profile)
    
    st.subheader("🎯 Your Personalized Action Plan")
    
    # Render recommendation categories
    categories = {
        '🛡️ Hearing Protection': recommendations.get('protection', []),
        '🏥 Health & Wellness': recommendations.get('health', []),
        '📱 Technology & Tools': recommendations.get('technology', []),
        '🎵 Lifestyle Adjustments': recommendations.get('lifestyle', []),
        '🔄 Follow-up Actions': recommendations.get('followup', [])
    }
    
    for category, recs in categories.items():
        if recs:
            with st.expander(category, expanded=True):
                for i, rec in enumerate(recs, 1):
                    st.write(f"{i}. {rec}")
    
    # Risk assessment
    render_risk_assessment(results_df, user_profile)
    
    # Educational resources
    render_educational_resources()

def generate_personalized_recommendations(results_df: pd.DataFrame, user_profile: Dict) -> Dict:
    """Generate comprehensive personalized recommendations"""
    recommendations = {
        'protection': [],
        'health': [],
        'technology': [],
        'lifestyle': [],
        'followup': []
    }
    
    # Analyze test results
    total_tests = len(results_df)
    heard_count = len(results_df[results_df['heard'] == True])
    accuracy = heard_count / total_tests if total_tests > 0 else 0
    
    # High frequency analysis
    high_freq_tests = results_df[results_df['frequency'] >= 4000]
    high_freq_heard = len(high_freq_tests[high_freq_tests['heard'] == True])
    high_freq_accuracy = high_freq_heard / len(high_freq_tests) if len(high_freq_tests) > 0 else 1
    
    # Age-based recommendations
    age = user_profile.get('age', 30)
    if age > 60:
        recommendations['health'].extend([
            "Schedule annual hearing evaluations due to age-related hearing changes",
            "Consider discussing hearing aid options with an audiologist if experiencing difficulties",
            "Monitor for signs of presbycusis (age-related hearing loss)"
        ])
        recommendations['lifestyle'].append("Use visual cues to supplement auditory information in conversations")
    
    # Noise exposure recommendations
    noise_exposure = user_profile.get('noise_exposure', 'Rarely')
    if noise_exposure in ['Often', 'Very Often', 'Daily']:
        recommendations['protection'].extend([
            "Use professional-grade hearing protection in noisy environments (NRR 25+ recommended)",
            "Take regular breaks from loud noise exposure (follow the 85dB/8-hour rule)",
            "Consider custom-molded earplugs for frequent noise exposure"
        ])
        recommendations['followup'].append("Document noise exposure levels and duration for professional review")
    
    # Headphone usage recommendations  
    headphone_usage = user_profile.get('headphone_usage', 'Less than 1 hour')
    if headphone_usage in ['3-6 hours', '6+ hours']:
        recommendations['lifestyle'].extend([
            "Follow the 60/60 rule: No more than 60% volume for 60 minutes at a time",
            "Take 5-minute breaks every hour during extended headphone use",
            "Use noise-canceling headphones to reduce need for high volumes"
        ])
        recommendations['technology'].append("Consider over-ear headphones instead of earbuds for extended use")
    
    # Results-based recommendations
    if accuracy < 0.7:
        recommendations['followup'].extend([
            "Schedule a comprehensive audiological evaluation",
            "Discuss results with a hearing healthcare professional"
        ])
        recommendations['lifestyle'].append("Inform family and friends about potential hearing difficulties")
    
    if high_freq_accuracy < 0.6:
        recommendations['health'].extend([
            "High-frequency hearing loss detected - avoid further noise damage",
            "Consider tinnitus monitoring as high-frequency loss often precedes tinnitus"
        ])
        recommendations['technology'].append("Explore hearing assistance devices with high-frequency amplification")
    
    # Hearing concerns recommendations
    concerns = user_profile.get('hearing_concerns', [])
    if 'Ringing in ears' in concerns:
        recommendations['health'].extend([
            "Consult an audiologist about tinnitus management strategies",
            "Consider tinnitus masking or sound therapy options"
        ])
        recommendations['lifestyle'].append("Avoid complete silence - use background sounds to mask tinnitus")
        
    if 'Difficulty in conversations' in concerns:
        recommendations['lifestyle'].extend([
            "Practice active listening techniques",
            "Position yourself to see speakers' faces in conversations"
        ])
        recommendations['technology'].append("Consider assistive listening devices for challenging environments")
    
    # General recommendations
    recommendations['lifestyle'].extend([
        "Maintain good ear hygiene - avoid cotton swabs in the ear canal",
        "Stay physically active to promote healthy blood circulation to the ears",
        "Manage stress levels, which can affect hearing perception"
    ])
    
    recommendations['health'].extend([
        "Stay up-to-date with ear and hearing health check-ups",
        "Monitor for ear infections or excessive wax buildup"
    ])
    
    return recommendations

def render_risk_assessment(results_df: pd.DataFrame, user_profile: Dict):
    """Render hearing health risk assessment"""
    st.subheader("⚠️ Hearing Health Risk Assessment")
    
    # Calculate risk factors
    risk_factors = []
    risk_score = 0
    
    # Age risk
    age = user_profile.get('age', 30)
    if age > 65:
        risk_factors.append("Age over 65 (increased risk of presbycusis)")
        risk_score += 2
    elif age > 50:
        risk_factors.append("Age over 50 (moderate risk of age-related changes)")
        risk_score += 1
        
    # Noise exposure risk
    noise_exposure = user_profile.get('noise_exposure', 'Rarely')
    if noise_exposure in ['Very Often', 'Daily']:
        risk_factors.append("High noise exposure (significant hearing damage risk)")
        risk_score += 3
    elif noise_exposure == 'Often':
        risk_factors.append("Frequent noise exposure (moderate risk)")
        risk_score += 2
    
    # Headphone usage risk
    headphone_usage = user_profile.get('headphone_usage', 'Less than 1 hour')
    if headphone_usage == '6+ hours':
        risk_factors.append("Excessive headphone use (high risk)")
        risk_score += 2
    elif headphone_usage in ['3-6 hours']:
        risk_factors.append("Heavy headphone use (moderate risk)")
        risk_score += 1
    
    # Test results risk
    if results_df is not None and len(results_df) > 0:
        heard_rate = len(results_df[results_df['heard'] == True]) / len(results_df)
        if heard_rate < 0.6:
            risk_factors.append("Poor test performance (potential hearing loss)")
            risk_score += 3
        elif heard_rate < 0.8:
            risk_factors.append("Moderate test performance (requires monitoring)")
            risk_score += 1
    
    # Hearing concerns risk
    concerns = user_profile.get('hearing_concerns', [])
    if 'Ringing in ears' in concerns:
        risk_factors.append("Tinnitus symptoms (potential hearing damage)")
        risk_score += 2
    if 'Recent hearing changes' in concerns:
        risk_factors.append("Recent hearing changes (requires immediate attention)")
        risk_score += 3
    
    # Display risk assessment
    if risk_score >= 6:
        st.error(f"🚨 **High Risk** (Score: {risk_score}/12)")
        st.error("Immediate professional evaluation recommended")
    elif risk_score >= 3:
        st.warning(f"⚠️ **Moderate Risk** (Score: {risk_score}/12)")
        st.warning("Regular monitoring and preventive measures recommended")
    else:
        st.success(f"✅ **Low Risk** (Score: {risk_score}/12)")
        st.success("Continue current hearing protection practices")
    
    if risk_factors:
        st.markdown("**Identified Risk Factors:**")
        for factor in risk_factors:
            st.write(f"• {factor}")

def render_educational_resources():
    """Render educational resources section"""
    st.subheader("📚 Educational Resources")
    
    with st.expander("🔊 Understanding Your Hearing", expanded=False):
        st.markdown("""
        **How Hearing Works:**
        - Sound waves travel through your outer ear to the eardrum
        - The middle ear bones amplify vibrations
        - The inner ear (cochlea) converts vibrations to electrical signals
        - The auditory nerve sends signals to your brain for processing
        
        **Common Causes of Hearing Loss:**
        - Age-related changes (presbycusis)
        - Noise exposure (temporary or permanent)
        - Ear infections or blockages
        - Genetic factors
        - Certain medications (ototoxic drugs)
        """)
    
    with st.expander("🛡️ Hearing Protection Guidelines", expanded=False):
        st.markdown("""
        **Safe Listening Levels:**
        - 85 dB: Maximum safe level for 8 hours
        - 90 dB: Safe for 2.5 hours
        - 95 dB: Safe for 47 minutes
        - 100 dB: Safe for 15 minutes
        - 110 dB: Safe for 1.5 minutes
        
        **Protection Strategies:**
        - Use earplugs or noise-canceling headphones
        - Follow the 60/60 rule for personal audio devices
        - Take breaks from noisy environments
        - Maintain safe distance from loud sound sources
        """)
    
    with st.expander("🏥 When to See a Professional", expanded=False):
        st.markdown("""
        **Consult an Audiologist if you experience:**
        - Sudden hearing loss
        - Persistent tinnitus (ringing in ears)
        - Ear pain or discharge
        - Difficulty understanding speech
        - Feeling of fullness in ears
        - Hearing loss after illness or medication
        
        **Regular Check-ups Recommended:**
        - Every 3 years for adults under 50
        - Every year for adults over 50
        - Immediately if you notice changes
        """)

def generate_test_summary(results_df: pd.DataFrame) -> Dict:
    """Generate comprehensive test summary statistics"""
    if results_df.empty:
        return {}
    
    # Basic statistics
    total_tests = len(results_df)
    heard_count = len(results_df[results_df['heard'] == True])
    
    # Calculate thresholds by ear
    left_ear_data = results_df[results_df['ear'] == 'left']
    right_ear_data = results_df[results_df['ear'] == 'right']
    
    left_heard = left_ear_data[left_ear_data['heard'] == True]
    right_heard = right_ear_data[right_ear_data['heard'] == True]
    
    summary = {
        'test_date': datetime.now().isoformat(),
        'total_tests': total_tests,
        'heard_count': heard_count,
        'completion_rate': (heard_count / total_tests * 100) if total_tests > 0 else 0,
        'average_threshold': left_heard['volume'].mean() if len(left_heard) > 0 else None,
        'left_ear_avg': left_heard['volume'].mean() if len(left_heard) > 0 else None,
        'right_ear_avg': right_heard['volume'].mean() if len(right_heard) > 0 else None,
        'frequencies_tested': sorted(results_df['frequency'].unique().tolist()),
        'volume_range': [results_df['volume'].min(), results_df['volume'].max()],
        'test_duration_minutes': 15  # Approximate
    }
    
    # Calculate average for both ears if both have data
    if summary['left_ear_avg'] and summary['right_ear_avg']:
        summary['average_threshold'] = (summary['left_ear_avg'] + summary['right_ear_avg']) / 2
    elif summary['left_ear_avg']:
        summary['average_threshold'] = summary['left_ear_avg']
    elif summary['right_ear_avg']:
        summary['average_threshold'] = summary['right_ear_avg']
    
    return summary

# =============================================================================
# NAVIGATION AND MAIN APPLICATION
# =============================================================================

def render_gamification_dashboard():
    """Render compact gamification dashboard for sidebar"""
    if not st.session_state.gamification_data:
        return
        
    gam_data = st.session_state.gamification_data
    
    st.sidebar.markdown("**🎮 Your Progress**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Level", gam_data.get('level', 1))
        st.metric("Points", gam_data.get('total_points', 0))
    
    with col2:
        st.metric("Tests", len(st.session_state.test_history))
        achievements = len(gam_data.get('achievements', []))
        st.metric("Badges", achievements)
    
    # Show recent achievements
    if gam_data.get('achievements'):
        st.sidebar.markdown("**🏆 Latest Badges:**")
        achievement_names = {
            'first_test': '🌟 First Steps',
            'consistent_user': '📅 Regular User', 
            'hearing_champion': '🏆 Champion',
            'perfect_score': '💯 Perfect Score'
        }
        
        recent_achievements = gam_data['achievements'][-2:]  # Show last 2
        for ach in recent_achievements:
            badge_name = achievement_names.get(ach, ach)
            st.sidebar.write(f"• {badge_name}")

def render_navigation():
    """Render navigation sidebar with user info"""
    st.sidebar.title("🔊 AI-Powered Hearing Health Suite")
    
    # User welcome message
    if st.session_state.user_profile:
        name = st.session_state.user_profile.get('name', 'User')
        st.sidebar.markdown(f"**Welcome back, {name}! 👋**")
        
        # Show gamification dashboard
        render_gamification_dashboard()
        st.sidebar.markdown("---")
    
    # Main navigation menu
    st.sidebar.markdown("**📋 Navigation Menu**")
    
    pages = {
        'onboarding': '👋 Get Started',
        'calibration': '🎵 Audio Setup',
        'hearing_test': '🔊 Hearing Test',
        'results': '📊 View Results',
        'challenge': '🎮 Audio Challenge',
        'feedback': '💡 Recommendations',
        'history': '📈 Progress History'
    }
    
    for page_key, page_name in pages.items():
        # Check prerequisites for certain pages
        disabled = False
        if page_key in ['calibration', 'hearing_test', 'results', 'challenge', 'feedback', 'history']:
            if not st.session_state.user_profile:
                disabled = True
        
        if page_key in ['results', 'feedback']:
            if not st.session_state.test_results:
                disabled = True
        
        button_type = "primary" if st.session_state.current_page == page_key else "secondary"
        
        if st.sidebar.button(
            page_name, 
            key=f"nav_{page_key}",
            use_container_width=True,
            disabled=disabled,
            type=button_type if not disabled else "secondary"
        ):
            if not disabled:
                st.session_state.current_page = page_key
                st.rerun()
            else:
                if not st.session_state.user_profile:
                    st.sidebar.error("Complete user setup first!")
                elif not st.session_state.test_results:
                    st.sidebar.error("Complete hearing test first!")
    
    st.sidebar.markdown("---")
    
    # Quick actions
    st.sidebar.markdown("**⚡ Quick Actions**")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Reset", key="reset_app", use_container_width=True):
            # Clear session state with confirmation
            if st.session_state.get('confirm_reset', False):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                initialize_session_state()
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.sidebar.warning("Click again to confirm reset")
    
    with col2:
        if st.button("💾 Export", key="export_data", use_container_width=True):
            export_user_data()
    
    # App info
    st.sidebar.info("💡 This app includes AI-powered analysis and real audio testing capabilities.")

def render_history():
    """Render test history and trends"""
    st.title("📈 Your Hearing Health History")
    
    if not st.session_state.test_history:
        st.info("📝 No test history available yet. Complete hearing assessments to track your progress!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔊 Take Hearing Test", use_container_width=True, type="primary"):
                st.session_state.current_page = 'hearing_test'
                st.rerun()
        with col2:
            if st.button("🎮 Try Audio Challenge", use_container_width=True):
                st.session_state.current_page = 'challenge'
                st.rerun()
        return
    
    # Display test history
    st.subheader("📋 Test Sessions History")
    
    history_data = []
    for i, session in enumerate(st.session_state.test_history):
        summary = session.get('summary', {})
        history_data.append({
            'Session': i + 1,
            'Date': session['date'][:10],
            'Time': session['date'][11:16],
            'Avg Threshold (dB)': f"{summary.get('average_threshold', 0):.1f}" if summary.get('average_threshold') else 'N/A',
            'Completion Rate': f"{summary.get('completion_rate', 0):.0f}%",
            'Tests Completed': summary.get('total_tests', 0)
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True)
    
    # Trend analysis
    if len(st.session_state.test_history) >= 2:
        st.subheader("📊 Trend Analysis")
        
        # Prepare trend data
        trend_data = []
        for i, session in enumerate(st.session_state.test_history):
            summary = session.get('summary', {})
            if summary.get('average_threshold'):
                trend_data.append({
                    'session': i + 1,
                    'date': session['date'][:10],
                    'avg_threshold': summary['average_threshold'],
                    'completion_rate': summary.get('completion_rate', 0),
                    'left_ear': summary.get('left_ear_avg', 0),
                    'right_ear': summary.get('right_ear_avg', 0)
                })
        
        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Threshold trend
                fig_threshold = px.line(
                    trend_df,
                    x='session',
                    y='avg_threshold',
                    title='Average Hearing Threshold Over Time',
                    markers=True,
                    line_shape='spline'
                )
                fig_threshold.update_layout(
                    xaxis_title="Session Number",
                    yaxis_title="Threshold (dB HL)",
                    yaxis=dict(autorange='reversed'),
                    height=400
                )
                st.plotly_chart(fig_threshold, use_container_width=True)
            
            with col2:
                # Ear comparison
                fig_ears = go.Figure()
                fig_ears.add_trace(go.Scatter(
                    x=trend_df['session'],
                    y=trend_df['left_ear'],
                    mode='lines+markers',
                    name='Left Ear',
                    line=dict(color='blue')
                ))
                fig_ears.add_trace(go.Scatter(
                    x=trend_df['session'],
                    y=trend_df['right_ear'], 
                    mode='lines+markers',
                    name='Right Ear',
                    line=dict(color='red')
                ))
                fig_ears.update_layout(
                    title='Left vs Right Ear Thresholds',
                    xaxis_title='Session Number',
                    yaxis_title='Threshold (dB HL)',
                    yaxis=dict(autorange='reversed'),
                    height=400
                )
                st.plotly_chart(fig_ears, use_container_width=True)
            
            # Statistical summary
            st.subheader("📈 Progress Summary")
            
            latest_session = trend_df.iloc[-1]
            first_session = trend_df.iloc[0]
            
            threshold_change = latest_session['avg_threshold'] - first_session['avg_threshold']
            completion_change = latest_session['completion_rate'] - first_session['completion_rate']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Threshold Change", 
                    f"{threshold_change:+.1f}dB",
                    delta=f"{threshold_change:+.1f}dB"
                )
            
            with col2:
                st.metric(
                    "Completion Change",
                    f"{completion_change:+.0f}%",
                    delta=f"{completion_change:+.0f}%"
                )
            
            with col3:
                st.metric("Total Sessions", len(st.session_state.test_history))
            
            with col4:
                days_span = (
                    datetime.fromisoformat(st.session_state.test_history[-1]['date']) - 
                    datetime.fromisoformat(st.session_state.test_history[0]['date'])
                ).days
                st.metric("Days Tracked", days_span)
            
            # Insights
            st.subheader("🔍 Progress Insights")
            
            if threshold_change < -2:
                st.success("✅ Your hearing thresholds have improved over time!")
            elif threshold_change > 2:
                st.warning("⚠️ Your hearing thresholds show some decline. Consider professional evaluation.")
            else:
                st.info("📊 Your hearing thresholds remain stable over time.")
            
            if completion_change > 10:
                st.success("🎯 Your test performance has significantly improved!")
            elif completion_change < -10:  
                st.warning("📉 Your test performance has declined. This may indicate changes worth monitoring.")

def export_user_data():
    """Export comprehensive user data"""
    if not st.session_state.user_profile:
        st.sidebar.error("No data to export. Complete user setup first.")
        return
    
    export_data = {
        'export_info': {
            'export_date': datetime.now().isoformat(),
            'app_version': '2.0.0',
            'data_format': 'JSON'
        },
        'user_profile': st.session_state.user_profile,
        'test_history': st.session_state.test_history,
        'gamification_data': st.session_state.gamification_data,
        'current_results': st.session_state.test_results,
        'hearing_data_matrix': st.session_state.hearing_data.tolist() if hasattr(st.session_state.hearing_data, 'tolist') else []
    }
    
    json_string = json.dumps(export_data, indent=2, default=str)
    
    st.sidebar.download_button(
        label="📁 Download Complete Data",
        data=json_string,
        file_name=f"hearing_health_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        help="Download all your hearing test data and progress history"
    )
    
    st.sidebar.success("📊 Data export ready!")

def main():
    """Main application function with professional medical interface"""
    st.set_page_config(
        page_title="ListenWell AI",
        page_icon="🔊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/ListenWell AI/issues',
            'Report a bug': 'https://github.com/yourusername/ListenWell AI/issues',
            'About': 'AI-Powered Hearing Health Assessment Suite v2.0'
        }
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-container {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-container {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add health check to sidebar
    render_health_check()
    
    # Render navigation sidebar
    render_navigation()
    
    # Main content routing (same as before)
    current_page = st.session_state.current_page
    
    if current_page == 'onboarding':
        render_onboarding()
    elif current_page == 'calibration':
        if not st.session_state.user_profile:
            st.error("⚠️ Please complete user onboarding first!")
            if st.button("👋 Go to Onboarding", type="primary"):
                st.session_state.current_page = 'onboarding'
                st.rerun()
        else:
            render_calibration()
    elif current_page == 'hearing_test':
        if not st.session_state.user_profile:
            st.error("⚠️ Please complete user setup first!")
            if st.button("👋 Complete Setup", type="primary"):
                st.session_state.current_page = 'onboarding'
                st.rerun()
        else:
            render_hearing_test()
    elif current_page == 'results':
        render_results()
    elif current_page == 'challenge':
        if not st.session_state.user_profile:
            st.error("⚠️ Please complete user setup first!")
        else:
            render_adaptive_challenge()
    elif current_page == 'feedback':
        render_personalized_feedback()
    elif current_page == 'history':
        render_history()
    else:
        st.error("❌ Page not found. Redirecting to main menu...")
        st.session_state.current_page = 'onboarding'
        st.rerun()
    
    # Footer with disclaimer and information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px; padding: 1rem;'>
        <p>⚠️ <strong>Medical Disclaimer:</strong> This application is a hearing screening tool for educational purposes only.</p>
        <p>🔊 AI-Powered Hearing Health Suite | 
           <a href="https://github.com/Kulsoom-Fatima/ListenWell AI">GitHub</a> | 
           <a href="https://github.com/Kulsoom-Fatima/ListenWell AI">Report Issues</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
    
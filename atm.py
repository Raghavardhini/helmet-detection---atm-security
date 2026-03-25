import cv2
from ultralytics import YOLO
import datetime
import os
import time
import threading
import numpy as np
from google.cloud import texttospeech
import json
from queue import Queue

class ATMSecuritySystem:
    def __init__(self):
        """Initialize ATM Security System"""
        self.print_header("ATM SECURITY SYSTEM INITIALIZING")
        
        # Initialize all components
        self.init_voice()
        self.init_yolo()
        self.init_face_detector()
        self.init_camera()
        self.init_evidence_folder()
        
        # Tracking variables
        self.face_missing_frames = 0
        self.face_present_frames = 0
        self.face_covered_frames = 0
        self.weapon_detected = False
        self.face_covered = False
        self.current_state = "INIT"
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds between voice alerts
        self.current_frame = None
        
        # Evidence tracking
        self.last_evidence_time = 0
        self.evidence_cooldown = 3  # seconds between evidence saves
        self.access_denied_evidence_saved = False
        self.weapon_evidence_saved = False
        self.access_granted_evidence_saved = False
        self.face_covered_evidence_saved = False
        
        # Detection thresholds
        self.FACE_VISIBLE_THRESHOLD = 5   # Frames to confirm face
        self.FACE_MISSING_THRESHOLD = 10  # Frames to confirm missing
        self.FACE_COVERED_THRESHOLD = 8   # Frames to confirm face covered
        self.CONFIDENCE_THRESHOLD = 0.5   # 50% confidence for weapons
        
        # Suspicious items (weapons)
        self.suspicious_items = ["knife", "scissors", "baseball bat", "gun"]
        
        # Items that cover face
        self.face_covering_items = ["helmet", "mask", "hat", "sunglasses", "hood", "cap"]
        
        # Colors for display
        self.COLORS = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'orange': (0, 165, 255),
            'purple': (255, 0, 255),
            'white': (255, 255, 255)
        }
        
        self.print_success("ATM Security System Ready!")
        self.show_instructions()
    
    def print_header(self, text):
        """Print header with decoration"""
        print("\n" + "="*70)
        print(f"{text}")
        print("="*70)
    
    def print_success(self, text):
        """Print success message"""
        print(f"{text}")
    
    def print_warning(self, text):
        """Print warning message"""
        print(f"{text}")
    
    def print_error(self, text):
        """Print error message"""
        print(f"{text}")
    
    def show_instructions(self):
        """Show usage instructions"""
        print("\n" + "-"*70)
        print(" CONTROLS:")
        print("  • 'q' - Quit System")
        print("  • 's' - Save Screenshot")
        print("  • 'v' - Test Voice")
        print("  • 'i' - Show Info")
        print("\n VOICE ALERTS:")
        # print("  •  Access Granted - 'Access Permitted'")
        # print("  •  No Face - 'Please show your face'")
        # print("  •  Face Covered - 'Please remove helmet or mask'")
        # print("  •  Weapon - 'Warning! Weapon detected'")
        print("\n EVIDENCE SAVING:")
        print("  • All access attempts saved automatically")
        print("-"*70 + "\n")
    
    def init_voice(self):
        """Initialize GCP Text-to-Speech system"""
        try:
            self.print_warning("Initializing Google Cloud Text-to-Speech...")
            
            # Load service account key from environment variable or file
            keyfile_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            
            if not keyfile_path or not os.path.exists(keyfile_path):
                self.print_error("GCP service account key not found!")
                self.print_warning("Set GOOGLE_APPLICATION_CREDENTIALS environment variable pointing to your service account JSON file")
                self.print_warning("OR save your key to: ./gcp_service_key.json")
                
                # Try default location
                if os.path.exists("./gcp_service_key.json"):
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./gcp_service_key.json"
                    self.print_success("Found key at ./gcp_service_key.json")
                else:
                    exit()
            
            # Initialize GCP Text-to-Speech client
            self.tts_client = texttospeech.TextToSpeechClient()
            self.voice_config = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Neural2-C"  # Female voice
            )
            self.audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            
            self.voice_enabled = True
            self.print_success("GCP Text-to-Speech: READY")
            
            # Voice queue system
            self.voice_queue = Queue()
            self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
            self.voice_thread.start()
            
            # Test voice with 60-second cooldown
            self.last_alert_time = 0
            self.alert_cooldown = 60  # 1 minute cooldown as per requirement
            
            # Test voice
            self.speak("ATM Security System activated", force=True)
            
        except Exception as e:
            self.print_error(f" Voice System FAILED: {e}")
            self.print_warning("Please install: pip install google-cloud-texttospeech")
            exit()
    
    def _voice_worker(self):
        """Background thread to handle voice queue"""
        while True:
            try:
                message = self.voice_queue.get()
                if message is None:
                    break
                
                # Wait for cooldown
                current_time = time.time()
                time_since_last = current_time - self.last_alert_time
                
                if time_since_last < self.alert_cooldown:
                    wait_time = self.alert_cooldown - time_since_last
                    print(f" Voice queued. Waiting {wait_time:.1f}s for cooldown...")
                    time.sleep(wait_time)
                
                # Play the voice
                self._play_voice(message)
                self.last_alert_time = time.time()
                self.voice_queue.task_done()
                
            except Exception as e:
                self.print_error(f" Voice worker error: {e}")
    
    def _play_voice(self, message):
        """Play voice using GCP Text-to-Speech"""
        try:
            print(f" VOICE: {message}")
            
            if not self.voice_enabled:
                return
            
            # Create synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=message)
            
            # Call TTS API
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice_config,
                audio_config=self.audio_config
            )
            
            # Save to temporary file and play
            temp_audio_file = "temp_tts.mp3"
            with open(temp_audio_file, "wb") as out:
                out.write(response.audio_content)
            
            # Play audio file
            os.system(f"start {temp_audio_file}" if os.name == 'nt' else f"afplay {temp_audio_file}")
            print(f" Voice played: {message}")
            
        except Exception as e:
            self.print_error(f" Voice playback error: {e}")
    
    def init_yolo(self):
        """Initialize YOLO model for weapon and helmet detection"""
        try:
            self.print_warning(" Loading YOLO model (first time may take a while)...")
            self.model = YOLO(" yolov8n.pt")
            self.print_success(" Weapon & Helmet Detection: READY")
        except Exception as e:
            self.print_error(f" Model Loading Failed: {e}")
            exit()
    
    def init_face_detector(self):
        """Initialize face detection cascade"""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if self.face_cascade.empty():
                raise Exception(" Failed to load cascade classifier")
            self.print_success(" Face Detection: READY")
        except Exception as e:
            self.print_error(f" Face Detection: FAILED - {e}")
            exit()
    
    def init_camera(self):
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                raise Exception(" Camera not found or busy")
            
            # Test camera
            ret, frame = self.cap.read()
            if not ret:
                raise Exception(" Failed to read from camera")
            
            self.print_success(" Camera: READY")
        except Exception as e:
            self.print_error(f" Camera: FAILED - {e}")
            exit()
    
    def init_evidence_folder(self):
        """Create evidence folder if not exists"""
        if not os.path.exists(" evidence"):
            os.makedirs(" evidence")
            self.print_success(" Evidence Folder: CREATED")
        else:
            self.print_success(" Evidence Folder: READY")
    
    def speak(self, message, force=False):
        """Queue voice message with 60-second cooldown"""
        if not self.voice_enabled:
            return
        
        current_time = time.time()
        
        # If forced, play immediately and reset cooldown
        if force:
            # Wait for cooldown first
            time_since_last = current_time - self.last_alert_time
            if time_since_last < self.alert_cooldown:
                wait_time = self.alert_cooldown - time_since_last
                print(f" Forced voice. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            
            self._play_voice(message)
            self.last_alert_time = time.time()
        else:
            # Queue the message
            print(f" Voice message queued: {message}")
            self.voice_queue.put(message)
    
    # ========== EVIDENCE SAVING ==========
    
    def save_evidence(self, reason, force=False):
        """Save evidence image with cooldown"""
        if self.current_frame is None:
            return
        
        current_time = time.time()
        
        # Check cooldown unless forced
        if not force and current_time - self.last_evidence_time < self.evidence_cooldown:
            return
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename based on reason
        if reason == "NO_FACE":
            filename = f"evidence/NO_FACE_{timestamp}.jpg"
            self.access_denied_evidence_saved = True
        elif reason == "FACE_COVERED":
            filename = f"evidence/FACE_COVERED_{timestamp}.jpg"
            self.face_covered_evidence_saved = True
        elif reason == "WEAPON":
            filename = f"evidence/WEAPON_{timestamp}.jpg"
            self.weapon_evidence_saved = True
        elif reason == "ACCESS_GRANTED":
            filename = f"evidence/ACCESS_GRANTED_{timestamp}.jpg"
            self.access_granted_evidence_saved = True
        elif reason == "MANUAL":
            filename = f"evidence/MANUAL_{timestamp}.jpg"
        else:
            filename = f"evidence/{reason}_{timestamp}.jpg"
        
        # Save the image
        cv2.imwrite(filename, self.current_frame)
        print(f" EVIDENCE SAVED: {filename}")
        
        self.last_evidence_time = current_time
    
    # ========== DETECTION METHODS ==========
    
    def detect_weapons_and_coverings(self, frame):
        """Detect weapons and face coverings in frame"""
        weapon_detected = False
        weapon_name = ""
        face_covered = False
        covering_item = ""
        
        try:
            results = self.model(frame, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if conf > self.CONFIDENCE_THRESHOLD:
                            label = self.model.names[cls]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Check for weapons
                            if label in self.suspicious_items:
                                weapon_detected = True
                                weapon_name = label
                                color = self.COLORS['red']
                            
                            # Check for face coverings
                            elif label in self.face_covering_items:
                                face_covered = True
                                covering_item = label
                                color = self.COLORS['purple']
                            
                            else:
                                color = self.COLORS['green']
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            self.print_error(f" Detection error: {e}")
        
        return weapon_detected, weapon_name, face_covered, covering_item
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
                maxSize=(300, 300)
            )
            
            # Draw faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), self.COLORS['blue'], 2)
                cv2.putText(frame, "FACE DETECTED", (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['blue'], 2)
            
            return len(faces) > 0
        except Exception as e:
            self.print_error(f"Face detection error: {e}")
            return False
    
    def draw_status_panel(self, frame):
        """Draw status panel on frame"""
        h, w = frame.shape[:2]
        
        # Top status bar
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
        
        # Status based on current state
        if self.weapon_detected:
            # Weapon detected
            cv2.putText(frame, "WEAPON DETECTED", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLORS['red'], 2)
            cv2.putText(frame, "ACCESS DENIED - SECURITY ALERT", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['red'], 2)
            cv2.rectangle(frame, (0, 0), (w-1, h-1), self.COLORS['red'], 5)
            
        elif self.face_covered:
            # Face covered
            cv2.putText(frame, "FACE COVERED", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLORS['purple'], 2)
            cv2.putText(frame, "ACCESS DENIED - REMOVE HELMET/MASK", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['purple'], 2)
            cv2.rectangle(frame, (0, 0), (w-1, h-1), self.COLORS['purple'], 5)
            
        elif self.face_missing_frames > self.FACE_MISSING_THRESHOLD:
            # No face
            cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLORS['red'], 2)
            cv2.putText(frame, "ACCESS DENIED - Please remove helmet and show your face", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['red'], 2)
            cv2.rectangle(frame, (0, 0), (w-1, h-1), self.COLORS['red'], 5)
            
        elif self.face_present_frames > self.FACE_VISIBLE_THRESHOLD:
            # Face verified
            cv2.putText(frame, "FACE VERIFIED", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLORS['green'], 2)
            cv2.putText(frame, "ACCESS PERMITTED - WELCOME", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['green'], 2)
            cv2.rectangle(frame, (0, 0), (w-1, h-1), self.COLORS['green'], 5)
            
        elif self.face_present_frames > 0:
            # Verifying
            progress = int((self.face_present_frames / self.FACE_VISIBLE_THRESHOLD) * 100)
            cv2.putText(frame, f" VERIFYING FACE {progress}%", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLORS['yellow'], 2)
            cv2.putText(frame, "PLEASE WAIT - SCANNING", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['yellow'], 2)
            cv2.rectangle(frame, (0, 0), (w-1, h-1), self.COLORS['yellow'], 5)
        
        # Bottom info panel
        y_start = h - 80
        cv2.rectangle(frame, (0, y_start), (w, h), (0, 0, 0), -1)
        
        # Status info
        info_text = f"State: {self.current_state} | Face: {self.face_present_frames}/{self.FACE_VISIBLE_THRESHOLD}"
        cv2.putText(frame, info_text, (10, y_start + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['white'], 1)
        
        # Evidence status
        cv2.putText(frame, " AUTO-EVIDENCE ACTIVE", (w - 250, y_start + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['yellow'], 1)
    
    def run(self):
        """Main loop"""
        self.print_header("ATM SECURITY MONITORING STARTED")
        print(" Voice alerts are ACTIVE")
        print(" Evidence will be saved for ALL cases\n")
        
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                self.print_error(" Failed to read from camera")
                break
            
            self.current_frame = frame.copy()
            
            # 1. DETECT WEAPONS AND FACE COVERINGS
            weapon_detected, weapon_name, face_covered, covering_item = self.detect_weapons_and_coverings(self.current_frame)
            
            if weapon_detected:
                self.weapon_detected = True
                self.face_covered = False
                
                if self.current_state != "WEAPON":
                    self.current_state = "WEAPON"
                    self.speak(f"Warning! {weapon_name} detected")
                    self.save_evidence("WEAPON", force=True)
                
                # Reset counters
                self.face_present_frames = 0
                self.face_missing_frames = 0
                self.face_covered_frames = 0
                
            elif face_covered:
                self.weapon_detected = False
                self.face_covered = True
                self.face_covered_frames += 1
                self.face_present_frames = 0
                self.face_missing_frames = 0
                
                if self.face_covered_frames > self.FACE_COVERED_THRESHOLD:
                    if self.current_state != "FACE_COVERED":
                        self.current_state = "FACE_COVERED"
                        self.speak(" Please remove helmet or mask")
                        self.save_evidence("FACE_COVERED", force=True)
            
            else:
                self.weapon_detected = False
                self.face_covered = False
                self.face_covered_frames = 0
                
                # 2. FACE DETECTION
                face_detected = self.detect_faces(self.current_frame)
                
                # Update face counters
                if face_detected:
                    self.face_missing_frames = 0
                    self.face_present_frames += 1
                else:
                    self.face_present_frames = 0
                    self.face_missing_frames += 1
                
                # 3. STATE MANAGEMENT WITH VOICE ALERTS
                if self.face_missing_frames > self.FACE_MISSING_THRESHOLD:
                    if self.current_state != "MISSING":
                        self.current_state = "MISSING"
                        self.speak("Please Remove your helmet")
                        self.save_evidence(" NO_FACE", force=True)
                
                elif self.face_present_frames > self.FACE_VISIBLE_THRESHOLD:
                    if self.current_state != " VISIBLE":
                        self.current_state = " VISIBLE"
                        self.speak("Access permitted")
                        self.save_evidence("ACCESS_GRANTED", force=True)
                
                elif face_detected:
                    if self.current_state != "VERIFYING":
                        self.current_state = "VERIFYING"
            
            # Draw status panel
            self.draw_status_panel(self.current_frame)
            
            # Show frame
            cv2.imshow("ATM INTELLIGENT SECURITY SYSTEM", self.current_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.print_header("SHUTTING DOWN SYSTEM")
                break
                
            elif key == ord('s'):
                self.save_evidence("MANUAL", force=True)
                print("Manual evidence saved!")
                
            elif key == ord('v'):
                print("Testing voice...")
                self.speak("This is a test message", force=True)
                
            elif key == ord('i'):
                self.show_instructions()
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        
        if self.voice_enabled:
            try:
                self.engine.stop()
            except:
                pass
        
        # Print evidence summary
        self.print_header("EVIDENCE SUMMARY")
        files = os.listdir("evidence") if os.path.exists("evidence") else []
        print(f" Total Evidence Files: {len(files)}")
        
        if files:
            print("\n Recent Files:")
            for f in sorted(files, reverse=True)[:8]:
                print(f"   • {f}")
        
        self.print_success("ATM Security System Stopped")

# ============================================
# MAIN PROGRAM
# ============================================
if __name__ == "__main__":
    try:
        # Create and run ATM security system
        atm = ATMSecuritySystem()
        atm.run()
        
    except KeyboardInterrupt:
        print("\n\n System stopped by user")
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        cv2.destroyAllWindows()
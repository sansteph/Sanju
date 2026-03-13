import eel
import cv2
import face_recognition
import pickle
from scipy.spatial import distance as dist
import numpy as np
import base64
import sqlite3
from datetime import date, datetime

# --- Constants and Settings ---
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password"
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
DB_FILE = "attendance_system.db"
# Indices for left/right eyes (dlib 68-point model)
(L_START, L_END) = (42, 48)
(R_START, R_END) = (36, 42)

# --- Global State ---
is_processing = False
KNOWN_ENCODINGS = []
KNOWN_NAMES = []
KNOWN_IDS = []


# --- Database Initialization for SQLite ---
def dict_factory(cursor, row):
    """Helper to return SQL results as dictionaries."""
    fields = [column[0] for column in cursor.description]
    return dict(zip(fields, row))


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Admins (
            admin_id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        )
    """)
    cursor.execute("INSERT OR IGNORE INTO Admins (admin_id, username, password_hash) VALUES (?, ?, ?)",
                   ('A001', ADMIN_USERNAME, ADMIN_PASSWORD))

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            student_name TEXT NOT NULL,
            course TEXT NOT NULL,
            semester TEXT NOT NULL,
            specialization TEXT NOT NULL,
            image_encoding BLOB NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS professors (
            prof_id TEXT PRIMARY KEY,
            prof_name TEXT NOT NULL UNIQUE,
            specialization TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            subj_id TEXT PRIMARY KEY,
            subj_name TEXT NOT NULL,
            semester TEXT NOT NULL,
            specialization TEXT NOT NULL,
            prof_name TEXT,
            FOREIGN KEY (prof_name) REFERENCES professors (prof_name)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS timetable (
            tt_id INTEGER PRIMARY KEY,
            tt_day TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            subj_id TEXT,
            FOREIGN KEY (subj_id) REFERENCES subjects (subj_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            att_id INTEGER PRIMARY KEY,
            student_id TEXT,
            login_date TEXT,
            period_id INTEGER,
            FOREIGN KEY (student_id) REFERENCES students (student_id),
            FOREIGN KEY (period_id) REFERENCES timetable (tt_id)
        )
    """)

    conn.commit()
    conn.close()
    print("[INFO] SQLite database and tables initialized successfully.")


# Must run once on startup
init_db()

# --- Eel Setup ---
eel.init('web')


# --- Helper Functions ---

def eye_aspect_ratio(eye):
    """Calculates the Eye Aspect Ratio (EAR)."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def get_face_data():
    """Loads all student encodings and identifiers from the database."""
    global KNOWN_ENCODINGS, KNOWN_NAMES, KNOWN_IDS
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, student_name, image_encoding FROM students")
    rows = cursor.fetchall()
    conn.close()

    KNOWN_ENCODINGS = [pickle.loads(row[2]) for row in rows]
    KNOWN_NAMES = [row[1] for row in rows]
    KNOWN_IDS = [row[0] for row in rows]

    print(f"[INFO] Loaded {len(KNOWN_ENCODINGS)} known faces.")
    return KNOWN_ENCODINGS, KNOWN_NAMES, KNOWN_IDS


def get_student_details(student_id):
    """Fetches full details for a single student."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = dict_factory
    cursor = conn.cursor()
    cursor.execute(
        "SELECT student_id, student_name, semester, course, specialization FROM students WHERE student_id=?",
        (student_id,))
    student = cursor.fetchone()
    conn.close()
    return student


def get_current_period():
    """
    Finds the currently scheduled class based on time and day.
    Uses current day of the week and current time.
    """
    now = datetime.now()
    current_day = now.strftime("%A")
    current_time = now.strftime("%H:%M:%S")

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    cursor.execute("""
        SELECT tt.tt_id AS period_id, s.subj_name, s.semester, s.specialization 
        FROM timetable tt
        JOIN subjects s ON tt.subj_id = s.subj_id
        WHERE tt.tt_day=? AND tt.start_time <= ? AND tt.end_time >= ?
        """,
                   (current_day, current_time, current_time))

    period = cursor.fetchone()
    conn.close()
    return period


# --- DB CRUD Helpers ---

def db_write(query, params, success_msg):
    """Helper for INSERT/UPDATE/DELETE operations."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        conn.close()
        return {"status": "success", "message": success_msg}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def db_read(query, params=None):
    """Helper for SELECT operations."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(query, params or ())
    result = cursor.fetchall()
    conn.close()
    return result

# GENERIC ADMIN CRUD FUNCTIONS (For Admin Panel Tables)

@eel.expose
def delete_record(table_name, pk_col, pk_value):
    """Generic function to delete a record from a specified table."""
    if table_name not in ['students', 'professors', 'subjects', 'timetable']:
        return {"status": "error", "message": "Invalid table name."}
    if pk_col not in ['student_id', 'prof_id', 'subj_id', 'tt_id']:
        return {"status": "error", "message": "Invalid primary key column."}

    query = f"DELETE FROM {table_name} WHERE {pk_col} = ?"
    result = db_write(query, (pk_value,), f"Record {pk_value} deleted from {table_name}.")

    # Reload face data if a student record was deleted
    if table_name == 'students' and result['status'] == 'success':
        get_face_data()

    return result


@eel.expose
def update_record(table_name, pk_col, pk_value, update_data):
    """Generic function to update fields in a record."""
    if table_name not in ['students', 'professors', 'subjects', 'timetable']:
        return {"status": "error", "message": "Invalid table name."}
    if pk_col not in ['student_id', 'prof_id', 'subj_id', 'tt_id']:
        return {"status": "error", "message": "Invalid primary key column."}

    if pk_col in update_data:
        del update_data[pk_col]

    if not update_data:
        return {"status": "error", "message": "No fields provided for update."}

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        set_clauses = [f"{col} = ?" for col in update_data.keys()]
        set_clause_str = ", ".join(set_clauses)

        query = f"UPDATE {table_name} SET {set_clause_str} WHERE {pk_col} = ?"

        params = list(update_data.values()) + [pk_value]

        cursor.execute(query, params)
        conn.commit()
        conn.close()

        if cursor.rowcount == 0:
            return {"status": "error", "message": f"No record found with {pk_col}={pk_value} in {table_name}."}

        # Reload face data if student info was updated
        if table_name == 'students':
            get_face_data()

        return {"status": "success", "message": f"Record {pk_value} updated successfully in {table_name}."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- Student Login & Liveness Loop ---

@eel.expose
def start_student_login():
    """
    Starts the continuous webcam feed and processing loop for Liveness and Recognition.
    """
    global is_processing
    is_processing = True

    KNOWN_ENCODINGS, KNOWN_NAMES, KNOWN_IDS = get_face_data()

    blink_counter = 0
    is_live = False
    attended_today = {}

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        eel.update_video_feed("error")
        is_processing = False
        return

    while is_processing:
        ret, frame = cam.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- 1. Face Detection ---
        boxes = face_recognition.face_locations(rgb_frame, model="hog")
        liveness_text = "Pending (Face Required)"

        if boxes:
            # --- 2. Liveness Check (Multi-frame EAR) ---
            landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations=boxes)

            if landmarks_list:
                landmarks = landmarks_list[0]
                left_eye = np.array(landmarks['left_eye'])
                right_eye = np.array(landmarks['right_eye'])

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                if ear < EYE_AR_THRESH:
                    blink_counter += 1
                else:
                    blink_counter = 0

                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    is_live = True
                    blink_counter = 0

                liveness_text = f"Liveness: {'Confirmed' if is_live else f'Blink Pending ({EYE_AR_CONSEC_FRAMES - blink_counter} frames)'}"

            # --- 3. Recognition (Only proceeds if Liveness is confirmed) ---
            if is_live:
                encodings = face_recognition.face_encodings(rgb_frame, boxes)

                for encoding, box in zip(encodings, boxes):
                    name_display = "Unknown"
                    current_period = get_current_period()

                    if not current_period:
                        eel.update_login_status(None, "No class scheduled at this time.", "error")
                        continue

                    # Attempt Match
                    matches = face_recognition.compare_faces(KNOWN_ENCODINGS, encoding, tolerance=0.5)

                    if True in matches:
                        student_index = matches.index(True)
                        student_id = KNOWN_IDS[student_index]

                        student_details = get_student_details(student_id)

                        # --- Basic Class/Time Validation ---
                        if student_details['semester'] != current_period['semester'] or \
                                student_details['specialization'] != current_period['specialization']:
                            eel.update_login_status(student_details,
                                                    f"Incorrect Class. This period is for Sem {current_period['semester']} ({current_period['specialization']}).",
                                                    "error")
                            continue

                        name = student_details['student_name']
                        name_display = f"{name} ({student_id})"
                        period_id = current_period['period_id']
                        subject_name = current_period['subj_name']

                        # Check if already marked for this period
                        if attended_today.get(student_id) != period_id:
                            today = date.today().strftime("%Y-%m-%d")
                            conn = sqlite3.connect(DB_FILE)
                            cursor = conn.cursor()

                            cursor.execute(
                                "SELECT * FROM attendance WHERE student_id = ? AND login_date = ? AND period_id = ?",
                                (student_id, today, period_id))

                            if cursor.fetchone() is None:
                                cursor.execute(
                                    "INSERT INTO attendance (student_id, login_date, period_id) VALUES (?, ?, ?)",
                                    (student_id, today, period_id))
                                conn.commit()
                                status_message = f"Attendance Marked for {subject_name}!"
                            else:
                                status_message = f"Attendance Already Marked for {subject_name}."

                            conn.close()

                            eel.update_login_status(student_details, status_message, "success")
                            attended_today[student_id] = period_id
                        else:
                            status_message = f"Attendance Confirmed for {subject_name}."
                            eel.update_login_status(student_details, status_message, "success")

                    else:
                        eel.update_login_status(None, "Unknown Face Detected", "error")

                    # Draw rectangle and name (Visual Feedback)
                    top, right, bottom, left = box
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name_display, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Liveness text drawing
        cv2.putText(frame, liveness_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Convert frame to Base64 JPEG and send to frontend
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        eel.update_video_feed(f'data:image/jpeg;base64,{jpg_as_text}')

        eel.sleep(0.01)

    cam.release()
    print("[INFO] Student login loop stopped.")


@eel.expose
def stop_processing():
    """Stops the webcam processing loop."""
    global is_processing
    is_processing = False


# --- Admin Management Functions (CRUD) ---

@eel.expose
def admin_login(u, p):
    """Hardcoded Admin Login check."""
    return {"status": "success", "message": "Login successful!"} if u == "admin" and p == "password" else {
        "status": "error", "message": "Invalid credentials."}


@eel.expose
def get_all_students():
    return db_read("SELECT student_id, student_name, course, semester, specialization FROM students")


@eel.expose
def add_student(sid, n, c, sem, spec, img_data):
    """Adds a student, computes encoding, and pickles it."""
    try:
        encoded_data = img_data.split(",")[1]
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded_data), np.uint8), cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb_frame, model="hog")
        if not boxes: return {"status": "error", "message": "No face detected."}

        enc = face_recognition.face_encodings(rgb_frame, boxes)[0]

        result = db_write("INSERT OR REPLACE INTO students VALUES (?,?,?,?,?,?)",
                          (sid, n, c, sem, spec, sqlite3.Binary(pickle.dumps(enc))), f"Student '{n}' added.")

        get_face_data()
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


@eel.expose
def get_professors():
    return db_read("SELECT prof_id, prof_name, specialization FROM professors")


@eel.expose
def add_professor(pid, n, spec=None):
    return db_write("INSERT OR REPLACE INTO professors VALUES (?,?,?)", (pid, n, spec), f"Professor '{n}' added.")


@eel.expose
def get_subjects():
    return db_read("SELECT subj_id, subj_name, semester, specialization, prof_name FROM subjects")


@eel.expose
def add_subject(id, name, sem, spec, prof_name):
    return db_write("INSERT OR REPLACE INTO subjects VALUES (?,?,?,?,?)", (id, name, sem, spec, prof_name),
                    f"Subject '{name}' added.")


@eel.expose
def get_timetable_slots():
    """Fetches all timetable slots."""
    return db_read(
        "SELECT tt.tt_id, tt.tt_day, tt.start_time, tt.end_time, s.subj_name, s.semester, s.specialization FROM timetable tt JOIN subjects s ON tt.subj_id = s.subj_id ORDER BY tt.tt_day, tt.start_time")


@eel.expose
def add_timetable_slot(day, start, end, subj_id):
    """Adds a timetable slot with time validation."""

    # 1. Validation Logic
    try:
        start_dt = datetime.strptime(start, '%H:%M')
        end_dt = datetime.strptime(end, '%H:%M')
    except ValueError:
        try:
            start_dt = datetime.strptime(start, '%H:%M:%S')
            end_dt = datetime.strptime(end, '%H:%M:%S')
        except ValueError:
            return {"status": "error", "message": "Invalid time format (use HH:MM or HH:MM:SS)."}

    today = datetime.now().date()
    start_dt = datetime.combine(today, start_dt.time())
    end_dt = datetime.combine(today, end_dt.time())

    MIN_START = datetime.combine(today, datetime.strptime('09:30', '%H:%M').time())
    MAX_END = datetime.combine(today, datetime.strptime('16:30', '%H:%M').time())

    if start_dt < MIN_START or end_dt > MAX_END:
        return {"status": "error", "message": "Class must be between 9:30 AM and 4:30 PM."}

    # Check for short break (10:30-10:45) and lunch break (13:45-14:15)
    break1_start = datetime.combine(today, datetime.strptime('10:30', '%H:%M').time())
    break1_end = datetime.combine(today, datetime.strptime('10:45', '%H:%M').time())
    break2_start = datetime.combine(today, datetime.strptime('13:45', '%H:%M').time())
    break2_end = datetime.combine(today, datetime.strptime('14:15', '%H:%M').time())

    if (start_dt < break1_end and end_dt > break1_start) or \
            (start_dt < break2_end and end_dt > break2_start):
        return {"status": "error", "message": "Class time overlaps with scheduled break/lunch time."}

    # 2. Insertion
    start_str = start_dt.strftime('%H:%M:%S')
    end_str = end_dt.strftime('%H:%M:%S')
    return db_write("INSERT INTO timetable (tt_day, start_time, end_time, subj_id) VALUES (?,?,?,?)",
                    (day, start_str, end_str, subj_id), "Timetable slot added.")


@eel.expose
def get_attendance_report(report_date):
    """Generates attendance report for a specific date."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        cursor.execute("SELECT student_id, student_name FROM students")
        all_students = cursor.fetchall()

        cursor.execute(
            "SELECT tt_id AS period_id, s.subj_name FROM timetable tt JOIN subjects s ON tt.subj_id = s.subj_id GROUP BY s.subj_name ORDER BY tt.start_time")
        periods = cursor.fetchall()

        cursor.execute("SELECT student_id, period_id FROM attendance WHERE login_date = ?", (report_date,))
        attendance_records = cursor.fetchall()

        present_map = [(rec['student_id'], rec['period_id']) for rec in attendance_records]

        report = {"periods": [p['subj_name'] for p in periods], "students": []}
        period_ids = [p['period_id'] for p in periods]

        subject_attendance_count = {p['subj_name']: 0 for p in periods}
        total_students = len(all_students)

        for student in all_students:
            student_row = {"name": student['student_name'], "student_id": student['student_id'], "status": []}

            for i, period_id in enumerate(period_ids):
                if (student['student_id'], period_id) in present_map:
                    student_row["status"].append("Present")
                    subject_name = report['periods'][i]
                    subject_attendance_count[subject_name] += 1
                else:
                    student_row["status"].append("Absent")

            report["students"].append(student_row)

        percentages = {}
        if total_students > 0:
            for subject, count in subject_attendance_count.items():
                percentages[subject] = round((count / total_students) * 100, 2)

        report["percentages"] = percentages
        conn.close()
        return report

    except Exception as e:
        print(f"Error generating report: {e}")
        return None


# --- Application Start ---
if __name__ == '__main__':
    get_face_data()
    try:
        eel.start('index.html', size=(1200, 800))
    except (SystemExit, MemoryError, KeyboardInterrupt):
        print("Application closed.")
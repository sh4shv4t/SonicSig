from flask import Flask, render_template, request, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename
from recognizer import recognize_song

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flashing messages

# Configure upload folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
DATABASE_FOLDER = os.path.join(UPLOAD_FOLDER, 'database')
QUERY_FOLDER = os.path.join(UPLOAD_FOLDER, 'queries')
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

# Create upload directories if they don't exist
os.makedirs(DATABASE_FOLDER, exist_ok=True)
os.makedirs(QUERY_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATABASE_FOLDER'] = DATABASE_FOLDER
app.config['QUERY_FOLDER'] = QUERY_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['QUERY_FOLDER'], filename)
            file.save(filepath)
            
            # Perform song recognition
            result = recognize_song(filepath)
            
            # Delete the query file after processing
            os.remove(filepath)
            
            return render_template('result.html', result=result)
    
    # Get list of songs in database
    database_songs = []
    for file in os.listdir(app.config['DATABASE_FOLDER']):
        if allowed_file(file):
            database_songs.append(file)
    
    return render_template('index.html', database_songs=database_songs)

@app.route('/upload_reference', methods=['POST'])
def upload_reference():
    if 'reference_file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('index'))
    
    file = request.files['reference_file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['DATABASE_FOLDER'], filename)
        file.save(filepath)
        flash(f'Successfully added {filename} to database')
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
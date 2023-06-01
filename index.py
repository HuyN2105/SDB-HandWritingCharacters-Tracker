from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
from waitress import serve

from main import process_img

app = Flask(__name__)
app.secret_key = 'secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB maximum file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # Check if the file is uploaded
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Check if the file has an allowed extension
        if not allowed_file(file.filename):
            flash('Invalid file extension')
            return redirect(request.url)

        # Check if there is already an uploaded image
        if 'uploaded_image' in request.files:
            flash('Only one image can be uploaded at a time')
            return redirect(request.url)

        # Save the uploaded image
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Process the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        result = process_img(image_path)

        # Display an alert based on the result
        flash(f'Processing result: {result}')

        return redirect(url_for('upload_image'))

    return render_template('index.html')

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)

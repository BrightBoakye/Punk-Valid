from flask import Flask, request, redirect, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from ViridAI import predict_single

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded_files'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    return render_template('upload.html')

@app.route('/uploader', methods=['POST', 'GET'])
def uploader():
    if request.method == "POST":
        try:
            f = request.files['file']
            
            # Check if the file has an allowed extension
            if '.' in f.filename and f.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
                num = len(os.listdir(app.config['UPLOAD_FOLDER']))
                file_format = f.filename.split('.')[-1]
                f.filename = f"image_{num}.{file_format}"

                filename = secure_filename(f.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                f.save(file_path)

                predicted_label, recommendation = predict_single(file_path)
                
                # Check if the land type is detected
                if predicted_label is None:
                    error_message = "Land type not detected. Please upload an image of a valid land type."
                    return render_template('uploader.html', error=error_message)
                
                return render_template('uploader.html', label=predicted_label, recommendation=recommendation)

            else:
                error_message = "Invalid file format. Please upload an image file (png, jpg, jpeg)."
                return render_template('upload.html', error=error_message)

        except Exception as e:
            print(str(e))
            return render_template('uploader.html', message="Error uploading files!")

    return redirect('/upload')

if __name__ == '__main__':
    app.run(debug=True)

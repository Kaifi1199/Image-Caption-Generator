from flask import Flask, render_template, request
from utils import extract_feature, generate_caption, load_model_assets
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model, tokenizer, max_length, cnn_model
model, tokenizer, max_length, cnn_model = load_model_assets()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    photo = extract_feature(filepath, cnn_model)
    caption = generate_caption(photo, model, tokenizer, max_length)

    return render_template('result.html', image_path=filepath, caption=caption)

if __name__ == '__main__':
    app.run(debug=True)

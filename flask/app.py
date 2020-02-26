from flask import Flask, render_template, request
from scripts.form import BasicForm, FindImageForm
import os
from scripts.ic import main, show_result
from scripts.sql import main as sql_main
from pynput.keyboard import Key, Listener
# import pickle 
# import random

# FILELIST = pickle.load(open('./static/resources/filelist.pkl', 'rb'))
app = Flask(__name__)
UPLOAD_FOLDER = './static/images/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'hard to guess string'

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')

''' BASIC '''
@app.route('/basic', methods=["GET", "POST"])
def basic():
    upload_form = BasicForm()

    filename = None

    if upload_form.validate_on_submit():
        image = upload_form.image.data
        if image:
            image.save(os.path.join(UPLOAD_FOLDER, image.filename))
            filename = image.filename
    return render_template('./basic/basic.html', upload_form=upload_form, filename=filename)

@app.route('/basic/<filename_flag>', methods=["GET", "POST"])
def predict(filename_flag):
    filename = filename_flag.split()[0]
    flag = filename_flag.split()[1]
    filepath = UPLOAD_FOLDER + filename
    if flag == 'ic':
        predict = main(filepath)
        return render_template('./basic/basic-predict.html', filename=filename, predict=predict)
    elif flag == 'vqa':
        question = request.form['user_input']
        predict, conf = show_result(filepath, question)
        predict = "{} with {}% confidence".format(predict, conf)
        return render_template('./basic/basic-predict-vqa.html', filename=filename, predict=predict, question=question)
    elif flag == 'again':
        upload_form = BasicForm()
        return render_template('./basic/basic.html', upload_form=upload_form, filename=filename)

    

def vqa_predict(filename):
    return filename

''' FIND IMAGE '''
@app.route('/find_image', methods=["GET", "POST"])
def find_image():
    form = FindImageForm()
    text = None
    images = None
    if form.validate_on_submit():
        text = form.user_input.data
        results = sql_main(text)
        images = results[:12]
    return render_template("./find/find_image.html", find_form=form, images=images)

'''SUPPORTER'''
@app.route('/support', methods=["GET", "POST"])
def support():
    upload_form = BasicForm()

    filename = None

    if upload_form.validate_on_submit():
        image = upload_form.image.data
        if image:
            image.save(os.path.join(UPLOAD_FOLDER, image.filename))
            filename = image.filename
    return render_template('./support/support.html', upload_form=upload_form, filename=filename)

@app.route('/support/<filename>')
def support_begin(filename):
    filepath = UPLOAD_FOLDER + filename
    command = ""
    # while command != 'end':
    command = check_key()
    if command == "Model 1":
        response = main(filepath)
    elif command == "Model 2":
        response = check_voice(filepath, you_said)
    elif command == "Model 3":
        response = check_voice(filepath, you_said, tag='position')
    elif command == 'end':
        response = "Good bye! Thanks you for asking"


    return render_template("./support/support-begin.html", filename=filename, you_said=you_said, response=response)

def check_key():
    command = ""
    def on_press(key):
        nonlocal command
        if key == Key.ctrl_l:
            command = "Model 1"
            return False
        elif key == Key.ctrl_r:
            command = "Model 2"
            return False
        elif key == Key.space:
            command = "Model 3"
            return False
        elif key == Key.esc:
            command = 'end'
            return False

    with Listener(on_press=on_press) as listener:
        listener.join()  
        
    return command




if __name__ == "__main__":
    app.run(debug=True, threaded=False)

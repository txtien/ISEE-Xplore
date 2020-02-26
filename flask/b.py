from flask import Flask, render_template, request
from scripts.form import BasicForm, FindImageForm
from scripts.speech_text import recognize_speech_from_mic
from scripts.text_speech import speak
import os
import speech_recognition as sr
from scripts.ic import main, show_result, finding
from scripts.sql import main as sql_main
from pynput.keyboard import Key, Listener

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
    speak("Hello! Wish you have a good day!").play()
    filepath = UPLOAD_FOLDER + filename
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    response = None
    you_said = None
    command = ""
    # while command != 'end':
    command = check_key()
    if command == "Model 1":
        response = main(filepath)
    elif command == "Model 2":
        # speak("What do you want to know?").play()
        you_said = recognize_speech_from_mic(recognizer, microphone, tag='model2')
        response = check_voice(filepath, you_said)
    elif command == "Model 3":
        # speak("What do you want to find?").play()
        you_said = recognize_speech_from_mic(recognizer, microphone, tag='model3')
        response = check_voice(filepath, you_said, tag='position')
    elif command == 'end':
        response = "Good bye! Thanks you for asking"

    speak(response).play()

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

def check_voice(filepath, you_said, tag=None):
    if you_said:
        if tag == 'position':
            response = show_result(filepath, you_said, tag='position')
        else:
            response, conf = show_result(filepath, you_said)
    else:
        response = "Sorry? Can you ask again?"
    return response

"""FINDER"""
@app.route('/finder', methods=["GET", "POST"])
def finder():
    form = FindImageForm()
    filepaths = ""
    home = './static/images/home/'
    filenames = os.listdir(home)
    filepaths = [home + filename for filename in filenames]
    if form.validate_on_submit():
        text = form.user_input.data
        speak("Wait me!").play()
        room_type, is_object_here = finding(filepaths, text)
        return render_template('./finder/found.html', filepaths=filepaths, obj=text, room_type=room_type, is_object_here=is_object_here, n=len(room_type))

    return render_template('./finder/finder.html', find_form=form, filepaths=filepaths)

if __name__ == "__main__":
    app.run(debug=True, threaded=False)

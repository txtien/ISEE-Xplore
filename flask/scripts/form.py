from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField
from wtforms.validators import DataRequired

class BasicForm(FlaskForm):
    image = FileField("Upload your image", validators=[DataRequired()])
    submit = SubmitField("Upload")
    

class FindImageForm(FlaskForm):
    user_input = StringField("What you want to find ?", validators=[DataRequired()])
    find = SubmitField("Find")
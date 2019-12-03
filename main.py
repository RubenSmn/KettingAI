# import rasa
from rasa.nlu.training_data import load_data
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Trainer
from rasa.nlu import config
from rasa.nlu.model import Metadata, Interpreter

# Import speech_recognition
import speech_recognition as sr

# This will load the nlu data in the md file, train a model and save it as the current model
# loading the nlu training samples
training_data = load_data("./data/nlu.md")
# trainer to educate our pipeline
trainer = Trainer(config.load("config.yml"))
# train the model!
interpreter = trainer.train(training_data)
# store it for future use
model_directory = trainer.persist("./models", fixed_model_name="current")

# Use this line when you already trained a model
# interpreter = Interpreter.load('./models/current')


# small helper to make dict dumps a bit prettier
def pprint(o):
   print(json.dumps(o, indent=2))

r = sr.Recognizer()

with sr.Microphone() as source:
    while True:
        print("Say something")
        audio = r.listen(source)

        with open("microphone-results.wav", "wb") as f:
            f.write(audio.get_wav_data())

        try:
            text = r.recognize_google(audio, language='nl-NL')
            prediction = interpreter.parse(text)
            pprint(prediction)
        except Exception as e:
            print(e)

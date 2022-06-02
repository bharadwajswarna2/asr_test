from flask import Flask, redirect, render_template, request
import speech_recognition as sr
import intent_hf
from intent_hf import LiveWav2Vec2

app = Flask(__name__)

@app.route("/")

def main_1():

    print("Live ASR")
    #indian_english = "dharmesh8b/indian-accent-english-asr"
    #english_model = "facebook/wav2vec2-base-960h"
    hindi_model = "Harveenchadha/hindi_model_with_lm_vakyansh"
    #english_vak = "Harveenchadha/vakyansh-wav2vec2-indian-english-enm-700"
    asr = LiveWav2Vec2(hindi_model)
    
    asr.start()
    #text_final = ""
    #try:        
    while True:
        text = asr.get_last_text()
        #text_final =  text
    
        print(text) #render_template('index.html',transcript = text)

            #text,sample_length,inference_time = asr.get_last_text()                        
            #print(f"{sample_length:.3f}s\t{inference_time:.3f}s\t{text}")
            
    #except KeyboardInterrupt:
    #    asr.stop()  
    #    exit()

if __name__ == "__main__":
    app.run(debug=True, threaded = True)



from flask import Flask, request, send_file
from flask_cors import CORS
import os
import mido
from mido import Message, MidiFile, MidiTrack, bpm2tempo
import numpy
from model_functions import TransformerEncoder
import torch

toggle_twinkle = True

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

model = TransformerEncoder(ntoken=150,
                           em_dim=64,
                           nhead=8,
                           nhid=128,
                           nlayers=8,
                           max_len=128,
                           dropout=0.2)

model_path = "mint_ii_model_epoch_50.pth"
trained = torch.load(model_path, map_location=torch.device("cpu"))
state_dict = trained['model_state_dict']
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

model.eval()
print("Model loaded successfully.")


@app.route('/')
def home():
    return "MINTii backend is running."


@app.route("/generate", methods=["POST"])
def generate():
    global toggle_twinkle

    key = request.json.get("key", "C")
    tempo = int(request.json.get("tempo", 120))

    if toggle_twinkle:
        create_twinkle_midi("melody.mid", tempo)
    else:
        create_twinkle_part2_midi("melody.mid", tempo)

    toggle_twinkle = not toggle_twinkle
    return send_file("melody.mid", as_attachment=True)


def create_twinkle_midi(filename, bpm):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    tempo = bpm2tempo(bpm)
    track.append(Message('program_change', program=0, time=0))
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    # Note values
    notes = [60, 60, 67, 67, 69, 69, 67]  # C C G G A A G
    durations = [480, 480, 480, 480, 480, 480, 960]  # last G is a half note

    for note, dur in zip(notes, durations):
        track.append(Message('note_on', note=note, velocity=64, time=0))
        track.append(Message('note_off', note=note, velocity=64, time=dur))

    mid.save(filename)

def create_twinkle_part2_midi(filename, bpm):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    tempo = bpm2tempo(bpm)
    track.append(Message('program_change', program=0, time=0))
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    notes = [65, 65, 64, 64, 62, 62, 60]  # F F E E D D C
    durations = [480, 480, 480, 480, 480, 480, 960]

    for note, dur in zip(notes, durations):
        track.append(Message('note_on', note=note, velocity=64, time=0))
        track.append(Message('note_off', note=note, velocity=64, time=dur))

    mid.save(filename)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

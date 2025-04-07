from flask import Flask, request, send_file, send_from_directory
import os
import torch
import numpy as np
from generator_functions import generate_music
from model_functions import TransformerEncoder
from mint_ii_functions import (
    beat_unmapping,
    decoder,
    BEAT_CNT_OFFSET,
    POSITION_OFFSET,
    INTERVAL_OFFSET,
    DURATION_OFFSET,
    TRACK_END_TOKEN
)

app = Flask(__name__)

model = TransformerEncoder(
    ntoken=150,
    em_dim=64,
    nhead=8,
    nhid=128,
    nlayers=8,
    max_len=128,
    dropout=0.2
)

model_path = "mint_ii_pop_melodies_500.pth"
trained = torch.load(model_path, map_location=torch.device("cpu"))
state_dict = trained['model_state_dict']
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()
print("Model loaded successfully.")

def get_random_seed(seed_folder="seeds", length=16):
    seed_files = [f for f in os.listdir(seed_folder) if f.endswith(".npy")]
    if not seed_files:
        print("No seed files found.")
        return []
    path = os.path.join(seed_folder, np.random.choice(seed_files))
    return np.load(path).astype(int).tolist()[:length]

def trim_leading_empty_beats(tokens, beatsteps_per_beat=12):
    i = 0
    while i + 3 < len(tokens):
        beat = tokens[i:i+4]
        types = [BEAT_CNT_OFFSET, POSITION_OFFSET, INTERVAL_OFFSET, DURATION_OFFSET]
        if all(token in types for token in beat):
            i += 4
        else:
            break
    return tokens[i:]

def trim_to_n_bars(tokens, bars):
    beatsteps_per_beat = 12
    max_beatstep = bars * 4 * beatsteps_per_beat - 1
    tokens = tokens.copy()

    trimmed = []
    i = 0
    current_beat = 0
    current_position = 0

    while i < len(tokens):
        token = tokens[i]
        trimmed.append(token)

        if token == BEAT_CNT_OFFSET:
            current_beat += 1
            current_position = 0
            i += 1

        elif POSITION_OFFSET <= token < INTERVAL_OFFSET:
            current_position = token - POSITION_OFFSET
            i += 1

        elif INTERVAL_OFFSET <= token < DURATION_OFFSET:
            i += 1
            if i >= len(tokens):
                break

            duration_token = tokens[i]
            duration = duration_token - DURATION_OFFSET
            start_beatstep = current_beat * beatsteps_per_beat + current_position
            end_beatstep = start_beatstep + duration

            if end_beatstep > max_beatstep:
                trimmed.pop()
                trimmed.pop()
                break

            trimmed.append(duration_token)
            i += 1
        else:
            i += 1

    return trimmed

def generate_melody(model, key, tempo, bars=8, k=10, seed_folder="seeds"):
    print("Starting melody generation...")
    seed = get_random_seed(seed_folder)
    if not seed:
        print("ERROR: No seed returned.")
        return None, False

    print(f"Seed loaded: {seed[:8]}... (length={len(seed)})")
    current_tokens = seed.copy()

    loop_count = 0
    while True:
        loop_count += 1
        print(f"--- Generation loop #{loop_count} ---")

        generated = generate_music(
            model=model,
            seed_sequence=current_tokens,
            max_length=128,
            strategy="top-k",
            k=k
        )

        if not generated or not isinstance(generated, tuple) or len(generated) < 3:
            print("ERROR: Unexpected format from generate_music()")
            return None, False

        print("Generated tokens:", generated[2][:8], "...")

        try:
            expanded = beat_unmapping(generated[2])
            print(f"Expanded tokens length: {len(expanded)}")
        except Exception as e:
            print("ERROR in beat_unmapping:", e)
            return None, False

        trimmed_start = trim_leading_empty_beats(expanded)
        print(f"Trimmed start length: {len(trimmed_start)}")

        total_beats = sum(1 for t in trimmed_start if t == BEAT_CNT_OFFSET)
        print(f"Detected beats: {total_beats}")

        if total_beats >= bars * 4:
            final_tokens = trim_to_n_bars(trimmed_start, bars)
            print(f"Final tokens length: {len(final_tokens)}")
            break

        print("Sequence too short, generating more...")
        current_tokens = trimmed_start

    output_path = "melody.mid"
    if final_tokens[-1] != TRACK_END_TOKEN:
        final_tokens.append(TRACK_END_TOKEN)
        print("Appended TRACK_END_TOKEN")

    try:
        decoder(tempo, key, final_tokens, output_path)
        print("MIDI successfully decoded and saved.")
    except Exception as e:
        print("ERROR in decoder:", e)
        return None, False

    return output_path, True


@app.route("/")
def index():
    return send_from_directory("static", "mint_ii.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/generate", methods=["POST"])
def generate():
    try:
        key = int(request.json.get("key", 0))
        tempo = int(request.json.get("tempo", 120))
        print(f"Received generate request: key={key}, tempo={tempo}")

        output_path, success = generate_melody(model, key, tempo, bars=8)
        if not success:
            print("Generation failed — no success flag.")
            return "Failed to generate", 500

        print(f"Generated file at {output_path}")
        return send_file(output_path, as_attachment=True)

    except Exception as e:
        print(f"Error in /generate: {e}")
        return "Internal Server Error", 500


@app.route("/melody.mid")
def serve_midi_file():
    return send_from_directory(".", "melody.mid", as_attachment=True)
  

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

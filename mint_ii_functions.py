from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
from collections import defaultdict

# ------------------ USER SPECIFIED  ------------------ 

int_max = 24
int_min = -47

dur_map = [(24, 1), (48, 3), (96, 6), (192, 12), (384, 24)]

beatstep_granularity = 12

beat_map = [1, 2, 4, 8, 16, 32, 64, 128]



# ---------------- NOT USER SPECIFIED  ---------------- 

key_map = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 58, 59]

int_count = int_max - int_min + 1 # the +1 is to include 0

dur_count = dur_map[0][0] // dur_map[0][1]
if len(dur_map) > 1:
    for i in range(1, len(dur_map)):
        dur_count += (dur_map[i][0] - dur_map[i-1][0]) // dur_map[i][1]


# token offsets
BEAT_CNT_OFFSET = 1
POSITION_OFFSET = BEAT_CNT_OFFSET + int(len(beat_map))
INTERVAL_OFFSET = POSITION_OFFSET + beatstep_granularity
DURATION_OFFSET = INTERVAL_OFFSET + int(int_count)
TRACK_END_TOKEN = DURATION_OFFSET + int(dur_count)

print(f"Offsets: {BEAT_CNT_OFFSET}, {POSITION_OFFSET}, {INTERVAL_OFFSET}, {DURATION_OFFSET}, {TRACK_END_TOKEN}")


def duration_to_token(beatsteps):
    if beatsteps > dur_map[-1][0]:
        beatsteps = dur_map[-1][0]  # Clamp duration to max allowed range

    bin_offset = DURATION_OFFSET  # Track where each bin starts in token space
    last_cap = 0

    for bin_cap, step_size in dur_map:
        bin_range = (bin_cap - last_cap) // step_size

        if last_cap <= beatsteps < bin_cap + last_cap:
            quantized_duration = ((beatsteps - last_cap) // step_size) * step_size
            token = bin_offset + (quantized_duration // step_size)
            return token - 1

        last_cap = bin_cap
        bin_offset += bin_range

    print("Error in 'quantize_and_tokenize_duration': Duration out of range.")
    return bin_offset - 1

def beat_mapping(tokens):
    new_tokens = []
    
    # Step 1: Split the token sequence into alternating groups of beat tokens and non-beat tokens
    grouped_sequences = []
    current_group = []
    is_beat_group = BEAT_CNT_OFFSET <= tokens[0] < POSITION_OFFSET  # Check if first token is a beat
    
    for token in tokens:
        if (BEAT_CNT_OFFSET <= token < POSITION_OFFSET) == is_beat_group:
            current_group.append(token)
        else:
            grouped_sequences.append(current_group)
            current_group = [token]
            is_beat_group = not is_beat_group  # Toggle group type

    if current_group:
        grouped_sequences.append(current_group)

    # Step 2: Process each sequence separately
    for group in grouped_sequences:
        if len(group) == 1 and BEAT_CNT_OFFSET <= group[0] < POSITION_OFFSET:
            # Skip single 1s, since they remain a 1
            new_tokens.append(group[0])
            continue

        if BEAT_CNT_OFFSET <= group[0] < POSITION_OFFSET:  # It's a beat sequence
            count = len(group)
            
            # While there are beats left to process
            while count > 0:
                for i in reversed(range(len(beat_map))):
                    val = beat_map[i]
                    if count >= val:
                        new_tokens.append(BEAT_CNT_OFFSET + i)
                        count -= val
                        break

        else:  # Non-beat tokens, just append them
            new_tokens.extend(group)

    return new_tokens

def shift_to_interval_bounds(beatstep_dict, key_root):    
    max_shifts = max(1, (int_max - int_min) // 12)
    
    max_pitch = key_root + int_max
    min_pitch = key_root + int_min
    
    tries = 0

    while tries < max_shifts + 1:
        all_pitches = [pitch for notes in beatstep_dict.values() for (pitch, _) in notes]
        low = min(all_pitches) < min_pitch
        high = max(all_pitches) > max_pitch

        if not (low or high):
            return beatstep_dict

        shift = 12 if low else -12

        for beatstep in beatstep_dict:
            beatstep_dict[beatstep] = [(pitch + shift, dur) for pitch, dur in beatstep_dict[beatstep]]

        tries += 1

    raise ValueError("Pitch range could not be shifted into allowed bounds.")


def token_to_duration(token):
    last_cap = 0  # Tracks cumulative bin offset
    bin_offset = DURATION_OFFSET  # Tracks where each bin starts in token space
    
    token += 1

    for bin_cap, step_size in dur_map:
        bin_range = (bin_cap - last_cap) // step_size
        bin_end = bin_offset + bin_range  # Where this bin ends in token space

        if bin_offset < token <= bin_end:
            duration_within_bin = (token - bin_offset) * step_size
            return last_cap + duration_within_bin

        last_cap = bin_cap
        bin_offset += bin_range

    # If something went really wrong, return max duration
    print(f"Error: Could not decode duration token {token}, using max duration.")
    return dur_map[-1][0]

def beat_unmapping(tokens):
    
    token_to_beat = {BEAT_CNT_OFFSET + i: val for i, val in enumerate(beat_map)}
    
    new_tokens = []

    # Step 1: Split the token sequence into alternating groups of beat tokens and non-beat tokens
    grouped_sequences = []
    current_group = []
    is_beat_group = tokens[0] in token_to_beat  # Check if first token is a beat token

    for token in tokens:
        if (token in token_to_beat) == is_beat_group:
            current_group.append(token)
        else:
            grouped_sequences.append(current_group)
            current_group = [token]
            is_beat_group = not is_beat_group  # Toggle group type

    if current_group:
        grouped_sequences.append(current_group)

    # Step 2: Process each sequence separately
    for group in grouped_sequences:
        if group[0] in token_to_beat:  # It's a beat sequence
            for token in group:
                new_tokens.extend([1] * token_to_beat[token])  # Expand each beat token into 1s
        else:  # Non-beat tokens, just append them
            new_tokens.extend(group)

    return new_tokens

def encoder(midi_file_path, key):    
    tokens = []
    last_beat = None
    key_root = key_map[key]  # Convert key (0-11) to reference MIDI pitch
    
    # print(key_root)

    # Load MIDI file
    midi = MidiFile(midi_file_path)
    ppq = midi.ticks_per_beat
    beatstep_length = ppq // beatstep_granularity

    beatstep_dict = defaultdict(list)

    # Track note-on times per pitch
    note_on_times = defaultdict(list)

    for track in midi.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                note_on_times[msg.note].append(abs_time)

            elif msg.type in ["note_off", "note_on"] and msg.velocity == 0:
                if note_on_times[msg.note]:
                    start_time = note_on_times[msg.note].pop(0)
                    duration_ticks = abs_time - start_time

                    beatstep = round(start_time / beatstep_length)
                    duration_beatsteps = max(1, round(duration_ticks / beatstep_length))

                    beatstep_dict[beatstep].append((msg.note, duration_beatsteps))
    
    # this requires certain things from the dataset or it will break
    # the pitch range must be able to be transposed by octaves into the specified range
    # additionally, every interval in the specified range of tookens must be included in the dataset
    beatstep_dict = shift_to_interval_bounds(beatstep_dict, key_root)

    # Encode tokens
    for beatstep in sorted(beatstep_dict.keys()):
        notes_in_beatstep = sorted(beatstep_dict[beatstep], key=lambda x: x[0])  # Sort by pitch (low to high)

        # **Determine the current beat and position**
        current_beat = beatstep // beatstep_granularity
        position = beatstep % beatstep_granularity  # Get the position within the beat (0-23)

        # **Insert beat tokens if changed**
        if last_beat is None:
            for _ in range(current_beat + 1):  
                tokens.append(BEAT_CNT_OFFSET)

        elif current_beat > last_beat:
            for _ in range(current_beat - last_beat):
                tokens.append(BEAT_CNT_OFFSET)

        last_beat = current_beat

        # **Insert position token**
        tokens.append(POSITION_OFFSET + position)

        for pitch, duration_beatsteps in notes_in_beatstep: 
            interval = pitch - key_root
            
            duration_token = duration_to_token(duration_beatsteps)
        
            tokens.extend([
                INTERVAL_OFFSET + interval - int_min,
                duration_token
            ])

    # **Apply beat compression to optimize sequence length**
    tokens = beat_mapping(tokens)
    
    tokens.append(TRACK_END_TOKEN)

    return tokens

def decoder(bpm, key, tokens, output_path):
    tokens = beat_unmapping(tokens)  # Expand compressed beats
    ppq = 480  # Standard MIDI resolution
    beatstep_length = ppq // beatstep_granularity
    tempo = bpm2tempo(bpm)
    
    key_root = key_map[key]
    
    mid = MidiFile(ticks_per_beat=ppq)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

    current_beat = 0
    current_position = 0
    events = []

    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == BEAT_CNT_OFFSET:  # BEAT_START
            current_beat += 1
            current_position = 0
            i += 1
            continue

        elif POSITION_OFFSET <= token < INTERVAL_OFFSET:
            current_position = token - POSITION_OFFSET
            i += 1
            continue

        elif INTERVAL_OFFSET <= token < DURATION_OFFSET:  # INTERVAL TOKENS
            interval = token - INTERVAL_OFFSET + int_min
            pitch = key_root + interval
            # print(f"DECODER: token={token}, interval={interval}, key_root={key_root}, pitch={pitch}")
            i += 1
            
            if i >= len(tokens):
                print("Error: Unexpected end of tokens after interval.")
                break

            duration_token = tokens[i]
            duration_beatsteps = token_to_duration(duration_token)

            start_tick = (current_beat * beatstep_granularity + current_position) * beatstep_length
            duration_ticks = max(1, duration_beatsteps * beatstep_length)
            end_tick = start_tick + duration_ticks

            events.append((start_tick, 'on', pitch))
            events.append((end_tick, 'off', pitch))

            i += 1
            
        elif token == TRACK_END_TOKEN:
            break

        else:
            print(f"Unknown token {token} at index {i}, skipping.")
            i += 1

    # Sort events by tick time
    events.sort()
    
    last_tick = 0
    for tick, event_type, pitch in events:
        delta = tick - last_tick
        last_tick = tick

        if event_type == 'on':
            track.append(Message('note_on', note=pitch, velocity=100, time=delta))
        else:
            track.append(Message('note_off', note=pitch, velocity=0, time=delta))

    track.append(MetaMessage('end_of_track', time=0))
    
    mid.save(output_path)
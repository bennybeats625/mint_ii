let isLoaded = false;
let part;

const genBtn = document.getElementById("generate");
const playBtn = document.getElementById("playstop");
const downloadBtn = document.getElementById("download");


async function loadMidi() {
  genBtn.innerText = "Loading...";

  const response = await fetch("/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ key: "0", tempo: 120 }),
  });

  const arrayBuffer = await response.arrayBuffer();
  const midi = new Midi(arrayBuffer);
  const synth = new Tone.PolySynth().toDestination();

const events = (midi.tracks[0]?.notes || []).map((note) => ({
    time: note.time,
    note: note.name,
    duration: note.duration,
    velocity: note.velocity,
  }));

  part = new Tone.Part((time, value) => {
    synth.triggerAttackRelease(value.note, value.duration, time, value.velocity);
  }, events).start(0);

  part.loop = true;
  // Force loop to exactly 8 bars (4 beats per bar), regardless of MIDI duration
  const beatsPerBar = 4;
  const bars = 8;
  const bpm = midi.header.tempos[0]?.bpm || 120;
    
  const secondsPerBeat = 60 / bpm;
  const loopLength = beatsPerBar * bars * secondsPerBeat;
    
  part.loopEnd = loopLength;


  Tone.Transport.bpm.value = bpm;
  isLoaded = true;

  // Enable all buttons once MIDI is generated and ready to play
  document.getElementById("playstop").disabled = false;
  document.getElementById("download").disabled = false;

  // Auto-start playback
  Tone.Transport.start("+0.1");

  // Set play/stop button to "Stop"
  document.getElementById("playstop").innerText = "Stop";
}

document.getElementById("generate").addEventListener("click", async () => {

  // Disable all interaction
  genBtn.disabled = true;
  playBtn.disabled = true;
  downloadBtn.disabled = true;

  // Stop playback if playing
  Tone.Transport.stop();
  Tone.Transport.position = 0;

  if (part) {
    part.dispose();
    part = null;
  }

  isLoaded = false;

  await loadMidi(); // fetch + decode + build + start

  // Enable interaction again
  genBtn.innerText = "Regenerate";
  genBtn.disabled = false;
  playBtn.disabled = false;
  downloadBtn.disabled = false;
});



document.getElementById("playstop").addEventListener("click", () => {
  if (!isLoaded) return;

  if (Tone.Transport.state === "started") {
    Tone.Transport.stop();
    Tone.Transport.position = 0;
    document.getElementById("playstop").innerText = "Play";
  } else {
    Tone.Transport.start("+0.1");
    document.getElementById("playstop").innerText = "Stop";
  }
});

document.getElementById("download").addEventListener("click", () => {
  window.location.href = "/melody.mid";
});

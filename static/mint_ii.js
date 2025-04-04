let isLoaded = false;
let part;

async function loadMidi() {
  const response = await fetch("https://mintii-music-generator.onrender.com/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      key: 0, // hardcoded for now—you can add dropdowns later
      tempo: 120,
    }),
  });

  const arrayBuffer = await response.arrayBuffer();
  const midi = new Midi(arrayBuffer);

  const synth = new Tone.PolySynth().toDestination();

  const events = midi.tracks[0].notes.map((note) => ({
    time: note.time,
    note: note.name,
    duration: note.duration,
    velocity: note.velocity,
  }));

  part = new Tone.Part((time, value) => {
    synth.triggerAttackRelease(
      value.note,
      value.duration,
      time,
      value.velocity,
    );
  }, events).start(0);

  part.loop = true;
  part.loopEnd = midi.duration;

  isLoaded = true;
  document.getElementById("download").disabled = false;
  Tone.Transport.bpm.value = midi.header.tempos[0]?.bpm || 120;
}

document.getElementById("play").addEventListener("click", async () => {
  if (!isLoaded) {
    await loadMidi();
  }
  Tone.Transport.start("+0.1");
});

document.getElementById("stop").addEventListener("click", () => {
  Tone.Transport.stop();
  Tone.Transport.position = 0;
});

document.getElementById("download").addEventListener("click", () => {
  window.location.href = "https://mintii-music-generator.onrender.com/melody.mid";
});

document.getElementById("regenerate").addEventListener("click", async () => {
  // Stop playback cleanly
  Tone.Transport.stop();
  Tone.Transport.position = 0;

  // Clear the old part (so we don't stack multiple loops)
  if (part) {
    part.dispose();
    part = null;
  }

  // Reset load flag
  isLoaded = false;

  // Fetch and play the new melody
  await loadMidi();
  Tone.Transport.start("+0.1");
});

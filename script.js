
"use strict";

// ==========================================
// ANIMALSOUND - ANIMAL SOUND RECOGNITION
// Uses MediaPipe AudioClassifier + YAMNet
// ==========================================

const $ = (id) => document.getElementById(id);

// HTML elements
const audioUpload = $("audioUpload");
const fileName = $("fileName");
const audioSection = $("audioSection");
const audioPlayer = $("audioPlayer");
const resultsSection = $("resultsSection");

const waveformCanvas = $("waveformCanvas");
const fftCanvas = $("fftCanvas");

const animalResult = $("animalResult");
const frequencyResult = $("frequencyResult");
const confidenceResult = $("confidenceResult");
const probabilityBars = $("probabilityBars");

// Global variables
let audioContext = null;
let audioClassifier = null;
let classifierPromise = null;
let audioObjectUrl = null;

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@0.10.14/wasm";

const ANIMAL_KEYWORDS = {
  Cat: [
    "cat", "meow", "purr", "caterwaul"
  ],
  Dog: [
    "dog", "bark", "bow-wow", "growling",
    "whimper", "yip", "howl"
  ],
  Cow: [
    "cattle", "moo", "cow"
  ],
  Frog: [
    "frog", "croak"
  ],
  Crow: [
    "crow", "caw"
  ],
  "Chirping Birds": [
    "bird", "chirp", "tweet", "birdsong",
    "bird song", "bird vocalization",
    "owl", "duck", "goose", "chicken"
  ]
};

// Start listening for uploads
if (audioUpload) {
  audioUpload.addEventListener("change", handleAudioUpload);
} else {
  console.error('Missing HTML element: "audioUpload".');
}

// ==========================================
// STATUS AND ERROR MESSAGES
// ==========================================

function setStatus(message) {
  let status = $("analysisStatus");

  if (!status && resultsSection) {
    status = document.createElement("p");
    status.id = "analysisStatus";
    status.setAttribute("role", "status");
    resultsSection.appendChild(status);
  }

  if (status) status.textContent = message;
}

function showError(message) {
  console.error("AnimalSound:", message);
  setStatus(message);
}

// ==========================================
// LOAD MEDIAPIPE AND YAMNET
// ==========================================

async function getAudioClassifier() {
  if (audioClassifier) return audioClassifier;
  if (classifierPromise) return classifierPromise;

  classifierPromise = (async () => {
    // The HTML must load audio_bundle.js before script.js.
    const resolver =
      window.FilesetResolver ||
      window.TasksAudio?.FilesetResolver;

    const classifierClass =
      window.AudioClassifier ||
      window.TasksAudio?.AudioClassifier;

    if (!resolver || !classifierClass) {
      throw new Error(
        "MediaPipe did not load. Check the audio_bundle.js script in index.html."
      );
    }

    setStatus("Loading the YAMNet model. Please wait...");

    const fileset = await resolver.forAudioTasks(WASM_URL);

    const classifier =
      await classifierClass.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: MODEL_URL
        },
        maxResults: 521,
        scoreThreshold: 0
      });

    audioClassifier = classifier;
    return classifier;
  })();

  try {
    return await classifierPromise;
  } catch (error) {
    classifierPromise = null;
    throw error;
  }
}

// ==========================================
// PROCESS UPLOADED AUDIO
// ==========================================

async function handleAudioUpload(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  if (file.size === 0) {
    showError("The selected audio file is empty.");
    return;
  }

  if (fileName) {
    fileName.textContent = "Selected: " + file.name;
  }

  audioSection?.classList.remove("hidden");
  resultsSection?.classList.remove("hidden");

  if (audioObjectUrl) {
    URL.revokeObjectURL(audioObjectUrl);
  }

  audioObjectUrl = URL.createObjectURL(file);

  if (audioPlayer) {
    audioPlayer.src = audioObjectUrl;
    audioPlayer.load();
  }

  if (animalResult) animalResult.textContent = "Analyzing...";
  if (confidenceResult) confidenceResult.textContent = "Please wait";
  if (frequencyResult) frequencyResult.textContent = "Calculating...";

  if (probabilityBars) probabilityBars.replaceChildren();

  const labelsBox = $("modelLabels");
  if (labelsBox) labelsBox.textContent = "";

  setStatus("Reading your audio file...");

  try {
    const AudioContextClass =
      window.AudioContext || window.webkitAudioContext;

    if (!AudioContextClass) {
      throw new Error("This browser does not support Web Audio.");
    }

    if (!audioContext || audioContext.state === "closed") {
      audioContext = new AudioContextClass();
    }

    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }

    const bytes = await file.arrayBuffer();
    const buffer = await audioContext.decodeAudioData(bytes.slice(0));

    if (!buffer || buffer.length < 2) {
      throw new Error("The audio could not be decoded.");
    }

    const mono = getMonoAudio(buffer);

    // Draw waveform and calculate FFT independently
    drawWaveform(mono, buffer.sampleRate);

    const fftResult = performFFT(mono, buffer.sampleRate);

    drawFrequencySpectrum(
      fftResult.frequencies,
      fftResult.magnitudes
    );

    const dominantFrequency = findDominantFrequency(
      fftResult.frequencies,
      fftResult.magnitudes
    );

    if (frequencyResult) {
      frequencyResult.textContent =
        dominantFrequency > 0
          ? dominantFrequency.toFixed(2) + " Hz"
          : "Not available";
    }

    // Classify the sound
    setStatus("Loading the model and classifying the sound...");

    const classifier = await getAudioClassifier();

    // MediaPipe expects raw Float32 audio samples and sample rate.
    // Very long recordings are limited to the first 15 seconds.
    const maxSamples = Math.min(
      mono.length,
      Math.floor(buffer.sampleRate * 15)
    );

    const audioClip = mono.slice(0, maxSamples);

    const results = classifier.classify(
      audioClip,
      buffer.sampleRate
    );

    console.log("YAMNet classification results:", results);

    const categories = extractCategories(results);

    if (!categories.length) {
      throw new Error(
        "The model returned no categories. Try another clear recording."
      );
    }

    const prediction = mapCategoriesToAnimals(categories);

    renderPrediction(prediction);

    if (prediction.animal === "Unknown") {
      setStatus(
        "Analysis finished, but no supported animal label was found. See the top model labels below."
      );
    } else {
      setStatus(
        "Analysis complete. The prediction is an estimate, not guaranteed identification."
      );
    }
  } catch (error) {
    console.error("AnimalSound analysis error:", error);

    if (animalResult) animalResult.textContent = "Analysis failed";
    if (confidenceResult) confidenceResult.textContent = "Unavailable";

    showError(
      "Could not analyze the audio: " +
      (error?.message || String(error)) +
      " Check your internet connection and browser console."
    );
  }
}

// ==========================================
// READ MODEL RESULTS SAFELY
// ==========================================

function extractCategories(results) {
  if (!Array.isArray(results) || results.length === 0) {
    return [];
  }

  // Standard MediaPipe AudioClassifier output
  const classifications = results[0]?.classifications;

  if (Array.isArray(classifications)) {
    return classifications.flatMap(
      (item) => item?.categories || []
    );
  }

  // Fallback for alternative result wrappers
  return results[0]?.categories || [];
}

// ==========================================
// CONVERT STEREO TO MONO
// ==========================================

function getMonoAudio(buffer) {
  const channelCount = buffer.numberOfChannels;
  const mono = new Float32Array(buffer.length);

  for (let channel = 0; channel < channelCount; channel++) {
    const samples = buffer.getChannelData(channel);

    for (let i = 0; i < samples.length; i++) {
      mono[i] += samples[i] / channelCount;
    }
  }

  return mono;
}

// ==========================================
// MAP YAMNET LABELS TO ANIMALS
// ==========================================

function mapCategoriesToAnimals(categories) {
  const scores = Object.fromEntries(
    Object.keys(ANIMAL_KEYWORDS).map(
      (animal) => [animal, 0]
    )
  );

  const labels = categories
    .map((category) => ({
      label: String(
        category.categoryName ||
        category.displayName ||
        ""
      ),
      score: Number(category.score) || 0
    }))
    .filter((item) => item.label && item.score >= 0)
    .sort((a, b) => b.score - a.score);

  for (const item of labels) {
    const label = item.label.toLowerCase();

    for (const [animal, keywords] of Object.entries(ANIMAL_KEYWORDS)) {
      const matches = keywords.some((keyword) => {
        return label === keyword || label.includes(keyword);
      });

      if (matches) {
        scores[animal] = Math.max(scores[animal], item.score);
      }
    }
  }

  const ranked = Object.entries(scores).sort(
    (a, b) => b[1] - a[1]
  );

  const [animal, score] = ranked[0] || ["Unknown", 0];

  // Avoid declaring an animal if the model has no matching label.
  const identified = score > 0;

  const probabilities = Object.fromEntries(
    Object.entries(scores).map(([name, value]) => [
      name,
      value * 100
    ])
  );

  return {
    animal: identified ? animal : "Unknown",
    confidence: identified ? score * 100 : 0,
    probabilities,
    topLabels: labels.slice(0, 8)
  };
}

// ==========================================
// DISPLAY PREDICTION AND MODEL LABELS
// ==========================================

function renderPrediction(prediction) {
  if (animalResult) {
    animalResult.textContent = prediction.animal;
  }

  if (confidenceResult) {
    confidenceResult.textContent =
      prediction.animal === "Unknown"
        ? "Not available"
        : prediction.confidence.toFixed(1) + "% model score";
  }

  displayProbabilities(prediction.probabilities);

  let labelsBox = $("modelLabels");

  if (!labelsBox && probabilityBars?.parentElement) {
    labelsBox = document.createElement("p");
    labelsBox.id = "modelLabels";
    labelsBox.className = "model-labels";
    probabilityBars.parentElement.appendChild(labelsBox);
  }

  if (labelsBox) {
    labelsBox.textContent = prediction.topLabels.length
      ? "Top model labels: " +
        prediction.topLabels
          .slice(0, 8)
          .map(
            (item) =>
              `${item.label} (${(item.score * 100).toFixed(1)}%)`
          )
          .join(", ")
      : "The model returned no readable labels.";
  }
}

function displayProbabilities(probabilities) {
  if (!probabilityBars) return;

  probabilityBars.replaceChildren();

  for (const [animal, percentage] of Object.entries(probabilities)) {
    const row = document.createElement("div");
    row.className = "probability-row";

    const label = document.createElement("span");
    label.textContent = animal;

    const value = document.createElement("span");
    value.textContent = percentage.toFixed(1) + "%";

    const track = document.createElement("div");
    track.className = "probability-track";

    const bar = document.createElement("div");
    bar.className = "probability-fill";
    bar.style.width = Math.min(100, percentage) + "%";

    track.appendChild(bar);
    row.append(label, value);
    probabilityBars.append(row, track);
  }
}

// ==========================================
// TIME-DOMAIN WAVEFORM
// ==========================================

function drawWaveform(data, sampleRate) {
  if (!waveformCanvas || !data.length) return;

  const ctx = waveformCanvas.getContext("2d");
  if (!ctx) return;

  const width = Math.max(300, waveformCanvas.clientWidth || 800);
  const height = 300;

  waveformCanvas.width = width;
  waveformCanvas.height = height;

  ctx.fillStyle = "#fafbfe";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "#d9dbe5";
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  ctx.strokeStyle = "#6c63ff";
  ctx.lineWidth = 1.2;
  ctx.beginPath();

  for (let x = 0; x < width; x++) {
    const index = Math.min(
      data.length - 1,
      Math.floor((x / width) * data.length)
    );

    const y = height / 2 - data[index] * height * 0.42;

    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }

  ctx.stroke();

  ctx.fillStyle = "#667085";
  ctx.font = "12px Arial";
  ctx.fillText("Amplitude", 10, 18);
  ctx.fillText(
    `Time → (${(data.length / sampleRate).toFixed(2)} s)`,
    Math.max(10, width - 150),
    height - 10
  );
}

// ==========================================
// FFT FREQUENCY ANALYSIS
// ==========================================

function performFFT(signal, sampleRate) {
  const fftSize = 2 ** Math.floor(
    Math.log2(Math.min(4096, signal.length))
  );

  if (fftSize < 2) {
    throw new Error("Audio is too short for FFT analysis.");
  }

  const start = Math.max(
    0,
    Math.floor((signal.length - fftSize) / 2)
  );

  const real = new Float64Array(fftSize);
  const imag = new Float64Array(fftSize);

  for (let i = 0; i < fftSize; i++) {
    const hann = 0.5 * (
      1 - Math.cos((2 * Math.PI * i) / (fftSize - 1))
    );

    real[i] = signal[start + i] * hann;
  }

  fft(real, imag);

  const half = Math.floor(fftSize / 2);
  const frequencies = new Float64Array(half);
  const magnitudes = new Float64Array(half);

  for (let i = 0; i < half; i++) {
    frequencies[i] = (i * sampleRate) / fftSize;
    magnitudes[i] = Math.hypot(real[i], imag[i]);
  }

  return { frequencies, magnitudes };
}

function fft(real, imag) {
  const n = real.length;

  if (n < 2 || (n & (n - 1)) !== 0) {
    throw new Error("FFT size must be a power of two.");
  }

  let j = 0;

  for (let i = 1; i < n; i++) {
    let bit = n >> 1;

    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }

    j ^= bit;

    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }

  for (let length = 2; length <= n; length <<= 1) {
    const angle = (-2 * Math.PI) / length;
    const stepReal = Math.cos(angle);
    const stepImag = Math.sin(angle);

    for (let offset = 0; offset < n; offset += length) {
      let wr = 1;
      let wi = 0;
      const half = length >> 1;

      for (let k = 0; k < half; k++) {
        const even = offset + k;
        const odd = even + half;

        const tr = wr * real[odd] - wi * imag[odd];
        const ti = wr * imag[odd] + wi * real[odd];

        real[odd] = real[even] - tr;
        imag[odd] = imag[even] - ti;
        real[even] += tr;
        imag[even] += ti;

        const nextWr = wr * stepReal - wi * stepImag;
        wi = wr * stepImag + wi * stepReal;
        wr = nextWr;
      }
    }
  }
}

// ==========================================
// DRAW FREQUENCY SPECTRUM
// ==========================================

function drawFrequencySpectrum(frequencies, magnitudes) {
  if (!fftCanvas || !frequencies.length) return;

  const ctx = fftCanvas.getContext("2d");
  if (!ctx) return;

  const width = Math.max(300, fftCanvas.clientWidth || 800);
  const height = 300;

  fftCanvas.width = width;
  fftCanvas.height = height;

  ctx.fillStyle = "#fafbfe";
  ctx.fillRect(0, 0, width, height);

  const maxFrequency = Math.min(
    10000,
    frequencies[frequencies.length - 1]
  );

  let maxMagnitude = 0;

  for (let i = 1; i < frequencies.length; i++) {
    if (frequencies[i] > maxFrequency) break;
    maxMagnitude = Math.max(maxMagnitude, magnitudes[i]);
  }

  if (maxMagnitude <= 0) {
    ctx.fillStyle = "#667085";
    ctx.fillText("No usable frequency content found", 12, 24);
    return;
  }

  const bottom = height - 30;
  ctx.strokeStyle = "#00a8cc";
  ctx.lineWidth = 1.5;
  ctx.beginPath();

  let started = false;

  for (let i = 0; i < frequencies.length; i++) {
    if (frequencies[i] > maxFrequency) break;

    const x = (frequencies[i] / maxFrequency) * width;
    const y = bottom - (magnitudes[i] / maxMagnitude) * (height - 55);

    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }

  ctx.stroke();

  ctx.fillStyle = "#667085";
  ctx.font = "12px Arial";
  ctx.fillText("Magnitude", 10, 18);
  ctx.fillText("Frequency (Hz)", Math.max(10, width - 110), height - 8);
}

// ==========================================
// DOMINANT FREQUENCY
// ==========================================

function findDominantFrequency(frequencies, magnitudes) {
  let maxMagnitude = 0;
  let dominantFrequency = 0;

  for (let i = 1; i < frequencies.length; i++) {
    if (frequencies[i] > 10000) break;

    if (magnitudes[i] > maxMagnitude) {
      maxMagnitude = magnitudes[i];
      dominantFrequency = frequencies[i];
    }
  }

  return dominantFrequency;
}

// Clean up when the page is closed
window.addEventListener("beforeunload", () => {
  if (audioObjectUrl) URL.revokeObjectURL(audioObjectUrl);
  if (audioClassifier?.close) audioClassifier.close();
  if (audioContext && audioContext.state !== "closed") {
    audioContext.close();
  }
});

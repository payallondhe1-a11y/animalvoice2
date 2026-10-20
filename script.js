"use strict";

// ==========================================
// ANIMALSOUND: ANIMAL SOUND RECOGNITION
// Features:
// 1. Audio upload and playback
// 2. Waveform visualization
// 3. FFT frequency spectrum
// 4. Dominant frequency calculation
// 5. Animal sound classification using YAMNet
// 6. Prediction scores and model labels
// ==========================================

// ---------- HTML ELEMENTS ----------

const $ = (id) => document.getElementById(id);

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

// ---------- GLOBAL VARIABLES ----------

let audioContext = null;
let audioBuffer = null;
let audioObjectUrl = null;

let audioClassifier = null;
let classifierPromise = null;

// MediaPipe YAMNet model
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@0.10.14/wasm";

// ---------- INITIALIZATION ----------

if (!audioUpload) {
  console.error(
    'AnimalSound: Could not find element with id="audioUpload". Check your HTML.'
  );
} else {
  audioUpload.addEventListener("change", handleAudioUpload);
}

// ==========================================
// 1. HANDLE AUDIO UPLOAD
// ==========================================

async function handleAudioUpload(event) {
  const file = event.target.files && event.target.files[0];

  if (!file) {
    return;
  }

  if (file.size === 0) {
    showError("The selected audio file is empty.");
    return;
  }

  if (fileName) {
    fileName.textContent = `Selected: ${file.name}`;
  }

  if (audioSection) {
    audioSection.classList.remove("hidden");
  }

  if (resultsSection) {
    resultsSection.classList.remove("hidden");
  }

  // Release the previous audio URL
  if (audioObjectUrl) {
    URL.revokeObjectURL(audioObjectUrl);
  }

  // Create a URL for audio playback
  audioObjectUrl = URL.createObjectURL(file);

  if (audioPlayer) {
    audioPlayer.src = audioObjectUrl;
  }

  setStatus("Loading audio...");

  try {
    const AudioContextClass =
      window.AudioContext || window.webkitAudioContext;

    if (!AudioContextClass) {
      throw new Error(
        "This browser does not support the Web Audio API."
      );
    }

    if (!audioContext || audioContext.state === "closed") {
      audioContext = new AudioContextClass();
    }

    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }

    // Read the uploaded audio file
    const bytes = await file.arrayBuffer();

    // Decode audio into numerical samples
    audioBuffer = await audioContext.decodeAudioData(
      bytes.slice(0)
    );

    if (!audioBuffer || audioBuffer.length < 2) {
      throw new Error(
        "The audio file could not be decoded or is too short."
      );
    }

    // Convert stereo audio into mono audio
    const mono = getMonoAudio(audioBuffer);

    // Display waveform
    drawWaveform(mono, audioBuffer.sampleRate);

    // Perform FFT analysis
    const fftResult = performFFT(
      mono,
      audioBuffer.sampleRate
    );

    // Display frequency spectrum
    drawFrequencySpectrum(
      fftResult.frequencies,
      fftResult.magnitudes
    );

    // Find dominant frequency
    const dominantFrequency = findDominantFrequency(
      fftResult.frequencies,
      fftResult.magnitudes
    );

    if (frequencyResult) {
      frequencyResult.textContent =
        Number.isFinite(dominantFrequency)
          ? dominantFrequency.toFixed(2)
          : "Not available";
    }

    // Classify the uploaded audio
    setStatus("Classifying sound using the YAMNet model...");

    const model = await getAudioClassifier();

    const results = model.classify(
      mono,
      audioBuffer.sampleRate
    );

    const categories =
      results?.[0]?.classifications?.[0]?.categories || [];

    // Convert model categories into animal groups
    const prediction = mapCategoriesToAnimals(categories);

    // Show prediction on the website
    renderPrediction(prediction);

    if (prediction.animal === "Unknown") {
      setStatus(
        "No clear animal sound was identified. Try a clearer recording."
      );
    } else {
      setStatus(
        "Analysis complete. The result is a model estimate, not guaranteed identification."
      );
    }

  } catch (error) {
    console.error("AnimalSound analysis error:", error);

    showError(
      `Could not analyze this file. ${
        error?.message || "Check the browser console for details."
      } Try a short WAV or MP3 recording and ensure you are online for the first model download.`
    );
  }
}

// ==========================================
// 2. LOAD THE YAMNET CLASSIFICATION MODEL
// ==========================================

async function getAudioClassifier() {
  if (audioClassifier) {
    return audioClassifier;
  }

  if (classifierPromise) {
    return classifierPromise;
  }

  classifierPromise = (async () => {
    if (
      !window.FilesetResolver ||
      !window.AudioClassifier
    ) {
      throw new Error(
        "MediaPipe did not load. Add audio_bundle.js before script.js in index.html."
      );
    }

    // Load the MediaPipe audio processing files
    const fileset =
      await window.FilesetResolver.forAudioTasks(WASM_URL);

    // Create the YAMNet audio classifier
    const classifier =
      await window.AudioClassifier.createFromOptions(
        fileset,
        {
          baseOptions: {
            modelAssetPath: MODEL_URL
          },
          maxResults: 521,
          scoreThreshold: 0
        }
      );

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
// 3. CONVERT AUDIO TO MONO
// ==========================================

function getMonoAudio(buffer) {
  // If audio is already mono, copy its samples
  if (buffer.numberOfChannels === 1) {
    return new Float32Array(
      buffer.getChannelData(0)
    );
  }

  // Combine all channels into one signal
  const mono = new Float32Array(buffer.length);

  for (
    let channel = 0;
    channel < buffer.numberOfChannels;
    channel++
  ) {
    const data = buffer.getChannelData(channel);

    for (let i = 0; i < data.length; i++) {
      mono[i] += data[i] / buffer.numberOfChannels;
    }
  }

  return mono;
}

// ==========================================
// 4. MAP MODEL LABELS TO ANIMAL GROUPS
// ==========================================

function mapCategoriesToAnimals(categories) {
  const groups = {
    Cat: [
      "cat",
      "meow",
      "purr",
      "caterwaul"
    ],

    Dog: [
      "dog",
      "bark",
      "bow-wow",
      "growling",
      "whimper",
      "yip"
    ],

    Cow: [
      "cattle",
      "moo",
      "cowbell"
    ],

    Frog: [
      "frog",
      "croak"
    ],

    Crow: [
      "crow",
      "caw"
    ],

    "Chirping Birds": [
      "bird",
      "chirp",
      "tweet",
      "bird vocalization",
      "bird song",
      "owl",
      "duck",
      "goose",
      "chicken"
    ]
  };

  const scores = Object.fromEntries(
    Object.keys(groups).map((name) => [name, 0])
  );

  const bestLabels = [];

  for (const category of categories || []) {
    const label = String(
      category.categoryName ||
      category.displayName ||
      ""
    ).toLowerCase();

    const score = Number(category.score) || 0;

    if (!label || score <= 0) {
      continue;
    }

    // Find animal-related model labels
    for (const [animal, keywords] of Object.entries(groups)) {
      const matches = keywords.some(
        (keyword) =>
          label === keyword || label.includes(keyword)
      );

      if (matches) {
        // Keep the strongest matching score
        scores[animal] = Math.max(
          scores[animal],
          score
        );
      }
    }

    bestLabels.push({
      label:
        category.categoryName ||
        category.displayName ||
        label,
      score
    });
  }

  // Check whether any animal-related label was found
  const total = Object.values(scores).reduce(
    (sum, value) => sum + value,
    0
  );

  if (total <= 0) {
    return {
      animal: "Unknown",
      confidence: 0,

      probabilities: Object.fromEntries(
        Object.keys(groups).map(
          (name) => [name, 0]
        )
      ),

      topLabels: bestLabels
        .sort((a, b) => b.score - a.score)
        .slice(0, 5)
    };
  }

  // Normalize group scores to display percentages
  const probabilities = Object.fromEntries(
    Object.entries(scores).map(
      ([name, value]) => [
        name,
        (value / total) * 100
      ]
    )
  );

  // Select the animal group with the highest score
  const ranked = Object.entries(scores).sort(
    (a, b) => b[1] - a[1]
  );

  const [animal, topScore] = ranked[0];

  // Find the strongest matching raw model score
  const matchingLabelScore = Math.max(
    0,

    ...bestLabels
      .filter((item) =>
        groups[animal].some((keyword) => {
          const label = item.label.toLowerCase();

          return (
            label === keyword ||
            label.includes(keyword)
          );
        })
      )
      .map((item) => item.score)
  );

  return {
    animal,

    // This is a model score, not a guaranteed accuracy rate
    confidence: matchingLabelScore * 100,

    probabilities,

    topLabels: bestLabels
      .sort((a, b) => b.score - a.score)
      .slice(0, 5),

    groupScore: topScore
  };
}

// ==========================================
// 5. DISPLAY THE PREDICTION
// ==========================================

function renderPrediction(prediction) {
  if (animalResult) {
    animalResult.textContent = prediction.animal;
  }

  if (confidenceResult) {
    confidenceResult.textContent =
      prediction.animal === "Unknown"
        ? "Not available"
        : `${prediction.confidence.toFixed(1)}% model score`;
  }

  displayProbabilities(
    prediction.probabilities
  );

  // Create a model-label section if it does not exist
  let labelsBox = $("modelLabels");

  if (
    !labelsBox &&
    probabilityBars?.parentElement
  ) {
    labelsBox = document.createElement("p");

    labelsBox.id = "modelLabels";
    labelsBox.className = "model-labels";

    probabilityBars.parentElement.appendChild(
      labelsBox
    );
  }

  // Show the top labels returned by the model
  if (labelsBox) {
    labelsBox.textContent =
      prediction.topLabels?.length
        ? "Top model labels: " +
          prediction.topLabels
            .slice(0, 5)
            .map(
              (item) =>
                `${item.label} (${(
                  item.score * 100
                ).toFixed(1)}%)`
            )
            .join(", ")
        : "No animal-related label was detected with a useful score.";
  }
}

// ==========================================
// 6. DISPLAY STATUS MESSAGES
// ==========================================

function setStatus(message) {
  let status = $("analysisStatus");

  // Create the status element when necessary
  if (!status && resultsSection) {
    status = document.createElement("p");

    status.id = "analysisStatus";
    status.setAttribute("role", "status");

    status.style.marginTop = "12px";
    status.style.fontSize = "14px";
    status.style.color = "#667085";

    resultsSection.appendChild(status);
  }

  if (status) {
    status.textContent = message;
  }
}

// ==========================================
// 7. DISPLAY ERROR MESSAGES
// ==========================================

function showError(message) {
  setStatus(message);
  console.error(message);
}

// ==========================================
// 8. DRAW THE TIME-DOMAIN WAVEFORM
// ==========================================

function drawWaveform(data, sampleRate) {
  if (!waveformCanvas || !data?.length) {
    return;
  }

  const ctx = waveformCanvas.getContext("2d");

  if (!ctx) {
    return;
  }

  const width = Math.max(
    1,
    waveformCanvas.clientWidth || 800
  );

  const height = 300;

  waveformCanvas.width = width;
  waveformCanvas.height = height;

  // Background
  ctx.clearRect(0, 0, width, height);

  ctx.fillStyle = "#fafbfe";
  ctx.fillRect(0, 0, width, height);

  // Draw the zero-amplitude line
  ctx.strokeStyle = "#d9dbe5";
  ctx.lineWidth = 1;

  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  // Draw waveform
  ctx.beginPath();

  ctx.strokeStyle = "#6c63ff";
  ctx.lineWidth = 1.5;

  for (let x = 0; x < width; x++) {
    const index = Math.min(
      data.length - 1,
      Math.floor((x / width) * data.length)
    );

    const y =
      height / 2 -
      data[index] * (height * 0.42);

    if (x === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }

  ctx.stroke();

  // Labels
  ctx.fillStyle = "#667085";
  ctx.font = "12px Arial";

  ctx.fillText(
    "Amplitude",
    10,
    18
  );

  ctx.fillText(
    `Time → (${(
      data.length / sampleRate
    ).toFixed(2)} s)`,
    Math.max(10, width - 150),
    height - 10
  );
}

// ==========================================
// 9. PERFORM FFT ANALYSIS
// ==========================================

function performFFT(signal, sampleRate) {
  if (
    !signal?.length ||
    !Number.isFinite(sampleRate) ||
    sampleRate <= 0
  ) {
    throw new Error(
      "Invalid audio signal or sample rate."
    );
  }

  // Limit the FFT size for browser performance
  const maxSize = Math.min(
    4096,
    signal.length
  );

  // FFT size must be a power of two
  const fftSize =
    2 ** Math.floor(Math.log2(maxSize));

  if (fftSize < 2) {
    throw new Error(
      "Audio is too short for FFT analysis."
    );
  }

  // Select a frame from the center of the audio
  const start = Math.max(
    0,
    Math.floor((signal.length - fftSize) / 2)
  );

  const real = new Float64Array(fftSize);
  const imag = new Float64Array(fftSize);

  // Apply a Hann window to reduce spectral leakage
  for (let i = 0; i < fftSize; i++) {
    const hann =
      0.5 *
      (
        1 -
        Math.cos(
          (2 * Math.PI * i) /
          (fftSize - 1)
        )
      );

    real[i] = signal[start + i] * hann;
  }

  // Calculate FFT
  fft(real, imag);

  const half = Math.floor(fftSize / 2);

  const frequencies = new Float64Array(half);
  const magnitudes = new Float64Array(half);

  // Calculate frequency bins and magnitudes
  for (let i = 0; i < half; i++) {
    frequencies[i] =
      (i * sampleRate) / fftSize;

    magnitudes[i] = Math.hypot(
      real[i],
      imag[i]
    );
  }

  return {
    frequencies,
    magnitudes
  };
}

// ==========================================
// 10. RADIX-2 FAST FOURIER TRANSFORM
// ==========================================

function fft(real, imag) {
  const n = real.length;

  if (
    n < 2 ||
    (n & (n - 1)) !== 0
  ) {
    throw new Error(
      "FFT size must be a power of two."
    );
  }

  // Bit-reversal permutation
  let j = 0;

  for (let i = 1; i < n; i++) {
    let bit = n >> 1;

    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }

    j ^= bit;

    if (i < j) {
      [real[i], real[j]] = [
        real[j],
        real[i]
      ];

      [imag[i], imag[j]] = [
        imag[j],
        imag[i]
      ];
    }
  }

  // Butterfly computations
  for (
    let length = 2;
    length <= n;
    length <<= 1
  ) {
    const angle =
      (-2 * Math.PI) / length;

    const stepReal = Math.cos(angle);
    const stepImag = Math.sin(angle);

    for (
      let offset = 0;
      offset < n;
      offset += length
    ) {
      let wr = 1;
      let wi = 0;

      const half = length >> 1;

      for (let k = 0; k < half; k++) {
        const even = offset + k;
        const odd = even + half;

        const tr =
          wr * real[odd] -
          wi * imag[odd];

        const ti =
          wr * imag[odd] +
          wi * real[odd];

        real[odd] =
          real[even] - tr;

        imag[odd] =
          imag[even] - ti;

        real[even] += tr;
        imag[even] += ti;

        const nextWr =
          wr * stepReal -
          wi * stepImag;

        wi =
          wr * stepImag +
          wi * stepReal;

        wr = nextWr;
      }
    }
  }
}

// ==========================================
// 11. DRAW FREQUENCY SPECTRUM
// ==========================================

function drawFrequencySpectrum(
  frequencies,
  magnitudes
) {
  if (
    !fftCanvas ||
    !frequencies?.length ||
    !magnitudes?.length
  ) {
    return;
  }

  const ctx = fftCanvas.getContext("2d");

  if (!ctx) {
    return;
  }

  const width = Math.max(
    1,
    fftCanvas.clientWidth || 800
  );

  const height = 300;

  fftCanvas.width = width;
  fftCanvas.height = height;

  ctx.clearRect(0, 0, width, height);

  ctx.fillStyle = "#fafbfe";
  ctx.fillRect(0, 0, width, height);

  // Display frequencies up to 10 kHz
  const maxFrequency = Math.min(
    10000,
    frequencies[frequencies.length - 1]
  );

  let maxMagnitude = 0;

  for (
    let i = 1;
    i < frequencies.length &&
    frequencies[i] <= maxFrequency;
    i++
  ) {
    if (magnitudes[i] > maxMagnitude) {
      maxMagnitude = magnitudes[i];
    }
  }

  if (
    !Number.isFinite(maxMagnitude) ||
    maxMagnitude <= 0
  ) {
    ctx.fillStyle = "#667085";
    ctx.font = "14px Arial";

    ctx.fillText(
      "No usable frequency content found",
      12,
      24
    );

    return;
  }

  const plotBottom = height - 30;

  // Draw the spectrum
  ctx.strokeStyle = "#00a8cc";
  ctx.lineWidth = 1.5;

  ctx.beginPath();

  let started = false;

  for (
    let i = 0;
    i < frequencies.length &&
    frequencies[i] <= maxFrequency;
    i++
  ) {
    const x =
      (frequencies[i] / maxFrequency) *
      width;

    const y =
      plotBottom -
      (magnitudes[i] / maxMagnitude) *
      (height - 55);

    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }

  ctx.stroke();

  // Axis labels
  ctx.fillStyle = "#667085";
  ctx.font = "12px Arial";

  ctx.fillText(
    "Magnitude",
    10,
    18
  );

  ctx.fillText(
    "Frequency (Hz) →",
    Math.max(10, width - 120),
    height - 8
  );

  // Frequency scale
  const ticks = [
    0,
    2000,
    4000,
    6000,
    8000,
    10000
  ];

  for (const frequency of ticks) {
    if (frequency > maxFrequency) {
      continue;
    }

    const x =
      (frequency / maxFrequency) *
      width;

    ctx.fillText(
      String(frequency),
      Math.min(
        width - 35,
        Math.max(0, x - 10)
      ),
      height - 15
    );
  }
}

// ==========================================
// 12. FIND DOMINANT FREQUENCY
// ==========================================

function findDominantFrequency(
  frequencies,
  magnitudes
) {
  let maxMagnitude = 0;
  let dominantFrequency = 0;

  const limit = Math.min(
    10000,
    frequencies[frequencies.length - 1] || 0
  );

  // Ignore the zero-frequency component
  for (
    let i = 1;
    i < frequencies.length &&
    frequencies[i] <= limit;
    i++
  ) {
    if (magnitudes[i] > maxMagnitude) {
      maxMagnitude = magnitudes[i];
      dominantFrequency = frequencies[i];
    }
  }

  return dominantFrequency;
}

// ==========================================
// 13. DISPLAY ANIMAL SCORES
// ==========================================

function displayProbabilities(probabilities) {
  if (!probabilityBars) {
    return;
  }

  // Remove previous results
  probabilityBars.replaceChildren();

  const entries = Object.entries(
    probabilities || {}
  ).sort(
    (a, b) => b[1] - a[1]
  );

  for (const [animal, rawValue] of entries) {
    const value = Math.max(
      0,
      Math.min(
        100,
        Number(rawValue) || 0
      )
    );

    // Main container
    const item = document.createElement("div");
    item.className = "probability-item";

    // Name and percentage
    const info = document.createElement("div");
    info.className = "probability-info";

    const name = document.createElement("span");
    name.textContent = animal;

    const percent = document.createElement("span");
    percent.textContent =
      `${value.toFixed(1)}%`;

    info.append(name, percent);

    // Bar background
    const background =
      document.createElement("div");

    background.className =
      "probability-background";

    // Bar fill
    const fill = document.createElement("div");

    fill.className = "probability-fill";
    fill.style.width = `${value}%`;

    background.appendChild(fill);

    item.append(info, background);

    probabilityBars.appendChild(item);
  }
}

// ==========================================
// 14. REDRAW CHARTS WHEN WINDOW RESIZES
// ==========================================

window.addEventListener("resize", () => {
  if (!audioBuffer) {
    return;
  }

  try {
    const mono = getMonoAudio(audioBuffer);

    drawWaveform(
      mono,
      audioBuffer.sampleRate
    );

    const result = performFFT(
      mono,
      audioBuffer.sampleRate
    );

    drawFrequencySpectrum(
      result.frequencies,
      result.magnitudes
    );

  } catch (error) {
    console.error(
      "Could not redraw charts:",
      error
    );
  }
});

// ==========================================
// 15. CLEAN UP WHEN PAGE CLOSES
// ==========================================

window.addEventListener("beforeunload", () => {
  if (audioObjectUrl) {
    URL.revokeObjectURL(audioObjectUrl);
  }

  if (
    audioClassifier &&
    typeof audioClassifier.close === "function"
  ) {
    audioClassifier.close();
  }

  if (
    audioContext &&
    audioContext.state !== "closed"
  ) {
    audioContext.close();
  }
});

// ==========================================
// APPLICATION READY
// ==========================================

console.log(
  "AnimalSound loaded: waveform + FFT + YAMNet classification."
);

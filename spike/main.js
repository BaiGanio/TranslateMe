import { pipeline, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

const $ = (id) => document.getElementById(id);
const logEl = $('log');

function log(msg) {
  console.log(`[spike] ${msg}`);
  logEl.textContent += `${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function showError(msg) {
  console.error(`[spike] ${msg}`);
  $('error').textContent = msg;
}

let generator = null;
let loadedModel = null;

// Transformers.js stores downloaded files in Cache Storage under this name.
async function isCached(modelId, dtype) {
  if (!('caches' in window)) return false;
  const cache = await caches.open('transformers-cache');
  const keys = await cache.keys();
  return keys.some((req) => req.url.includes(modelId) && req.url.includes(`model_${dtype}.onnx`));
}

async function loadModel(modelId) {
  for (const dtype of ['q4f16', 'q4']) {
    const cached = await isCached(modelId, dtype);
    log(`Loading ${modelId} (dtype=${dtype}, ${cached ? 'cached load' : 'first load'})...`);
    const t0 = performance.now();
    try {
      const gen = await pipeline('text-generation', modelId, { device: 'webgpu', dtype });
      const ms = performance.now() - t0;
      log(`Loaded ${modelId} dtype=${dtype} in ${(ms / 1000).toFixed(2)} s (${cached ? 'cached load' : 'first load'})`);
      return gen;
    } catch (err) {
      log(`dtype=${dtype} failed: ${err?.message ?? err}`);
      if (dtype === 'q4f16') log('Falling back to dtype=q4');
      else throw err;
    }
  }
}

async function ensureModel() {
  const modelId = $('model').value;
  if (generator && loadedModel === modelId) return;
  if (generator) {
    await generator.dispose();
    generator = null;
  }
  generator = await loadModel(modelId);
  loadedModel = modelId;
}

async function translate() {
  const text = $('source').value.trim();
  const lang = $('target').value;
  if (!text) return;

  await ensureModel();

  const messages = [
    { role: 'system', content: `Translate the user's text into ${lang}. Output only the translation. No notes, no explanations, no quotes.` },
    { role: 'user', content: text },
  ];

  $('output').textContent = '';
  let tokens = 0;
  let firstTokenAt = 0;
  const t0 = performance.now();

  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (chunk) => { $('output').textContent += chunk; },
    token_callback_function: () => {
      tokens++;
      if (tokens === 1) firstTokenAt = performance.now();
    },
  });

  await generator(messages, { max_new_tokens: 128, do_sample: false, streamer });
  const end = performance.now();

  const ttft = firstTokenAt - t0;
  const decodeTps = tokens > 1 ? (tokens - 1) / ((end - firstTokenAt) / 1000) : 0;
  const totalTps = tokens / ((end - t0) / 1000);
  log(`[${loadedModel.split('/')[1]} -> ${lang}] TTFT ${ttft.toFixed(0)} ms | ${tokens} tokens | ` +
      `${decodeTps.toFixed(1)} tok/s (decode) | ${totalTps.toFixed(1)} tok/s (overall) | total ${((end - t0) / 1000).toFixed(2)} s`);
}

async function run(fn) {
  $('load').disabled = $('translate').disabled = true;
  $('error').textContent = '';
  $('output').textContent = '';
  try {
    await fn();
  } catch (err) {
    showError(`Error: ${err?.message ?? err}`);
  } finally {
    $('load').disabled = $('translate').disabled = false;
  }
}

if (!navigator.gpu) {
  showError('WebGPU is not available (navigator.gpu is missing). Use a recent Chrome on localhost or HTTPS.');
  $('load').disabled = $('translate').disabled = true;
} else {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    showError('WebGPU is present, but no GPU adapter was found.');
  } else {
    log(`WebGPU OK. shader-f16: ${adapter.features.has('shader-f16') ? 'yes' : 'no'}`);
  }
  $('load').onclick = () => run(ensureModel);
  $('translate').onclick = () => run(translate);
}

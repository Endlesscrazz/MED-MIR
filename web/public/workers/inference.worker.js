/**
 * Med-MIR Inference Worker
 *
 * Runs BiomedCLIP text encoder directly using ONNX Runtime Web.
 * This file is served statically from public/ — no webpack bundling.
 *
 * Architecture:
 *   1. Loads vocab.txt → builds a BERT WordPiece tokenizer
 *   2. Downloads the ONNX model (BiomedCLIP text encoder)
 *   3. Creates an ONNX Runtime inference session (WASM backend)
 *   4. On EMBED_TEXT messages: tokenize → infer → return 512-dim embedding
 *
 * Why CDN?  Loading onnxruntime-web from a CDN means:
 *   - No webpack bundling issues (onnxruntime-node, sharp, .node binaries)
 *   - WASM files are served from the CDN automatically
 *   - Zero configuration required
 */

// ================================================================
// POLYFILL: onnxruntime-web references `window` internally.
// Web Workers only have `self`, so we alias window → self.
// This MUST run before importScripts loads ort.min.js.
// ================================================================
if (typeof window === 'undefined') {
  self.window = self;
}

// Load ONNX Runtime Web from CDN
const ORT_VERSION = '1.19.2';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist`;
importScripts(`${ORT_CDN}/ort.min.js`);

// Configure ONNX Runtime
ort.env.wasm.wasmPaths = `${ORT_CDN}/`;
ort.env.wasm.numThreads = 1; // Single thread in worker (worker IS the thread)


// ================================================================
// BERT WORDPIECE TOKENIZER
// Replicates PubMedBERT tokenization for BiomedCLIP compatibility.
// ================================================================

class BertTokenizer {
  /**
   * Build tokenizer from vocab.txt contents.
   * @param {string} vocabText - Contents of vocab.txt (one token per line)
   */
  constructor(vocabText) {
    /** @type {Map<string, number>} token → id */
    this.vocab = new Map();
    /** @type {Map<number, string>} id → token */
    this.idsToTokens = new Map();

    const lines = vocabText.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const token = lines[i].trimEnd(); // preserve leading whitespace in some vocabs
      if (token !== '') {
        this.vocab.set(token, i);
        this.idsToTokens.set(i, token);
      }
    }

    // Standard BERT special token IDs
    this.clsId = this.vocab.get('[CLS]') ?? 101;
    this.sepId = this.vocab.get('[SEP]') ?? 102;
    this.padId = this.vocab.get('[PAD]') ?? 0;
    this.unkId = this.vocab.get('[UNK]') ?? 100;
  }

  /**
   * Basic tokenization: lowercase, strip accents, split on whitespace & punctuation.
   * Mirrors HuggingFace BasicTokenizer with do_lower_case=True.
   *
   * @param {string} text
   * @returns {string[]}
   */
  basicTokenize(text) {
    // Lowercase
    text = text.toLowerCase().trim();
    // Strip accents (NFD decomposition + remove combining marks)
    text = text.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
    // Clean control characters
    text = text.replace(/[\x00-\x1f\x7f-\x9f]/g, '');
    // Add spaces around punctuation
    text = text.replace(/([\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e])/g, ' $1 ');
    // Split on whitespace and filter empties
    return text.split(/\s+/).filter(function (t) { return t.length > 0; });
  }

  /**
   * WordPiece tokenization for a single word.
   * Greedily matches the longest prefix from the vocabulary.
   *
   * @param {string} word
   * @returns {number[]} token IDs
   */
  wordPieceTokenize(word) {
    if (word.length > 200) return [this.unkId];

    var tokens = [];
    var start = 0;

    while (start < word.length) {
      var end = word.length;
      var found = false;

      while (start < end) {
        var sub = word.substring(start, end);
        if (start > 0) sub = '##' + sub;

        if (this.vocab.has(sub)) {
          tokens.push(this.vocab.get(sub));
          found = true;
          break;
        }
        end--;
      }

      if (!found) {
        tokens.push(this.unkId);
        break; // Entire remaining word is unknown
      }
      start = end;
    }

    return tokens;
  }

  /**
   * Full tokenization pipeline: text → padded token ID array.
   *
   * Output format: [CLS] token1 token2 ... [SEP] [PAD] [PAD] ...
   * This matches HuggingFace AutoTokenizer with padding='max_length'.
   *
   * @param {string} text - Input text
   * @param {number} maxLength - Maximum sequence length (default 256)
   * @returns {number[]} Array of token IDs, length === maxLength
   */
  encode(text, maxLength) {
    maxLength = maxLength || 256;
    var words = this.basicTokenize(text);
    var ids = [this.clsId]; // Start with [CLS]

    for (var w = 0; w < words.length; w++) {
      var wpIds = this.wordPieceTokenize(words[w]);
      for (var j = 0; j < wpIds.length; j++) {
        if (ids.length >= maxLength - 1) break; // Leave room for [SEP]
        ids.push(wpIds[j]);
      }
      if (ids.length >= maxLength - 1) break;
    }

    ids.push(this.sepId); // End with [SEP]

    // Pad to maxLength
    while (ids.length < maxLength) {
      ids.push(this.padId);
    }

    return ids;
  }
}


// ================================================================
// MODEL MANAGEMENT
// ================================================================

var MODEL_URL = '/demo-data/model/model_flat.onnx';
var VOCAB_URL = '/demo-data/model/vocab.txt';
var MAX_LENGTH = 256;

/** @type {ort.InferenceSession|null} */
var session = null;
/** @type {BertTokenizer|null} */
var tokenizer = null;
var isInitializing = false;

/**
 * Send a message to the main thread.
 * @param {string} type
 * @param {string} [id]
 * @param {Object} [payload]
 */
function send(type, id, payload) {
  self.postMessage({ type: type, id: id, payload: payload });
}

/**
 * Download a file with progress tracking.
 *
 * @param {string} url - URL to fetch
 * @param {string} label - Human-readable label for progress messages
 * @returns {Promise<ArrayBuffer>}
 */
async function downloadWithProgress(url, label) {
  var response = await fetch(url);

  if (!response.ok) {
    throw new Error('Failed to fetch ' + url + ': ' + response.status + ' ' + response.statusText);
  }

  var contentLength = response.headers.get('Content-Length');
  var total = contentLength ? parseInt(contentLength, 10) : 0;

  // If no streaming support or unknown size, fall back to simple download
  if (!response.body || total === 0) {
    send('PROGRESS', undefined, {
      progress: 0.5,
      status: 'Downloading ' + label + '...',
    });
    return await response.arrayBuffer();
  }

  var reader = response.body.getReader();
  var chunks = [];
  var loaded = 0;

  while (true) {
    var result = await reader.read();
    if (result.done) break;
    chunks.push(result.value);
    loaded += result.value.length;

    if (total > 0) {
      var pct = Math.round((loaded / total) * 100);
      var loadedMB = (loaded / 1048576).toFixed(1);
      var totalMB = (total / 1048576).toFixed(1);
      send('PROGRESS', undefined, {
        progress: 0.05 + (loaded / total) * 0.85,
        status: 'Downloading ' + label + ': ' + pct + '% (' + loadedMB + ' / ' + totalMB + ' MB)',
      });
    }
  }

  // Combine chunks
  var buffer = new Uint8Array(loaded);
  var offset = 0;
  for (var i = 0; i < chunks.length; i++) {
    buffer.set(chunks[i], offset);
    offset += chunks[i].length;
  }

  return buffer.buffer;
}

/**
 * Initialize the tokenizer and ONNX model.
 */
async function initializeModel() {
  if (session || isInitializing) return;
  isInitializing = true;

  try {
    // Step 1: Load vocabulary
    send('PROGRESS', undefined, { progress: 0.02, status: 'Loading tokenizer vocabulary...' });
    var vocabResponse = await fetch(VOCAB_URL);
    if (!vocabResponse.ok) {
      throw new Error('Failed to load vocab.txt (HTTP ' + vocabResponse.status + '). Run the Python export script first.');
    }
    var vocabText = await vocabResponse.text();
    tokenizer = new BertTokenizer(vocabText);
    console.log('[Med-MIR Worker] ✓ Tokenizer loaded (' + tokenizer.vocab.size + ' tokens)');

    // Step 2: Download ONNX model (this is the big download)
    send('PROGRESS', undefined, { progress: 0.05, status: 'Downloading BiomedCLIP model...' });
    var modelBuffer = await downloadWithProgress(MODEL_URL, 'BiomedCLIP');
    console.log('[Med-MIR Worker] ✓ Model downloaded (' + (modelBuffer.byteLength / 1048576).toFixed(1) + ' MB)');

    // Step 3: Create inference session
    send('PROGRESS', undefined, { progress: 0.92, status: 'Initializing AI engine (WASM)...' });
    session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    console.log('[Med-MIR Worker] ✓ ONNX session created');
    console.log('[Med-MIR Worker]   Inputs:', session.inputNames);
    console.log('[Med-MIR Worker]   Outputs:', session.outputNames);

    send('INIT_COMPLETE');
  } catch (error) {
    console.error('[Med-MIR Worker] Initialization failed:', error);
    send('INIT_ERROR', undefined, { error: error.message || 'Unknown initialization error' });
  } finally {
    isInitializing = false;
  }
}

/**
 * Generate a 512-dim L2-normalized embedding for the given text.
 *
 * @param {string} id - Request ID for callback matching
 * @param {string} text - Input text to embed
 */
async function embedText(id, text) {
  if (!session || !tokenizer) {
    send('EMBED_ERROR', id, { error: 'Model not initialized' });
    return;
  }

  try {
    // Tokenize
    var tokenIds = tokenizer.encode(text, MAX_LENGTH);

    // Create input tensor — the ONNX model expects int64 (torch.long)
    var inputData = new BigInt64Array(tokenIds.length);
    for (var i = 0; i < tokenIds.length; i++) {
      inputData[i] = BigInt(tokenIds[i]);
    }
    var inputTensor = new ort.Tensor('int64', inputData, [1, MAX_LENGTH]);

    // Run inference
    var feeds = {};
    feeds[session.inputNames[0]] = inputTensor; // 'input_ids'
    var results = await session.run(feeds);

    // Get output embedding
    var outputName = session.outputNames[0]; // 'embeddings'
    var rawEmbedding = results[outputName].data; // Float32Array

    // L2 normalize (safety net — the ONNX model already normalizes)
    var norm = 0;
    for (var j = 0; j < rawEmbedding.length; j++) {
      norm += rawEmbedding[j] * rawEmbedding[j];
    }
    norm = Math.sqrt(norm);

    var embedding = new Float32Array(rawEmbedding.length);
    if (norm > 0) {
      for (var k = 0; k < rawEmbedding.length; k++) {
        embedding[k] = rawEmbedding[k] / norm;
      }
    }

    // Send as plain array (structured clone doesn't preserve Float32Array across workers)
    send('EMBED_RESULT', id, { embedding: Array.from(embedding) });
  } catch (error) {
    console.error('[Med-MIR Worker] Inference error:', error);
    send('EMBED_ERROR', id, { error: error.message || 'Inference failed' });
  }
}


// ================================================================
// MESSAGE HANDLER
// ================================================================

self.onmessage = async function (event) {
  var data = event.data;
  var type = data.type;
  var id = data.id;
  var payload = data.payload;

  switch (type) {
    case 'INIT':
      await initializeModel();
      break;

    case 'EMBED_TEXT':
      if (payload && payload.text) {
        await embedText(id, payload.text);
      } else {
        send('EMBED_ERROR', id, { error: 'No text provided' });
      }
      break;

    default:
      console.warn('[Med-MIR Worker] Unknown message type:', type);
  }
};

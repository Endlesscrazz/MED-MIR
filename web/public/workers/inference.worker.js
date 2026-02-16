/**
 * web/public/workers/inference.worker.js
 * 
 * Med-MIR Inference Worker
 * Handles Text and Vision encoding using ONNX Runtime Web.
 */

// 1. Polyfill window for ORT (Required for some versions of onnxruntime-web)
if (typeof window === 'undefined') {
    self.window = self;
}

// 2. Load ONNX Runtime Web from CDN
const ORT_VERSION = '1.19.2';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist`;

try {
    importScripts(`${ORT_CDN}/ort.min.js`);
    console.log("[Worker] ORT library loaded successfully");
} catch (e) {
    console.error("[Worker] Failed to load ORT library", e);
}

// 3. Configure WASM
ort.env.wasm.wasmPaths = `${ORT_CDN}/`;
ort.env.wasm.numThreads = 1; // Optimization for M1/Browser environment

// 4. State
let textSession = null;
let visionSession = null;
let tokenizer = null;
let dataBaseUrl = '/demo-data';

function joinUrl(base, path) {
    const cleanBase = (base || '/demo-data').replace(/\/+$/, '');
    const cleanPath = (path || '').replace(/^\/+/, '');
    return `${cleanBase}/${cleanPath}`;
}

// 5. Tokenizer Class (Fixed loops)
class BertTokenizer {
    constructor(vocabText) {
        this.vocab = new Map();
        const lines = vocabText.split('\n');
        // FIX: Explicitly using 'let i'
        for (let i = 0; i < lines.length; i++) {
            const token = lines[i].trimEnd();
            if (token !== '') {
                this.vocab.set(token, i);
            }
        }
        this.clsId = this.vocab.get('[CLS]') ?? 101;
        this.sepId = this.vocab.get('[SEP]') ?? 102;
        this.padId = this.vocab.get('[PAD]') ?? 0;
        this.unkId = this.vocab.get('[UNK]') ?? 100;
    }

    basicTokenize(text) {
        const normalized = text.toLowerCase().trim().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
        const spaced = normalized.replace(/([\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e])/g, ' $1 ');
        return spaced.split(/\s+/).filter(t => t.length > 0);
    }

    wordPieceTokenize(word) {
        if (word.length > 100) return [this.unkId];
        
        let tokens = [];
        let start = 0;
        while (start < word.length) {
            let end = word.length;
            let found = false;
            
            while (start < end) {
                let sub = word.substring(start, end);
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
                break;
            }
            start = end;
        }
        return tokens;
    }

    encode(text, maxLength) {
        const words = this.basicTokenize(text);
        const ids = [this.clsId];
        
        // FIX: Explicitly using 'let w'
        for (let w of words) {
            const wpIds = this.wordPieceTokenize(w);
            // FIX: Explicitly using 'let id'
            for (let id of wpIds) {
                if (ids.length >= maxLength - 1) break;
                ids.push(id);
            }
            if (ids.length >= maxLength - 1) break;
        }
        
        ids.push(this.sepId);
        while (ids.length < maxLength) {
            ids.push(this.padId);
        }
        return ids;
    }
}

// 6. Helpers
function send(type, id, payload) {
    self.postMessage({ type, id, payload });
}

async function fetchFile(url, label) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load ${label}: ${response.statusText}`);
    
    const contentLength = response.headers.get('Content-Length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;
    
    // If it's a large model, show progress
    if (total > 1000000) { // > 1MB
        const reader = response.body.getReader();
        let loaded = 0;
        const chunks = [];
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            loaded += value.length;
            
            send('PROGRESS', undefined, { 
                progress: loaded / total, 
                status: `Downloading ${label}...` 
            });
        }
        
        const buffer = new Uint8Array(loaded);
        let offset = 0;
        // FIX: Explicitly using 'let chunk'
        for (let chunk of chunks) {
            buffer.set(chunk, offset);
            offset += chunk.length;
        }
        return buffer.buffer;
    } else {
        // Small files (vocab) just load directly
        return await response.arrayBuffer();
    }
}

// 7. Message Handler
self.onmessage = async (e) => {
    const { type, id, payload } = e.data;
    console.log(`[Worker] Received message: ${type}`);

    try {
        if (type === 'INIT') {
            dataBaseUrl = payload?.dataBaseUrl || '/demo-data';

            // Load Vocab
            send('PROGRESS', undefined, { progress: 0.05, status: 'Loading vocabulary...' });
            const vocabBuffer = await fetchFile(joinUrl(dataBaseUrl, 'model/vocab.txt'), 'Vocabulary');
            const vocabText = new TextDecoder().decode(vocabBuffer);
            tokenizer = new BertTokenizer(vocabText);

            // Load Text Model
            send('PROGRESS', undefined, { progress: 0.1, status: 'Loading Text Encoder...' });
            const textModelBuffer = await fetchFile(joinUrl(dataBaseUrl, 'model/text_encoder_quantized.onnx'), 'Text Model');
            
            textSession = await ort.InferenceSession.create(textModelBuffer, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            console.log("[Worker] Text Model Initialized");
            send('INIT_COMPLETE', 'text');
        }

        if (type === 'EMBED_TEXT') {
            if (!tokenizer || !textSession) throw new Error("Model not initialized");
            
            const tokenIds = tokenizer.encode(payload.text, 77);
            const input = new ort.Tensor('int64', BigInt64Array.from(tokenIds.map(BigInt)), [1, 77]);
            
            const results = await textSession.run({ input_ids: input });
            send('EMBED_RESULT', id, { embedding: Array.from(results.embeds.data) });
        }

        if (type === 'EMBED_IMAGE') {
            // Lazy load vision model
            if (!visionSession) {
                console.log("[Worker] Lazy loading Vision Model...");
                send('PROGRESS', undefined, { progress: 0.1, status: 'Downloading Vision Encoder (84MB)...' });
                
                const visionBuffer = await fetchFile(joinUrl(dataBaseUrl, 'model/vision_encoder_quantized.onnx'), 'Vision Model');
                
                send('PROGRESS', undefined, { progress: 0.9, status: 'Compiling Vision Model...' });
                visionSession = await ort.InferenceSession.create(visionBuffer, {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all'
                });
                console.log("[Worker] Vision Model Ready");
            }

            const input = new ort.Tensor('float32', payload.pixelValues, [1, 3, 224, 224]);
            const results = await visionSession.run({ pixel_values: input });
            send('EMBED_RESULT', id, { embedding: Array.from(results.embeds.data) });
        }

    } catch (err) {
        console.error("[Worker Error]", err);
        send('EMBED_ERROR', id, { error: err.message });
    }
};

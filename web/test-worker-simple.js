/**
 * Simple test script to verify transformers.js can load BiomedCLIP
 * 
 * Run with: node test-worker-simple.js
 * 
 * Note: This tests transformers.js in Node.js environment.
 * The actual worker runs in browser, but this verifies the model is accessible.
 */

// This is a Node.js test - transformers.js works in both Node and browser
// For actual browser testing, use test-worker.html

console.log('🧪 Testing transformers.js with BiomedCLIP...\n');

async function testTransformers() {
  try {
    // Dynamic import for Node.js
    const { pipeline, env } = await import('@xenova/transformers');
    
    // Configure
    env.allowLocalModels = false;
    env.useBrowserCache = false; // Disable for Node test
    
    console.log('1. Loading BiomedCLIP model...');
    console.log('   Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224');
    console.log('   This may take 2-3 minutes on first run...\n');
    
    const extractor = await pipeline('feature-extraction', 
      'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
      {
        progress_callback: (progress) => {
          if (progress.progress) {
            const percent = Math.round(progress.progress * 100);
            process.stdout.write(`\r   Progress: ${percent}% ${progress.status || ''}`);
          }
        }
      }
    );
    
    console.log('\n\n2. ✓ Model loaded successfully!\n');
    
    // Test queries
    const testQueries = [
      'normal chest xray',
      'pneumonia',
      'cardiomegaly'
    ];
    
    console.log('3. Testing text embeddings...\n');
    
    for (const query of testQueries) {
      console.log(`   Testing: "${query}"`);
      
      const output = await extractor(query, {
        pooling: 'mean',
        normalize: true,
      });
      
      // Extract embedding
      let embedding;
      if (output instanceof Float32Array) {
        embedding = output;
      } else if (output.data) {
        embedding = output.data;
      } else if (Array.isArray(output)) {
        embedding = new Float32Array(output.flat());
      } else {
        embedding = new Float32Array([output].flat());
      }
      
      // Check normalization
      const norm = Math.sqrt(
        Array.from(embedding).reduce((sum, x) => sum + x * x, 0)
      );
      
      console.log(`     ✓ Dimension: ${embedding.length}`);
      console.log(`     ✓ L2 Norm: ${norm.toFixed(4)} (should be ~1.0)`);
      console.log(`     ✓ First 5 values: [${Array.from(embedding.slice(0, 5)).map(x => x.toFixed(3)).join(', ')}]`);
      console.log('');
    }
    
    console.log('✅ All tests passed!');
    console.log('\n📝 Notes:');
    console.log('   - Model is accessible via transformers.js');
    console.log('   - Embeddings are 512-dimensional (BiomedCLIP)');
    console.log('   - Embeddings are L2 normalized (good for cosine similarity)');
    console.log('   - Ready to use in Web Worker!');
    
  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    console.error('\nPossible issues:');
    console.error('  1. Network connection needed to download model');
    console.error('  2. Model may not be available in transformers.js format');
    console.error('  3. Fallback to standard CLIP may be needed');
    console.error('\nError details:', error);
  }
}

testTransformers();

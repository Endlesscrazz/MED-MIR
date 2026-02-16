/**
 * image-processor.ts
 * Prepares a raw browser image for the BiomedCLIP Vision Encoder.
 */

export async function preprocessImage(file: File): Promise<Float32Array> {
  const IMG_SIZE = 224;
  
  // 1. Load image into HTML Image object
  const img = await loadImage(file);
  
  // 2. Create a canvas to resize the image
  const canvas = document.createElement('canvas');
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error("Could not get canvas context");

  // Draw image to canvas (this performs resizing)
  ctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);
  
  // 3. Get pixel data
  const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;

  // 4. Normalize (BiomedCLIP / ImageNet stats)
  // These MUST match the values used in the Python script
  const mean = [0.48145466, 0.4578275, 0.40821073];
  const std = [0.26862954, 0.26130258, 0.27577711];

  // We need to output [Channel, Height, Width] format (3 * 224 * 224)
  const floatArray = new Float32Array(3 * IMG_SIZE * IMG_SIZE);

  for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    // Canvas gives us [R, G, B, A, R, G, B, A...]
    const r = imageData[i * 4] / 255.0;
    const g = imageData[i * 4 + 1] / 255.0;
    const b = imageData[i * 4 + 2] / 255.0;

    // Apply normalization and store in Planar format [RRR... GGG... BBB...]
    floatArray[i] = (r - mean[0]) / std[0];                   // Red channel
    floatArray[i + IMG_SIZE * IMG_SIZE] = (g - mean[1]) / std[1]; // Green
    floatArray[i + 2 * IMG_SIZE * IMG_SIZE] = (b - mean[2]) / std[2]; // Blue
  }

  return floatArray;
}

function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}
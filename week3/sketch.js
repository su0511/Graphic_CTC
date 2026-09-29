let video;
let camShader;
let faceDetector;
let handLandmarker;
let trackingReady = false;
let trackingFailed = false;
let cameraReady = false;
let lastDetection = -Infinity;
let lastVideoTime = -1;
let faceSeenAt = -Infinity;
let targetStrength = 0;
let strength = 0;
let faceCenter = [0.5, 0.5];
let faceRadius = [0.2, 0.25];

const VISION_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models";

function setStatus(message) {
  const status = document.getElementById("status");
  if (status.textContent !== message) status.textContent = message;
}

function setup() {
  createCanvas(640, 480, WEBGL).parent("camera");
  // Inline shaders avoid file:// fetch restrictions when opening index.html directly.
  camShader = createShader(
    document.getElementById("vertex-shader").textContent,
    document.getElementById("fragment-shader").textContent
  );
  pixelDensity(1);
  noStroke();
  video = createVideo([]);
  video.size(640, 480);
  video.hide();
  video.elt.muted = true;
  video.elt.setAttribute("playsinline", "");
  startCamera();
}

async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera access is unavailable here. Open this page with Live Server (localhost) or HTTPS.");
    return;
  }
  setStatus("Allow camera access in your browser to begin.");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    video.elt.srcObject = stream;
    await video.elt.play();
    cameraReady = true;
    setStatus("Camera ready. Loading face and hand tracking...");
    loadTracking();
  } catch (error) {
    console.error("Camera unavailable:", error);
    setStatus("Camera unavailable. Allow camera access, check that it is not in use, then reload.");
  }
}

async function loadTracking() {
  try {
    const { FilesetResolver, FaceDetector, HandLandmarker } = await import(`${VISION_URL}/vision_bundle.mjs`);
    const vision = await FilesetResolver.forVisionTasks(`${VISION_URL}/wasm`);
    faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: `${MODEL_URL}/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite` },
      runningMode: "VIDEO",
      minDetectionConfidence: 0.5,
    });
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: `${MODEL_URL}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    trackingReady = true;
    setStatus("Show your face, then bring a hand close to it.");
  } catch (error) {
    failTracking(error);
  }
}

function failTracking(error) {
  console.error("Tracking unavailable:", error);
  trackingReady = false;
  trackingFailed = true;
  targetStrength = 0;
  setStatus("Tracking unavailable. The original wave is still running. Check your connection and reload.");
}

// Distances use the face's size so interaction works at different camera distances.
// The nearest visible hand landmark (including fingertips) controls the wave.
function handProximity(hands, center, radius) {
  let nearest = Infinity;
  for (const hand of hands) {
    for (const point of hand) {
      const dx = (point.x - center[0]) / radius[0];
      const dy = (point.y - center[1]) / radius[1];
      nearest = Math.min(nearest, Math.hypot(dx, dy));
    }
  }
  const t = Math.max(0, Math.min(1, (2.5 - nearest) / 1.6));
  return t * t * (3 - 2 * t);
}

function updateTracking(now) {
  const source = video.elt;
  // Limit inference to about 12 fps; rendering and easing continue between detections.
  if (!trackingReady || now - lastDetection < 80 || source.currentTime === lastVideoTime) return;
  lastDetection = now;
  lastVideoTime = source.currentTime;
  try {
    const faces = faceDetector.detectForVideo(source, now).detections;
    const hands = handLandmarker.detectForVideo(source, now).landmarks;
    // Use the largest face if several people are visible.
    const face = faces.reduce((largest, item) => {
      const box = item.boundingBox;
      if (!box) return largest;
      return !largest || box.width * box.height > largest.width * largest.height ? box : largest;
    }, null);
    if (face) {
      const center = [(face.originX + face.width / 2) / source.videoWidth,
        (face.originY + face.height / 2) / source.videoHeight];
      const radius = [Math.max(0.025, face.width * 0.6 / source.videoWidth),
        Math.max(0.025, face.height * 0.7 / source.videoHeight)];
      const blend = now - faceSeenAt > 400 ? 1 : 0.4;
      faceCenter = faceCenter.map((value, i) => value + (center[i] - value) * blend);
      faceRadius = faceRadius.map((value, i) => value + (radius[i] - value) * blend);
      faceSeenAt = now;
      targetStrength = handProximity(hands, center, radius);
      setStatus(hands.length ? "Move your hand closer to your face for stronger waves. Move it away to release." : "Face detected. Raise a hand and bring it close to your face.");
    } else {
      targetStrength = 0;
      setStatus("Face the camera and keep your face visible.");
    }
  } catch (error) {
    failTracking(error);
  }
}

function draw() {
  background(0);
  if (!cameraReady || video.elt.readyState < 2) return;
  const now = performance.now();
  updateTracking(now);
  if (now - faceSeenAt > 400 || trackingFailed) targetStrength = 0;
  // Fast response, slower release. Delta time keeps the fade consistent across devices.
  const easing = 1 - Math.exp(-Math.min(deltaTime, 100) / (targetStrength > strength ? 140 : 420));
  strength += (targetStrength - strength) * easing;
  shader(camShader);
  camShader.setUniform("tex0", video);
  camShader.setUniform("u_time", millis() / 1000.0);
  camShader.setUniform("u_faceCenter", faceCenter);
  camShader.setUniform("u_faceRadius", faceRadius);
  camShader.setUniform("u_handStrength", strength);
  rect(-width / 2, -height / 2, width, height);
}

window.addEventListener("pagehide", () => {
  video?.elt.srcObject?.getTracks().forEach((track) => track.stop());
  faceDetector?.close();
  handLandmarker?.close();
});

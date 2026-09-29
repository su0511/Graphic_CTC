#ifdef GL_ES
precision mediump float;
#endif

uniform sampler2D tex0;
uniform float u_time;
uniform vec2 u_faceCenter;
uniform vec2 u_faceRadius;
uniform float u_handStrength;
varying vec2 vTexCoord;

void main() {
  vec2 uv = vTexCoord;
  uv.y = 1.0 - uv.y;

  // Layer 1: the original, always-on horizontal wave.
  uv.x += sin(uv.y * 20.0 + u_time * 2.0) * 0.02;

  // Work in webcam texture coordinates, matching the detector's coordinates.
  // An elliptical feathered mask keeps the second wave near the face.
  vec2 local = (uv - u_faceCenter) / max(u_faceRadius, vec2(0.025));
  float mask = 1.0 - smoothstep(0.55, 1.25, length(local));

  // Layer 2: denser vertical waves, controlled by the hand's proximity.
  float wave = sin(local.x * 11.0 - u_time * 7.0);
  uv.y += wave * u_faceRadius.y * 0.24 * mask * u_handStrength;

  gl_FragColor = vec4(texture2D(tex0, clamp(uv, 0.001, 0.999)).rgb, 1.0);
}

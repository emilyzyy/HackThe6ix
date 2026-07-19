import { interpolate, Easing } from "remotion";
import { EASE, EASE_IN_OUT } from "../theme";

export const bez = (p: readonly [number, number, number, number]) =>
  Easing.bezier(p[0], p[1], p[2], p[3]);

export const easeOut = bez(EASE);
export const easeInOut = bez(EASE_IN_OUT);

// A local 0..1 progress across [start, end] frames, clamped, eased.
// Guards against a zero/negative-length range (returns a hard step).
export const ramp = (
  frame: number,
  start: number,
  end: number,
  easing = easeOut
) => {
  if (end <= start) return frame >= start ? 1 : 0;
  return interpolate(frame, [start, end], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing,
  });
};

// Symmetric appear/hold/disappear envelope for a text or element.
export const appear = (
  frame: number,
  start: number,
  inDur: number,
  hold: number,
  outDur: number,
  easing = easeOut
) => {
  const a = ramp(frame, start, start + inDur, easing);
  if (outDur <= 0) return a; // hold indefinitely (no fade-out)
  const outStart = start + inDur + hold;
  const b = interpolate(frame, [outStart, outStart + outDur], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeInOut,
  });
  return Math.min(a, b);
};

export const mix = (a: number, b: number, t: number) => a + (b - a) * t;

// Deterministic pseudo-random in [0,1) from an integer seed (no Math.random).
export const rand = (seed: number) => {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};

// Centralised visual identity. Black-and-yellow LEGO-brick world.
// Timing lives in demoConfig.ts; this file is colour, type, and motion feel.

export const FPS = 30;

export const COLORS = {
  // Backgrounds
  black: "#050505",
  nearBlack: "#0A0A0B",
  panel: "#101012",
  // Brand
  yellow: "#FFD500", // LEGO-like warm yellow
  yellowDeep: "#E8B800",
  yellowSoft: "rgba(255, 213, 0, 0.14)",
  // Text
  offWhite: "#F4F1E8",
  gray: "#7A7A7E",
  grayDim: "#4A4A4E",
  // Real LEGO piece colours (used sparingly for physical bricks)
  brickRed: "#C91A09",
  brickBlue: "#0055BF",
  brickGreen: "#237841",
  brickYellow: "#F2CD37",
  brickWhite: "#E8E6DE",
  brickGray: "#5B5C60",
} as const;

// A strong, restrained modern sans available on macOS render hosts.
export const FONT =
  '"Helvetica Neue", "Inter", "Segoe UI", Arial, sans-serif';

export const TYPE = {
  // Big restrained statements
  statement: {
    fontFamily: FONT,
    fontWeight: 800,
    letterSpacing: "-0.02em",
    lineHeight: 1.02,
    color: COLORS.offWhite,
  },
  // Small uppercase technical labels
  technical: {
    fontFamily: FONT,
    fontWeight: 600,
    letterSpacing: "0.22em",
    textTransform: "uppercase" as const,
    color: COLORS.gray,
  },
} as const;

// Motion feel constants — one shared vocabulary so nothing feels arbitrary.
export const MOTION = {
  // A confident spring-like snap used when pieces lock onto studs.
  snap: { damping: 14, mass: 0.7, stiffness: 150 },
  // Softer settle for camera / world moves.
  settle: { damping: 22, mass: 1, stiffness: 90 },
  // Global intensity dial (0..1) for how energetic motion is.
  intensity: 1,
} as const;

// A cubic ease used everywhere for hand-authored interpolations.
export const EASE = [0.22, 1, 0.36, 1] as const; // easeOutQuint-ish
export const EASE_IN_OUT = [0.65, 0, 0.35, 1] as const;

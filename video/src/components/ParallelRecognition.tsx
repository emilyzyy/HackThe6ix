import React from "react";
import { useCurrentFrame } from "remotion";
import { COLORS, FONT } from "../theme";
import { LegoBrick } from "./LegoBrick";
import { ramp, rand } from "../lib/anim";

// Ten parallel yellow processing lanes; crops distribute across them and
// results return concurrently. An engineering flex, only lasts a moment.
export const ParallelRecognition: React.FC<{
  lanes?: number;
  start: number;
  width?: number;
  height?: number;
}> = ({ lanes = 10, start, width = 1200, height = 520 }) => {
  const frame = useCurrentFrame();
  const laneH = height / lanes;
  const PIECE = [
    COLORS.brickRed,
    COLORS.brickBlue,
    COLORS.brickYellow,
    COLORS.brickWhite,
    COLORS.brickGreen,
  ];
  return (
    <div style={{ position: "relative", width, height }}>
      {Array.from({ length: lanes }).map((_, li) => {
        const on = ramp(frame, start + li * 1.5, start + li * 1.5 + 8);
        const y = li * laneH + laneH / 2;
        return (
          <React.Fragment key={li}>
            {/* lane rail */}
            <div
              style={{
                position: "absolute",
                left: 0,
                top: y,
                width,
                height: 2,
                background: `linear-gradient(90deg, rgba(255,213,0,${
                  0.05 * on
                }), rgba(255,213,0,${0.5 * on}), rgba(255,213,0,${0.05 * on}))`,
              }}
            />
            {/* travelling crops */}
            {Array.from({ length: 2 }).map((__, ci) => {
              const seed = li * 10 + ci;
              const speed = 3 + rand(seed) * 2;
              const local = (frame - start - li * 1.5) * speed;
              const cycle = width + 200;
              const x = ((local + rand(seed) * cycle) % cycle) - 100;
              const cols = 1 + Math.floor(rand(seed) * 2);
              const rows = 1 + Math.floor(rand(seed + 1) * 2);
              const past = x > width * 0.62;
              return (
                <div
                  key={ci}
                  style={{
                    position: "absolute",
                    left: x,
                    top: y - laneH * 0.32,
                    opacity: on * (local > 0 ? 1 : 0),
                    transform: "scale(0.7)",
                  }}
                >
                  <LegoBrick
                    studs={[cols, rows]}
                    unit={14}
                    color={PIECE[seed % PIECE.length]}
                    glow={past ? 1 : 0}
                  />
                </div>
              );
            })}
            {/* result tick on the right */}
            <div
              style={{
                position: "absolute",
                right: -6,
                top: y - 4,
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: COLORS.yellow,
                opacity: on,
                boxShadow: `0 0 12px ${COLORS.yellow}`,
              }}
            />
          </React.Fragment>
        );
      })}
      {/* left intake bracket */}
      <div
        style={{
          position: "absolute",
          left: -2,
          top: 0,
          bottom: 0,
          width: 3,
          background: `linear-gradient(180deg, transparent, ${COLORS.yellow}, transparent)`,
          opacity: ramp(frame, start, start + 10),
        }}
      />
      <div
        style={{
          position: "absolute",
          right: 0,
          bottom: -34,
          fontFamily: FONT,
          fontSize: 15,
          letterSpacing: "0.24em",
          fontWeight: 600,
          color: COLORS.gray,
          opacity: ramp(frame, start + 6, start + 16),
        }}
      >
        10× PARALLEL RECOGNITION
      </div>
    </div>
  );
};

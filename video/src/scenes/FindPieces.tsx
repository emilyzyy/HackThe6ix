import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { BigStatement } from "../components/BigStatement";
import { TechnicalLabel } from "../components/TechnicalLabel";
import { LegoBrick } from "../components/LegoBrick";
import { COLORS } from "../theme";
import { ramp } from "../lib/anim";
import { COPY } from "../demoConfig";

// Deterministic piece layout so masks align exactly with pieces. Two adjacent
// WHITE pieces (touching) get two separate masks — the differentiating moment.
const PIECES = [
  { x: 300, y: 300, studs: [2, 4] as [number, number], color: COLORS.brickRed, rot: -12 },
  { x: 560, y: 250, studs: [2, 2] as [number, number], color: COLORS.brickBlue, rot: 8 },
  { x: 760, y: 420, studs: [1, 6] as [number, number], color: COLORS.brickYellow, rot: 20 },
  { x: 1120, y: 300, studs: [2, 3] as [number, number], color: COLORS.brickGreen, rot: -6 },
  { x: 1380, y: 430, studs: [2, 2] as [number, number], color: COLORS.brickGray, rot: 14 },
  // two touching white pieces
  { x: 980, y: 560, studs: [2, 2] as [number, number], color: COLORS.brickWhite, rot: 4, touch: true },
  { x: 1090, y: 566, studs: [2, 2] as [number, number], color: COLORS.brickWhite, rot: 4, touch: true },
  { x: 480, y: 560, studs: [2, 2] as [number, number], color: COLORS.brickRed, rot: -18 },
];
const UNIT = 30;

export const FindPieces: React.FC = () => {
  const frame = useCurrentFrame();
  const outFade = ramp(240 - frame, 0, 14);
  const expand = ramp(frame, 0, 12); // one view expands to RGB
  const boundary = ramp(frame, 20, 60); // workspace boundary traces once

  return (
    <AbsoluteFill style={{ opacity: 1 }}>
      <Backdrop />
      {/* the RGB view panel */}
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <div
          style={{
            position: "relative",
            width: 1500,
            height: 760,
            transform: `scale(${0.9 + expand * 0.1})`,
            opacity: expand,
            borderRadius: 10,
            background: "linear-gradient(180deg,#111014,#0a0a0c)",
            border: "1px solid rgba(255,255,255,0.08)",
            overflow: "hidden",
          }}
        >
          {/* workspace boundary trace */}
          <svg style={{ position: "absolute", inset: 0 }} width={1500} height={760}>
            <rect
              x={70}
              y={80}
              width={1360}
              height={600}
              rx={16}
              fill="none"
              stroke={COLORS.yellow}
              strokeWidth={2}
              strokeDasharray={2000}
              strokeDashoffset={2000 * (1 - boundary)}
              opacity={0.55}
            />
          </svg>

          {/* pieces */}
          {PIECES.map((p, i) => (
            <div key={i} style={{ position: "absolute", left: p.x, top: p.y, transform: `rotate(${p.rot}deg)` }}>
              <LegoBrick studs={p.studs} unit={UNIT} color={p.color} />
            </div>
          ))}

          {/* segmentation masks snap on, staggered */}
          {PIECES.map((p, i) => {
            const t = ramp(frame, 55 + i * 6, 55 + i * 6 + 10);
            if (t <= 0) return null;
            const w = p.studs[0] * UNIT;
            const h = p.studs[1] * UNIT;
            return (
              <div
                key={`m${i}`}
                style={{
                  position: "absolute",
                  left: p.x - 8,
                  top: p.y - 8,
                  width: w + 16,
                  height: h + 16,
                  transform: `rotate(${p.rot}deg) scale(${0.6 + t * 0.4})`,
                  border: `2px solid ${COLORS.yellow}`,
                  borderRadius: 8,
                  background: `rgba(255,213,0,${0.16 * t})`,
                  boxShadow: `0 0 ${18 * t}px rgba(255,213,0,${0.4 * t})`,
                  opacity: t,
                }}
              />
            );
          })}
        </div>
      </AbsoluteFill>

      {/* text */}
      <div style={{ position: "absolute", top: 60, width: "100%", display: "flex", justifyContent: "center" }}>
        <BigStatement lines={[COPY.pieces.fromPile]} start={20} inDur={12} hold={40} outDur={14} size={54} />
      </div>
      <div style={{ position: "absolute", top: 60, width: "100%", display: "flex", justifyContent: "center" }}>
        <BigStatement lines={[COPY.pieces.toPieces]} start={110} inDur={12} hold={90} outDur={16} size={54} accentLineIndex={0} />
      </div>

      {/* the touching-white moment, very brief */}
      <div style={{ position: "absolute", bottom: 120, width: "100%", display: "flex", justifyContent: "center" }}>
        <TechnicalLabel text={COPY.pieces.evenTouch} start={120} inDur={10} hold={44} outDur={12} size={22} active />
      </div>
    </AbsoluteFill>
  );
};

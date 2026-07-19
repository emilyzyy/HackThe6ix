import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { BigStatement } from "../components/BigStatement";
import { TechnicalLabel } from "../components/TechnicalLabel";
import { LegoBrick } from "../components/LegoBrick";
import { EvidenceMarker } from "../components/EvidenceMarker";
import { ParallelRecognition } from "../components/ParallelRecognition";
import { COLORS, FONT } from "../theme";
import { ramp, appear } from "../lib/anim";
import { COPY } from "../demoConfig";

export const IdentifyPieces: React.FC = () => {
  const frame = useCurrentFrame();
  const outFade = ramp(360 - frame, 0, 16);

  const pullIn = ramp(frame, 0, 26);
  const guessT = appear(frame, 24, 12, 40, 12);
  const evidenceStart = 80;
  const lock = ramp(frame, 170, 190);
  const colorT = appear(frame, 195, 12, 60, 14);
  const parallelStart = 250;
  const pieceExit = ramp(frame, 236, 258); // piece + verdict recede before lanes

  return (
    <AbsoluteFill style={{ opacity: 1 }}>
      <Backdrop />

      {/* representative piece pulled forward */}
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <div
          style={{
            position: "relative",
            transform: `scale(${0.6 + pullIn * 0.4}) translateY(${(1 - pullIn) * 40}px)`,
            opacity: pullIn * (1 - pieceExit),
          }}
        >
          <LegoBrick studs={[2, 4]} unit={64} color={COLORS.brickRed} glow={lock} />

          {/* evidence converges inward */}
          {frame > evidenceStart && frame < 250 && (
            <>
              <EvidenceMarker label={COPY.pieces.evidence[0]} angleDeg={-125} radius={260} start={evidenceStart} />
              <EvidenceMarker label={COPY.pieces.evidence[1]} angleDeg={-40} radius={280} start={evidenceStart + 14} />
              <EvidenceMarker label={COPY.pieces.evidence[2]} angleDeg={70} radius={250} start={evidenceStart + 28} />
            </>
          )}
        </div>
      </AbsoluteFill>

      {/* tentative guess -> locked identity */}
      <div style={{ position: "absolute", top: 120, width: "100%", display: "flex", justifyContent: "center" }}>
        {lock < 0.5 ? (
          <div style={{ fontFamily: FONT, fontWeight: 800, fontSize: 58, letterSpacing: "0.02em", color: COLORS.gray, opacity: guessT }}>
            {COPY.pieces.guess}
          </div>
        ) : (
          <div style={{ display: "flex", alignItems: "center", gap: 20, opacity: lock, transform: `scale(${0.9 + lock * 0.1})` }}>
            <span style={{ fontFamily: FONT, fontWeight: 800, fontSize: 62, color: COLORS.offWhite }}>{COPY.pieces.verified}</span>
            <span style={{ color: COLORS.yellow, fontSize: 54, fontWeight: 800 }}>✓</span>
          </div>
        )}
      </div>

      {/* independent color result */}
      <div style={{ position: "absolute", bottom: 210, width: "100%", display: "flex", justifyContent: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14, opacity: colorT }}>
          <div style={{ width: 22, height: 22, borderRadius: 4, background: COLORS.brickRed }} />
          <span style={{ fontFamily: FONT, fontWeight: 700, fontSize: 30, letterSpacing: "0.2em", color: COLORS.offWhite }}>{COPY.pieces.color}</span>
        </div>
      </div>

      {/* verify statement */}
      <div style={{ position: "absolute", bottom: 90, width: "100%", display: "flex", justifyContent: "center" }}>
        <BigStatement lines={[COPY.pieces.verifyA, COPY.pieces.verifyB]} start={120} inDur={12} hold={90} outDur={16} size={40} accentLineIndex={1} />
      </div>

      {/* parallel recognition flex */}
      {frame > parallelStart - 10 && (
        <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
          <ParallelRecognition start={parallelStart} width={1200} height={480} />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

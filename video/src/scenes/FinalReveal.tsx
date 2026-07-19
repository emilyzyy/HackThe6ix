import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { BigStatement } from "../components/BigStatement";
import { BuildModel } from "../components/BuildModel";
import { LegoBin } from "../components/LegoBin";
import { COLORS } from "../theme";
import { ramp } from "../lib/anim";
import { COPY } from "../demoConfig";

export const FinalReveal: React.FC = () => {
  const frame = useCurrentFrame();

  const explode = ramp(frame, 60, 150); // burst apart
  const recede = ramp(frame, 90, 230); // drift back toward the box
  const stop = frame > 230 && frame < 260;
  const reassemble = ramp(frame, 262, 360); // reverse: snap back together
  const lock = ramp(frame, 350, 400);

  // Net model state: exploded during burst, back to assembled after reverse.
  const modelExplode = Math.max(0, explode - reassemble);
  const modelAssemble = 1 - modelExplode;
  // Recede toward the bin (down/back), then return to centre.
  const drift = recede * (1 - reassemble);
  const scale = 1 - drift * 0.4 + lock * 0.05;
  const ty = drift * 180;

  const binIn = ramp(frame, 150, 210) * (1 - reassemble);
  const titleStart = 410;
  // As the title arrives, lift + shrink the locked model so text has room.
  const titlePush = ramp(frame, titleStart - 6, titleStart + 40);
  const fadeOut = ramp(630 - frame, 0, 40);

  return (
    <AbsoluteFill style={{ opacity: fadeOut, background: "#020202" }}>
      <Backdrop studGrid={false} vignette={1} />

      {/* the yellow bin, briefly threatening to swallow the pieces */}
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "flex-end" }}>
        <div style={{ marginBottom: 40, opacity: binIn, transform: `translateY(${(1 - binIn) * 120}px) scale(0.9)` }}>
          <LegoBin width={560} height={320} open={1} />
        </div>
      </AbsoluteFill>

      {/* the final model: explode -> recede -> reverse -> lock, then lift */}
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <div
          style={{
            transform: `translateY(${ty - titlePush * 235}px) scale(${scale * (1 - titlePush * 0.42)})`,
            filter: stop ? "brightness(0.9)" : undefined,
            opacity: interpolate(frame, [0, 40], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          <BuildModel assemble={modelAssemble} explode={modelExplode} glow={lock} unit={52} />
        </div>
      </AbsoluteFill>

      {/* title — mirrors the opening world, sits below the lifted model */}
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <div style={{ transform: "translateY(120px)", opacity: titlePush }}>
          <BigStatement lines={[COPY.final.titleA, COPY.final.titleB]} start={titleStart} inDur={18} hold={200} outDur={0} size={118} accentLineIndex={1} weight={900} />
          <div style={{ marginTop: 34 }}>
            <BigStatement lines={[COPY.final.subA, COPY.final.subB]} start={titleStart + 40} inDur={16} hold={150} outDur={0} size={32} />
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

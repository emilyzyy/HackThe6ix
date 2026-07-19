import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { BigStatement } from "../components/BigStatement";
import { CameraFrame } from "../components/CameraFrame";
import { LegoBrick } from "../components/LegoBrick";
import { COLORS } from "../theme";
import { ramp } from "../lib/anim";
import { COPY } from "../demoConfig";

// The same red brick seen in 3 views detaches and snaps into ONE physical
// object on the shared table, then the effect accelerates across the pile.
export const FusePieces: React.FC = () => {
  const frame = useCurrentFrame();
  const outFade = ramp(210 - frame, 0, 14);
  const targetX = 960;
  const targetY = 560;

  // three source detections converge to the shared position
  const views = [
    { x: 430, y: 250, ox: 0.3, oy: 0.55 },
    { x: 960, y: 200, ox: 0.6, oy: 0.4 },
    { x: 1490, y: 250, ox: 0.45, oy: 0.62 },
  ];
  const detach = ramp(frame, 30, 60);
  const move = ramp(frame, 55, 110);
  const snap = ramp(frame, 100, 120);
  const accelerate = ramp(frame, 130, 200);

  return (
    <AbsoluteFill style={{ opacity: 1 }}>
      <Backdrop />

      {/* the three views */}
      {views.map((v, i) => {
        const appearT = ramp(frame, i * 8, i * 8 + 14);
        return (
          <div key={i} style={{ position: "absolute", left: v.x, top: v.y, transform: "translate(-50%,-50%)", opacity: appearT * (1 - move * 0.5) }}>
            <CameraFrame width={250} height={165} label={`VIEW ${i + 1}`} active>
              <div style={{ position: "absolute", left: `${v.ox * 100}%`, top: `${v.oy * 100}%`, transform: "translate(-50%,-50%)", opacity: 1 - detach }}>
                <LegoBrick studs={[2, 4]} unit={11} color={COLORS.brickRed} />
              </div>
            </CameraFrame>
          </div>
        );
      })}

      {/* the detached observations flying toward one shared object */}
      {views.map((v, i) => {
        const startX = v.x + (v.ox - 0.5) * 250;
        const startY = v.y + (v.oy - 0.5) * 165;
        const x = startX + (targetX - startX) * move;
        const y = startY + (targetY - startY) * move;
        const s = 1 - move * 0.2 + snap * 0.2;
        return (
          <div key={`o${i}`} style={{ position: "absolute", left: x, top: y, transform: `translate(-50%,-50%) scale(${s})`, opacity: detach * (1 - snap * (i > 0 ? 1 : 0)) }}>
            <LegoBrick studs={[2, 4]} unit={22} color={COLORS.brickRed} glow={snap} />
          </div>
        );
      })}

      {/* the accelerated resolution across the rest of the pile */}
      {accelerate > 0 &&
        Array.from({ length: 14 }).map((_, i) => {
          const a = (i / 14) * Math.PI * 2;
          const rad = 260 + (i % 3) * 70;
          const x = targetX + Math.cos(a) * rad;
          const y = targetY + Math.sin(a) * rad * 0.55;
          const on = ramp(frame, 130 + i * 4, 130 + i * 4 + 10);
          const colors = [COLORS.brickBlue, COLORS.brickYellow, COLORS.brickGreen, COLORS.brickWhite, COLORS.brickGray];
          return (
            <div key={`p${i}`} style={{ position: "absolute", left: x, top: y, transform: "translate(-50%,-50%)", opacity: on * accelerate }}>
              <LegoBrick studs={[1 + (i % 2), 2]} unit={18} color={colors[i % colors.length]} glow={on} />
            </div>
          );
        })}

      <div style={{ position: "absolute", top: 70, width: "100%", display: "flex", justifyContent: "center" }}>
        <BigStatement lines={[COPY.pieces.multiA, COPY.pieces.multiB]} start={70} inDur={14} hold={90} outDur={16} size={54} accentLineIndex={1} />
      </div>
    </AbsoluteFill>
  );
};

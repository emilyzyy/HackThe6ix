import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { BigStatement } from "../components/BigStatement";
import { TechnicalLabel } from "../components/TechnicalLabel";
import { TableMap } from "../components/TableMap";
import { CameraFrame } from "../components/CameraFrame";
import { LegoBrick } from "../components/LegoBrick";
import { COLORS } from "../theme";
import { ramp, rand } from "../lib/anim";
import { COPY, INTENSITY } from "../demoConfig";

export const UnderstandScan: React.FC = () => {
  const frame = useCurrentFrame();
  const outFade = ramp(480 - frame, 0, 16);

  const planeFit = ramp(frame, 10, 110); // point cloud -> clean plane
  const tilt = ramp(frame, 90, 150); // rotate toward top-down
  const rotX = (1 - tilt) * 58; // start tilted, settle flat-ish top-down

  // three-view convergence (120-240)
  const converge = ramp(frame, 150, 235);
  // hundreds -> five (300-430)
  const hundreds = ramp(frame, 300, 350);
  const collapse = ramp(frame, 360, 420);

  const centerX = 960;
  const tableCX = centerX;
  const tableCY = 620;

  return (
    <AbsoluteFill style={{ opacity: 1 }}>
      <Backdrop />

      {/* The table map, tilting from perspective toward top-down */}
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <div
          style={{
            perspective: 1400,
            transform: `translateY(${60 - tilt * 20}px)`,
          }}
        >
          <div style={{ transform: `rotateX(${rotX}deg)`, transformStyle: "preserve-3d" }}>
            <TableMap width={780} height={520} fit={planeFit} gridReveal={planeFit} points={70}>
              {/* one red brick sitting on the table */}
              <div style={{ position: "absolute", left: "46%", top: "50%", transform: "translate(-50%,-50%)" }}>
                <LegoBrick studs={[2, 4]} unit={26} color={COLORS.brickRed} />
              </div>
            </TableMap>
          </div>
        </div>
      </AbsoluteFill>

      {/* THREE cameras seeing the same red brick at different pixel positions */}
      {frame > 120 && frame < 300 &&
        [0, 1, 2].map((i) => {
          const on = ramp(frame, 125 + i * 10, 125 + i * 10 + 16);
          const px = [430, 960, 1480][i];
          const py = [230, 175, 235][i];
          // the red object sits at a different in-frame position per view
          const objX = [0.28, 0.62, 0.44][i];
          const objY = [0.6, 0.35, 0.7][i];
          return (
            <div key={i} style={{ position: "absolute", left: px, top: py, transform: "translate(-50%,-50%)", opacity: on }}>
              <CameraFrame width={260} height={170} label={`VIEW ${i + 1}`} active>
                <div style={{ position: "absolute", left: `${objX * 100}%`, top: `${objY * 100}%`, transform: "translate(-50%,-50%)" }}>
                  <LegoBrick studs={[2, 4]} unit={12} color={COLORS.brickRed} />
                </div>
              </CameraFrame>
              {/* projection line down to the shared table position */}
              <svg style={{ position: "absolute", left: 130, top: 85, overflow: "visible", pointerEvents: "none" }}>
                <line
                  x1={0}
                  y1={0}
                  x2={(tableCX - px) * converge}
                  y2={(tableCY - py) * converge}
                  stroke={COLORS.yellow}
                  strokeWidth={1.3}
                  strokeDasharray="5 5"
                  opacity={0.55 * on}
                />
              </svg>
            </div>
          );
        })}

      {/* HUNDREDS of frames imply volume, then collapse to five */}
      {frame > 290 &&
        Array.from({ length: INTENSITY.impliedFrameCount }).map((_, i) => {
          const keep = i < INTENSITY.cameraFrameCount;
          const a = (i / INTENSITY.impliedFrameCount) * Math.PI * 2 + rand(i) * 0.6;
          const rad = 360 + rand(i) * 160;
          const sx = tableCX + Math.cos(a) * rad;
          const sy = tableCY + Math.sin(a) * rad * 0.62;
          // five selected settle to clean bearings; the rest fade out
          const fa = (keep ? (i / 5) : rand(i)) * Math.PI * 2 - Math.PI / 2;
          const fRad = 330;
          const fx = tableCX + Math.cos(fa) * fRad;
          const fy = tableCY + Math.sin(fa) * fRad * 0.55;
          const x = sx + (fx - sx) * collapse;
          const y = sy + (fy - sy) * collapse;
          const opacity =
            hundreds * (keep ? 1 : 1 - collapse) * (frame > 300 ? 1 : 0);
          return (
            <div key={`f${i}`} style={{ position: "absolute", left: x, top: y, transform: "translate(-50%,-50%)", opacity }}>
              <CameraFrame width={keep ? 190 : 120} height={keep ? 124 : 78} active={keep && collapse > 0.5} label={keep ? `V${i + 1}` : undefined} />
            </div>
          );
        })}

      {/* TEXT progression */}
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "flex-start", paddingTop: 70 }}>
        <BigStatement lines={[COPY.understandScan.everyCameraA, COPY.understandScan.everyCameraB]} start={122} inDur={14} hold={70} outDur={16} size={52} accentLineIndex={1} />
      </AbsoluteFill>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "flex-start", paddingTop: 70 }}>
        <BigStatement lines={[COPY.understandScan.sameMapA, COPY.understandScan.sameMapB]} start={236} inDur={14} hold={44} outDur={16} size={52} accentLineIndex={1} />
      </AbsoluteFill>

      <div style={{ position: "absolute", bottom: 90, width: "100%", display: "flex", justifyContent: "center" }}>
        <BigStatement lines={[COPY.understandScan.hundreds]} start={300} inDur={12} hold={40} outDur={14} size={40} />
      </div>
      <div style={{ position: "absolute", bottom: 90, width: "100%", display: "flex", justifyContent: "center" }}>
        <BigStatement lines={[COPY.understandScan.five]} start={360} inDur={12} hold={80} outDur={16} size={40} accentLineIndex={0} />
      </div>
    </AbsoluteFill>
  );
};

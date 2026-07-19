import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { BigStatement } from "../components/BigStatement";
import { TechnicalLabel } from "../components/TechnicalLabel";
import { ClipSlot } from "../components/ClipSlot";
import { BrickPile } from "../components/BrickPile";
import { COLORS, FONT } from "../theme";
import { ramp, rand } from "../lib/anim";
import { ASSETS, COPY, DemoProps, useRealClip } from "../demoConfig";

// Animated fallback: LEFT stylized phone panning over the pile; RIGHT the
// capture interface with coverage filling and camera markers accumulating.
const ScanFallback: React.FC = () => {
  const frame = useCurrentFrame();
  const pan = Math.sin(frame / 46) * 120;
  const coverage = Math.min(0.97, ramp(frame, 40, 300));
  return (
    <div style={{ display: "flex", width: "100%", height: "100%" }}>
      {/* LEFT: physical scan */}
      <div style={{ flex: 1, position: "relative", background: "#0d0d0f", overflow: "hidden" }}>
        <div style={{ position: "absolute", left: "50%", top: "54%", transform: "translate(-50%,-50%)" }}>
          <BrickPile count={16} width={560} height={340} seed={4} settle={1} unit={30} />
        </div>
        {/* stylized phone */}
        <div
          style={{
            position: "absolute",
            left: `calc(50% + ${pan}px)`,
            top: 60,
            transform: "translateX(-50%) rotate(-8deg)",
            width: 150,
            height: 300,
            borderRadius: 22,
            border: `3px solid ${COLORS.yellow}`,
            background: "rgba(255,213,0,0.05)",
            boxShadow: `0 0 40px rgba(255,213,0,0.25)`,
          }}
        >
          <div style={{ position: "absolute", inset: 12, border: "1px solid rgba(255,213,0,0.4)", borderRadius: 12 }} />
          {/* scan cone */}
          <div
            style={{
              position: "absolute",
              left: "50%",
              top: "100%",
              transform: "translateX(-50%)",
              width: 0,
              height: 0,
              borderLeft: "120px solid transparent",
              borderRight: "120px solid transparent",
              borderTop: `260px solid rgba(255,213,0,0.08)`,
            }}
          />
        </div>
        <div style={{ position: "absolute", bottom: 14, left: 16 }}>
          <TechnicalLabel text="PHYSICAL SCAN" start={0} inDur={12} hold={9999} outDur={0} size={13} active marker={false} />
        </div>
      </div>

      {/* divider */}
      <div style={{ width: 2, background: "rgba(255,255,255,0.08)" }} />

      {/* RIGHT: capture interface */}
      <div style={{ flex: 1, position: "relative", background: "#08080a", overflow: "hidden" }}>
        {/* workspace coverage grid filling in */}
        <div style={{ position: "absolute", left: "50%", top: "50%", transform: "translate(-50%,-50%)", width: 520, height: 340 }}>
          <div style={{ position: "absolute", inset: 0, border: `1.5px solid rgba(255,213,0,0.4)`, borderRadius: 12 }} />
          <div
            style={{
              position: "absolute",
              inset: 6,
              borderRadius: 8,
              backgroundImage: `radial-gradient(circle, rgba(255,213,0,0.5) 1.4px, transparent 2px)`,
              backgroundSize: "34px 34px",
              clipPath: `inset(0 ${(1 - coverage) * 100}% 0 0)`,
            }}
          />
          {/* camera markers accumulate around the workspace */}
          {Array.from({ length: 5 }).map((_, i) => {
            const on = ramp(frame, 80 + i * 34, 80 + i * 34 + 14);
            const a = (i / 5) * Math.PI * 2 - Math.PI / 2;
            return (
              <div
                key={i}
                style={{
                  position: "absolute",
                  left: 260 + Math.cos(a) * 300,
                  top: 170 + Math.sin(a) * 210,
                  width: 42,
                  height: 30,
                  border: `1.5px solid ${COLORS.yellow}`,
                  borderRadius: 3,
                  opacity: on * 0.9,
                  transform: `translate(-50%,-50%) scale(${0.6 + on * 0.4})`,
                }}
              />
            );
          })}
        </div>
        <div style={{ position: "absolute", bottom: 14, left: 16 }}>
          <TechnicalLabel text={`COVERAGE ${Math.round(coverage * 100)}%`} start={0} inDur={12} hold={9999} outDur={0} size={13} active marker={false} />
        </div>
      </div>
    </div>
  );
};

export const ScanDemo: React.FC<{ demo: DemoProps }> = ({ demo }) => {
  const frame = useCurrentFrame();
  const useReal = useRealClip(demo, "scan");
  const outFade = ramp(630 - frame, 0, 18);
  const statusA = frame > 210 && frame < 360;
  const statusB = frame >= 360 && frame < 560;

  return (
    <AbsoluteFill style={{ opacity: 1 }}>
      <Backdrop vignette={0.95} />

      {/* opening text, then let the clip breathe */}
      <div style={{ position: "absolute", top: 70, width: "100%", display: "flex", justifyContent: "center" }}>
        <BigStatement
          lines={[COPY.scan.line1, COPY.scan.line2]}
          start={10}
          inDur={16}
          hold={120}
          outDur={20}
          size={46}
          accentLineIndex={1}
        />
      </div>

      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", paddingTop: 60 }}>
        <ClipSlot
          useReal={useReal}
          src={ASSETS.scanClip}
          start={8}
          width={1500}
          height={720}
          fallback={<ScanFallback />}
          overlay={
            <div style={{ position: "absolute", top: 16, left: 18 }}>
              {statusA && (
                <TechnicalLabel text={COPY.scan.statusA} start={0} inDur={10} hold={9999} outDur={0} size={15} active />
              )}
              {statusB && (
                <TechnicalLabel text={COPY.scan.statusB} start={0} inDur={10} hold={9999} outDur={0} size={15} active />
              )}
            </div>
          }
        />
      </AbsoluteFill>

      {/* end: extract-a-frame flash that hands to the technical world */}
      <FrameExtract start={560} />
    </AbsoluteFill>
  );
};

const FrameExtract: React.FC<{ start: number }> = ({ start }) => {
  const frame = useCurrentFrame();
  const grab = ramp(frame, start, start + 40);
  if (grab <= 0) return null;
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
      <div
        style={{
          width: 1500 * (1 - grab * 0.55),
          height: 720 * (1 - grab * 0.55),
          border: `2px solid ${COLORS.yellow}`,
          borderRadius: 8,
          boxShadow: `0 0 60px rgba(255,213,0,${grab * 0.4})`,
          transform: `translateY(${grab * -40}px)`,
          opacity: grab < 0.9 ? 1 : (1 - grab) * 10,
          background: "rgba(255,213,0,0.03)",
        }}
      />
    </AbsoluteFill>
  );
};

import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { TechnicalLabel } from "../components/TechnicalLabel";
import { ClipSlot } from "../components/ClipSlot";
import { AgentProgress } from "../components/AgentProgress";
import { BuildModel } from "../components/BuildModel";
import { LegoBrick } from "../components/LegoBrick";
import { COLORS, FONT } from "../theme";
import { ramp } from "../lib/anim";
import { ASSETS, COPY, DemoProps, generationIdeas, useRealClip } from "../demoConfig";

// Pipeline stage from a local frame. Two passes (suggested idea, then custom).
const stageIndex = (frame: number): number => {
  const f = frame % 420; // each build pass ~14s
  if (f < 90) return 0; // PLAN
  if (f < 250) return 1; // BUILD
  return 2; // REPAIR
};

const GenerationFallback: React.FC = () => {
  const frame = useCurrentFrame();
  const pass = frame < 420 ? 0 : 1;
  const f = frame % 420;
  const idea = generationIdeas[pass === 0 ? 2 : 0]; // RACE CAR, then ROVER
  const planT = ramp(f, 20, 80);
  const assemble = ramp(f, 60, 230); // model starts forming during PLAN
  const repair = ramp(f, 250, 330);
  const lock = ramp(f, 330, 380);
  // concept scaffold visible during PLAN so the stage is never empty
  const planGhost = Math.min(ramp(f, 24, 70), 1 - assemble);

  return (
    <div style={{ position: "absolute", inset: 0, background: "#0a0a0c", overflow: "hidden" }}>
      {/* inventory dock */}
      <div style={{ position: "absolute", left: 30, top: 30, bottom: 30, width: 170, borderRight: "1px solid rgba(255,213,0,0.2)", paddingRight: 16 }}>
        <TechnicalLabel text="INVENTORY" start={0} inDur={10} hold={9999} outDur={0} size={12} active marker={false} />
        {[COLORS.brickRed, COLORS.brickBlue, COLORS.brickYellow, COLORS.brickGreen, COLORS.brickWhite].map((c, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 12, opacity: 0.8 }}>
            <div style={{ width: 12, height: 12, background: c, borderRadius: 2 }} />
            <div style={{ width: 60, height: 6, background: "rgba(255,255,255,0.12)", borderRadius: 3 }} />
          </div>
        ))}
      </div>

      {/* selected idea label */}
      <div style={{ position: "absolute", top: 30, left: 230, opacity: planT }}>
        <div style={{ fontFamily: FONT, fontWeight: 800, fontSize: 30, color: COLORS.offWhite, letterSpacing: "0.06em" }}>{idea.name}</div>
        <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
          {["chassis", "wheels", "cabin"].map((feat, i) => (
            <div key={feat} style={{ opacity: ramp(f, 40 + i * 12, 56 + i * 12), padding: "4px 10px", border: "1px solid rgba(255,213,0,0.3)", borderRadius: 3, fontFamily: FONT, fontSize: 12, letterSpacing: "0.1em", color: COLORS.gray }}>
              {feat.toUpperCase()}
            </div>
          ))}
        </div>
      </div>

      {/* concept scaffold during PLAN (a forming wireframe) */}
      {planGhost > 0.02 && (
        <div style={{ position: "absolute", left: "56%", top: "52%", transform: "translate(-50%,-50%)", opacity: planGhost }}>
          <svg width={360} height={280} style={{ overflow: "visible" }}>
            {[
              [60, 200, 300, 200],
              [90, 200, 140, 120],
              [140, 120, 240, 120],
              [240, 120, 300, 200],
              [180, 120, 180, 60],
            ].map(([x1, y1, x2, y2], i) => (
              <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={COLORS.yellow} strokeWidth={1.2} strokeDasharray="4 5" opacity={0.7} />
            ))}
            {[[60, 200], [300, 200], [140, 120], [240, 120], [180, 60]].map(([cx, cy], i) => (
              <circle key={i} cx={cx} cy={cy} r={4} fill={COLORS.yellow} opacity={0.9} />
            ))}
          </svg>
        </div>
      )}

      {/* the model assembling in isometric space */}
      <div style={{ position: "absolute", left: "56%", top: "52%", transform: "translate(-50%,-50%) scale(1.15)" }}>
        <BuildModel assemble={assemble} glow={lock} />
        {/* repair highlight sweeps one section */}
        {repair > 0 && lock < 1 && (
          <div
            style={{
              position: "absolute",
              left: 40,
              top: 10,
              width: 90,
              height: 90,
              border: `2px solid ${COLORS.yellow}`,
              borderRadius: 6,
              boxShadow: `0 0 24px rgba(255,213,0,0.5)`,
              opacity: (1 - lock) * Math.max(0, Math.sin(f / 6)),
            }}
          />
        )}
      </div>

      {pass === 1 && (
        <div style={{ position: "absolute", bottom: 26, left: 230, opacity: ramp(f, 10, 40) }}>
          <span style={{ fontFamily: FONT, fontSize: 16, color: COLORS.gray }}>“{COPY.generationDemo.customPrompt}”</span>
        </div>
      )}
    </div>
  );
};

export const GenerationDemo: React.FC<{ demo: DemoProps }> = ({ demo }) => {
  const frame = useCurrentFrame();
  const useReal = useRealClip(demo, "generation");
  const outFade = ramp(840 - frame, 0, 16);
  const active = stageIndex(frame);

  return (
    <AbsoluteFill style={{ opacity: 1 }}>
      <Backdrop />

      <div style={{ position: "absolute", top: 56, width: "100%", display: "flex", justifyContent: "center" }}>
        <TechnicalLabel text={COPY.generationDemo.pipeline} start={6} inDur={14} hold={90} outDur={16} size={24} active />
      </div>

      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", paddingTop: 40 }}>
        <ClipSlot
          useReal={useReal}
          src={ASSETS.generationClip}
          start={2}
          width={1420}
          height={700}
          fallback={<GenerationFallback />}
          overlay={
            <div style={{ position: "absolute", bottom: 18, left: "50%", transform: "translateX(-50%)" }}>
              <div style={{ padding: "10px 22px", background: "rgba(0,0,0,0.5)", borderRadius: 8, border: "1px solid rgba(255,255,255,0.1)" }}>
                <AgentProgress stages={[...COPY.generationDemo.stages]} activeIndex={active} />
              </div>
            </div>
          }
        />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

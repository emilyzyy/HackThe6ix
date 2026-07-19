import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { BigStatement } from "../components/BigStatement";
import { TechnicalLabel } from "../components/TechnicalLabel";
import { LegoBrick } from "../components/LegoBrick";
import { COLORS, FONT } from "../theme";
import { ramp, rand } from "../lib/anim";
import { generationIdeas, COPY } from "../demoConfig";

// A small idea token: an angled brick cluster + its name. Not a pricing card.
const IdeaToken: React.FC<{ name: string; accent: string; i: number; t: number }> = ({ name, accent, i, t }) => (
  <div
    style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 16,
      opacity: t,
      transform: `translateY(${(1 - t) * 60}px)`,
    }}
  >
    <div style={{ position: "relative", width: 130, height: 110 }}>
      {[0, 1, 2].map((k) => (
        <div
          key={k}
          style={{
            position: "absolute",
            left: 20 + k * 22 + rand(i * 3 + k) * 8,
            top: 20 + k * 16,
            transform: `rotate(${(rand(i + k) - 0.5) * 30}deg)`,
          }}
        >
          <LegoBrick studs={[2, 1]} unit={20} color={[accent, COLORS.brickWhite, COLORS.brickGray][k]} />
        </div>
      ))}
    </div>
    <div style={{ fontFamily: FONT, fontWeight: 800, fontSize: 26, letterSpacing: "0.08em", color: COLORS.offWhite }}>{name}</div>
  </div>
);

export const GenerationIntro: React.FC = () => {
  const frame = useCurrentFrame();
  const outFade = ramp(480 - frame, 0, 16);
  const invEnter = ramp(frame, 0, 30);
  const ideasShiftUp = ramp(frame, 300, 350);
  const promptReveal = ramp(frame, 330, 370);
  const cursorOn = Math.floor(frame / 15) % 2 === 0;

  return (
    <AbsoluteFill style={{ opacity: 1 }}>
      <Backdrop />

      {/* inventory remains the constraint, docked on the left */}
      <div
        style={{
          position: "absolute",
          left: 60,
          top: "50%",
          transform: `translateY(-50%) translateX(${(1 - invEnter) * -260}px)`,
          opacity: invEnter,
          width: 280,
          padding: 24,
          borderRadius: 10,
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,213,0,0.25)",
        }}
      >
        <TechnicalLabel text="YOUR INVENTORY" start={6} inDur={12} hold={9999} outDur={0} size={15} active />
        <div style={{ height: 14 }} />
        {["BRICK 2×4  ×3", "PLATE 1×6  ×2", "BRICK 2×2  ×4", "SLOPE 2×2  ×1"].map((r, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 0", opacity: ramp(frame, 20 + i * 6, 34 + i * 6) }}>
            <div style={{ width: 16, height: 16, borderRadius: 3, background: [COLORS.brickRed, COLORS.brickBlue, COLORS.brickYellow, COLORS.brickGreen][i] }} />
            <span style={{ fontFamily: FONT, fontSize: 17, color: COLORS.gray, letterSpacing: "0.06em" }}>{r}</span>
          </div>
        ))}
        <div style={{ marginTop: 8, fontFamily: FONT, fontSize: 12, letterSpacing: "0.2em", color: COLORS.grayDim }}>THE CONSTRAINT</div>
      </div>

      {/* ideas emerge from the inventory */}
      <div
        style={{
          position: "absolute",
          left: 420,
          right: 80,
          top: `calc(46% - ${ideasShiftUp * 150}px)`,
          transform: "translateY(-50%)",
          display: "flex",
          justifyContent: "space-around",
        }}
      >
        {generationIdeas.map((idea, i) => (
          <IdeaToken key={idea.name} name={idea.name} accent={idea.accent} i={i} t={ramp(frame, 70 + i * 12, 70 + i * 12 + 18)} />
        ))}
      </div>

      {/* prompt input beneath */}
      <div
        style={{
          position: "absolute",
          left: 460,
          right: 120,
          top: "62%",
          opacity: promptReveal,
          transform: `translateY(${(1 - promptReveal) * 40}px)`,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            height: 68,
            padding: "0 22px",
            borderRadius: 8,
            background: "rgba(255,255,255,0.04)",
            border: `1.5px solid ${COLORS.yellow}`,
          }}
        >
          <span style={{ fontFamily: FONT, fontSize: 24, color: COLORS.gray }}>{COPY.generationIntro.promptPlaceholder}</span>
          <span style={{ marginLeft: 4, width: 3, height: 30, background: COLORS.yellow, opacity: cursorOn ? 1 : 0 }} />
        </div>
      </div>

      {/* text */}
      <div style={{ position: "absolute", top: 80, left: 420, right: 80, display: "flex", justifyContent: "center" }}>
        <BigStatement lines={[COPY.generationIntro.startA, COPY.generationIntro.startB]} start={30} inDur={14} hold={230} outDur={16} size={44} accentLineIndex={1} />
      </div>
      <div style={{ position: "absolute", bottom: 90, width: "100%", display: "flex", justifyContent: "center" }}>
        <BigStatement lines={[COPY.generationIntro.orYours]} start={360} inDur={14} hold={80} outDur={16} size={46} accentLineIndex={0} />
      </div>
    </AbsoluteFill>
  );
};

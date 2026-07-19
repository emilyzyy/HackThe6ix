import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { BigStatement } from "../components/BigStatement";
import { TechnicalLabel } from "../components/TechnicalLabel";
import { SetBoxSilhouette } from "../components/SetBoxSilhouette";
import { BrickPile } from "../components/BrickPile";
import { LegoBin } from "../components/LegoBin";
import { COLORS, FONT } from "../theme";
import { appear, ramp } from "../lib/anim";
import { setCards, COPY } from "../demoConfig";

export const ProblemReveal: React.FC = () => {
  const frame = useCurrentFrame();
  // Phase 1: kicker + price cards (0-9s). Phase 2: statement + collapse to pile.
  const cardsOut = ramp(frame, 250, 285);
  const pileIn = ramp(frame, 270, 340);
  const outFade = interpolateOut(frame);

  return (
    <AbsoluteFill style={{ opacity: 1 }}>
      <Backdrop />

      {/* kicker */}
      <div style={{ position: "absolute", top: 120, width: "100%", display: "flex", justifyContent: "center" }}>
        <TechnicalLabel text={COPY.problem.kicker} start={8} inDur={14} hold={230} outDur={20} size={22} active />
      </div>

      {/* price cards */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          gap: 70,
          flexDirection: "row",
          opacity: 1 - cardsOut,
          transform: `translateY(${-cardsOut * 60}px)`,
        }}
      >
        {setCards.map((card, i) => {
          const t = appear(frame, 30 + i * 12, 18, 999, 0);
          const floaty = Math.sin((frame + i * 40) / 34) * 8;
          return (
            <div
              key={card.name}
              style={{
                opacity: t,
                transform: `translateY(${(1 - t) * 60 + floaty}px) rotate(${(i - 1) * 2}deg)`,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 22,
              }}
            >
              <SetBoxSilhouette name={card.name} accent={card.color} />
              <div
                style={{
                  fontFamily: FONT,
                  fontWeight: 800,
                  fontSize: 46,
                  color: COLORS.yellow,
                  letterSpacing: "0.02em",
                }}
              >
                {card.price}
              </div>
            </div>
          );
        })}
      </AbsoluteFill>

      {/* the reveal statement */}
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <BigStatement
          lines={[COPY.problem.line1, COPY.problem.line2]}
          start={95}
          inDur={16}
          hold={150}
          outDur={20}
          size={78}
          accentLineIndex={1}
        />
      </AbsoluteFill>

      {/* collapse into a pile inside the bin */}
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "flex-end" }}>
        <div style={{ position: "relative", marginBottom: 40, opacity: pileIn }}>
          <div style={{ position: "absolute", left: "50%", bottom: -30, transform: "translateX(-50%)" }}>
            <LegoBin width={640} height={360} />
          </div>
          <div style={{ position: "relative", transform: "translateY(30px)" }}>
            <BrickPile count={22} settle={ramp(frame, 300, 380)} seed={9} />
          </div>
          <div
            style={{
              position: "absolute",
              left: "50%",
              bottom: -70,
              transform: "translateX(-50%)",
            }}
          >
            <TechnicalLabel text={COPY.problem.line3} start={330} inDur={16} hold={200} outDur={20} size={24} active marker={false} />
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

// fade the whole scene out in its last 18 frames for continuity
const interpolateOut = (frame: number) =>
  ramp(600 - frame, 0, 18);

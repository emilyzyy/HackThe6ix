import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { BigStatement } from "../components/BigStatement";
import { FallingBricks } from "../components/FallingBricks";
import { ramp } from "../lib/anim";
import { COPY } from "../demoConfig";

// A deliberately quiet beat after the dense CV sequence. A few pieces fall;
// the question changes.
export const QuestionTransition: React.FC = () => {
  const frame = useCurrentFrame();
  const outFade = ramp(300 - frame, 0, 16);
  const impact = ramp(frame, 250, 300);

  return (
    <AbsoluteFill style={{ opacity: 1, background: "#020202" }}>
      <Backdrop studGrid={false} vignette={1} />
      <FallingBricks count={6} speed={0.6} seed={21} opacity={0.7} unit={26} />

      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <BigStatement lines={[COPY.question.know]} start={20} inDur={16} hold={70} outDur={18} size={58} />
      </AbsoluteFill>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <BigStatement lines={[COPY.question.become]} start={140} inDur={16} hold={80} outDur={16} size={78} accentLineIndex={0} />
      </AbsoluteFill>

      {/* impact flash into the generation world */}
      <AbsoluteFill style={{ background: "#FFD500", opacity: impact * 0.0 }} />
    </AbsoluteFill>
  );
};

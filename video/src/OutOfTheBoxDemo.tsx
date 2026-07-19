import React from "react";
import {
  AbsoluteFill,
  Audio,
  Sequence,
  staticFile,
  useCurrentFrame,
} from "remotion";
import { ASSETS, DemoProps, SCENES } from "./demoConfig";
import { COLORS } from "./theme";
import { ramp } from "./lib/anim";

import { ProblemReveal } from "./scenes/ProblemReveal";
import { ScanDemo } from "./scenes/ScanDemo";
import { UnderstandScan } from "./scenes/UnderstandScan";
import { FindPieces } from "./scenes/FindPieces";
import { FusePieces } from "./scenes/FusePieces";
import { IdentifyPieces } from "./scenes/IdentifyPieces";
import { InventoryReveal } from "./scenes/InventoryReveal";
import { QuestionTransition } from "./scenes/QuestionTransition";
import { GenerationIntro } from "./scenes/GenerationIntro";
import { GenerationDemo } from "./scenes/GenerationDemo";
import { FinalReveal } from "./scenes/FinalReveal";
import { BrickWipe } from "./components/BrickWipe";

// Frames each scene starts BEFORE the previous one ends, so the previous stays
// fully visible underneath while the incoming scene fades in — a true
// crossfade, never a dip to black. Internal scene timing shifts by this small
// amount, which is harmless (content simply begins during the dissolve).
const OVERLAP = 16;

const FadeIn: React.FC<{ dur: number; children: React.ReactNode }> = ({
  dur,
  children,
}) => {
  const frame = useCurrentFrame();
  const o = dur <= 0 ? 1 : ramp(frame, 0, dur);
  return <AbsoluteFill style={{ opacity: o }}>{children}</AbsoluteFill>;
};

const S: React.FC<{
  name: keyof typeof SCENES;
  first?: boolean;
  children: React.ReactNode;
}> = ({ name, first = false, children }) => {
  const nominal = SCENES[name].from;
  const start = first ? nominal : Math.max(0, nominal - OVERLAP);
  const duration = SCENES[name].to - start;
  return (
    <Sequence from={start} durationInFrames={duration}>
      <FadeIn dur={first ? 0 : OVERLAP}>{children}</FadeIn>
    </Sequence>
  );
};

export const OutOfTheBoxDemo: React.FC<DemoProps> = (props) => {
  return (
    <AbsoluteFill style={{ background: COLORS.black }}>
      {props.hasMusic && (
        <Audio src={staticFile(ASSETS.music)} volume={0.55} />
      )}

      <S name="problemReveal" first>
        <ProblemReveal />
      </S>
      <S name="scanDemo">
        <ScanDemo demo={props} />
      </S>
      <S name="understandScan">
        <UnderstandScan />
      </S>
      <S name="findPieces">
        <FindPieces />
      </S>
      <S name="fusePieces">
        <FusePieces />
      </S>
      <S name="identifyPieces">
        <IdentifyPieces />
      </S>
      <S name="inventoryReveal">
        <InventoryReveal />
      </S>
      <S name="questionTransition">
        <QuestionTransition />
      </S>
      <S name="generationIntro">
        <GenerationIntro />
      </S>
      <S name="generationDemo">
        <GenerationDemo demo={props} />
      </S>
      <S name="finalReveal">
        <FinalReveal />
      </S>

      {/* Signature brick-wipe across the single biggest world change:
          the CV world handing off to the generation world. */}
      <Sequence
        from={SCENES.questionTransition.to - 12}
        durationInFrames={40}
      >
        <BrickWipe start={0} dur={30} direction="right" />
      </Sequence>
    </AbsoluteFill>
  );
};

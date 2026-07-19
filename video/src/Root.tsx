import React from "react";
import { Composition } from "remotion";
import { OutOfTheBoxDemo } from "./OutOfTheBoxDemo";
import { DEFAULT_PROPS, TOTAL_FRAMES, DemoProps } from "./demoConfig";
import { FPS } from "./theme";

// Remotion's Composition constrains props to Record<string, unknown>; our
// DemoProps is a concrete shape, so bridge the two at the boundary only.
const DemoComp = OutOfTheBoxDemo as unknown as React.FC<Record<string, unknown>>;

export const Root: React.FC = () => {
  return (
    <Composition
      id="OutOfTheBoxDemo"
      component={DemoComp}
      durationInFrames={TOTAL_FRAMES}
      fps={FPS}
      width={1920}
      height={1080}
      defaultProps={DEFAULT_PROPS as unknown as Record<string, unknown>}
    />
  );
};

// re-export for type consumers
export type { DemoProps };

import { DRUM_PITCH_CLASSES } from "./constants";

function signedMod(value: number, base: number): number {
  const mod = value % base;
  if (mod <= base / 2) {
    return mod;
  } else {
    return mod - base;
  }
}

function scale(
  value: number,
  minIn: number,
  maxIn: number,
  minOut: number,
  maxOut: number
): number {
  value = Math.min(Math.max(value, minIn), maxIn);
  value = ((value - minIn) / (maxIn - minIn)) * (maxOut - minOut) + minOut;
  return value;
}

function zeroArray(size: number): number[] {
  return Array.from({ length: size }, () => 0.0);
}

function zeroMatrix(shape: number[]): number[][] {
  return Array.from({ length: shape[0] }, () => {
    const array = Array.from({ length: shape[1] }, () => 0);
    return array;
  });
}

function pitchToIndexMap(
  pitchMap: Record<string, number[]> = DRUM_PITCH_CLASSES.pitch,
  indexMap: Record<string, number> = DRUM_PITCH_CLASSES.index
): Record<string, number> {
  const pitchIndexMap = {};
  for (const [instrument, pitches] of Object.entries(pitchMap)) {
    for (const p of pitches) {
      pitchIndexMap[p.toString()] = indexMap[instrument];
    }
  }
  return pitchIndexMap;
}

function linspace(min: number, max: number, length: number): number[] {
  const delta = (max - min) / length;
  const output: number[] = [];
  let currentIndex = max;
  for (let i = 0; i < length; i++) {
    output.push(round(currentIndex, 3));
    currentIndex -= delta;
  }
  output.push(min);
  return output.reverse();
}

function round(value: number, depth: number): number {
  const scale = 10 ** depth;
  return Math.round(value * scale) / scale;
}

export {
  signedMod,
  scale,
  zeroArray,
  zeroMatrix,
  pitchToIndexMap,
  linspace,
  round,
};

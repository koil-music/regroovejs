import path from "path";
import DRUM_PITCH_CLASSES from "./pitch-classes";

const DRUM_PITCH_MAP = Object.keys(DRUM_PITCH_CLASSES.pitch);
const CHANNELS = DRUM_PITCH_MAP.length;
const PITCHES = Object.values(DRUM_PITCH_MAP);
const LOOP_DURATION = 16;
const MIN_VELOCITY_THRESHOLD = 0.01;
const NOTE_THRESHOLD = 0.5;
const MIN_ONSET_THRESHOLD = 0.3;
const MAX_ONSET_THRESHOLD = 0.7;
const NUM_SAMPLES = 100;
const NOTE_DROPOUT = 0.5;

let ENV = "staging";
if (typeof process.env.REGROOVE_ENV === "string") {
  ENV = process.env.REGROOVE_ENV;
}
const LOCAL_MODEL_DIR = path.join(process.cwd(), `/regroove-models/${ENV}/`);
console.log(`Loading models from ${LOCAL_MODEL_DIR}`);

export {
  DRUM_PITCH_CLASSES,
  DRUM_PITCH_MAP,
  CHANNELS,
  PITCHES,
  LOOP_DURATION,
  MIN_VELOCITY_THRESHOLD,
  NOTE_THRESHOLD,
  MIN_ONSET_THRESHOLD,
  MAX_ONSET_THRESHOLD,
  NUM_SAMPLES,
  NOTE_DROPOUT,
  LOCAL_MODEL_DIR,
};

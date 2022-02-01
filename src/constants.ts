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
const SYNC_LATENT_SIZE = 2;
const GROOVE_LATENT_SIZE = 32;

export {
  CHANNELS,
  DRUM_PITCH_CLASSES,
  DRUM_PITCH_MAP,
  GROOVE_LATENT_SIZE,
  LOOP_DURATION,
  MIN_VELOCITY_THRESHOLD,
  MIN_ONSET_THRESHOLD,
  MAX_ONSET_THRESHOLD,
  NOTE_DROPOUT,
  NOTE_THRESHOLD,
  NUM_SAMPLES,
  PITCHES,
  SYNC_LATENT_SIZE,
};

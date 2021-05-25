/* eslint-disable @typescript-eslint/no-var-requires */
import fs from "fs";

import { Pattern } from "./pattern";
import { LOOP_DURATION, CHANNELS, DRUM_PITCH_CLASSES } from "./constants";
const JZZ = require("jzz");
require("jzz-midi-smf")(JZZ);

// TODO: Refactor using EventEmitter
function processMidiEvent(
  event,
  pitchMapping: Record<string, number>,
  stepsPer16th: number
): [number, number, number, number] {
  const step = event.tt / stepsPer16th;
  const stepIndex = Math.round(step);
  const pitch = parseInt(event["1"]);
  const channelIndex = pitchMapping[pitch];
  const velocity = event["2"];
  const offset = step - stepIndex;
  return [stepIndex, channelIndex, velocity, offset];
}

async function readMidiFile(
  filePath: string,
  pitchMapping: Record<string, number>
): Promise<[Pattern, Pattern, Pattern]> {
  // get MIDI data
  const binary = fs.readFileSync(filePath, "binary");
  const midiSMF = new JZZ.MIDI.SMF(binary);
  const stepsPerQuarter = midiSMF.ppqn;
  const stepsPer16th = stepsPerQuarter / 4;

  const onsets = Pattern.zeros([1, LOOP_DURATION, CHANNELS]);
  const velocities = Pattern.zeros([1, LOOP_DURATION, CHANNELS]);
  const offsets = Pattern.zeros([1, LOOP_DURATION, CHANNELS]);

  midiSMF.forEach((seq) => {
    for (let j = 0; j < seq.length; j++) {
      const event = seq[j];
      if (event["0"] === 144 && event["2"] > 0) {
        const [stepIndex, channelIndex, velocity, offset] = processMidiEvent(
          event,
          pitchMapping,
          stepsPer16th
        );
        if (stepIndex < LOOP_DURATION) {
          onsets[0][stepIndex][channelIndex] = 1.0;
          velocities[0][stepIndex][channelIndex] = velocity / 127;
          offsets[0][stepIndex][channelIndex] = offset;
        }
      }
    }
  });

  const onsetsPattern = new Pattern(onsets, [1, LOOP_DURATION, CHANNELS]);
  const velocitiesPattern = new Pattern(velocities, [
    1,
    LOOP_DURATION,
    CHANNELS,
  ]);
  const offsetsPattern = new Pattern(offsets, [1, LOOP_DURATION, CHANNELS]);
  return [onsetsPattern, velocitiesPattern, offsetsPattern];
}

async function writeMidiFile(
  onsetsPattern: Pattern,
  velocitiesPattern: Pattern,
  offsetsPattern: Pattern,
  filePath: string
): Promise<void> {
  if (onsetsPattern.shape[0] !== 1) {
    throw new Error("Only patterns of batch size 1 can be written to MIDI");
  }
  const ticksPerBeat = 96;
  const smf = new JZZ.MIDI.SMF(0, ticksPerBeat);
  smf.push(new JZZ.MIDI.SMF.MTrk());

  smf[0].smfBPM(120);
  const onsets = onsetsPattern.tensor()[0];
  const velocities = velocitiesPattern.tensor()[0];
  const offsets = offsetsPattern.tensor()[0];

  for (let step = 0; step < LOOP_DURATION; step++) {
    const tick = (ticksPerBeat * step) / 4;
    // smf[0].tick(tick)
    for (let instrument = 0; instrument < CHANNELS; instrument++) {
      if (onsets[step][instrument] === 1) {
        const offsetTick =
          tick + (ticksPerBeat / 4) * offsets[step][instrument];
        const pitch = DRUM_PITCH_CLASSES.drum_index[instrument.toString()];
        const velocity = Math.round(velocities[step][instrument] * 127);
        smf[0]
          .tick(Math.round(offsetTick))
          .note(0, pitch, velocity, ticksPerBeat / 24);
      }
    }
  }
  smf[0].tick(LOOP_DURATION * ticksPerBeat).smfEndOfTrack();
  fs.writeFileSync(filePath, smf.dump(), "binary");
}

export { readMidiFile, writeMidiFile };

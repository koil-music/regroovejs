import assert from "assert"
import { DRUM_PITCH_CLASSES, LOOP_DURATION, CHANNELS } from "../src/constants";
import { Pattern } from "../src/pattern";
import { readMidiFile, writeMidiFile } from "../src/midi";
import { arraysEqual } from "./helpers";
import { pitchToIndexMap } from "../src/util";

describe("read/writeMidiFile", function() {
  const dims = [1, LOOP_DURATION, CHANNELS]
  const onsetsPattern = new Pattern(new Float32Array(dims[1] * dims[2]), dims)
  const velocitiesPattern = new Pattern(new Float32Array(dims[1] * dims[2]), dims)
  const offsetsPattern = new Pattern(new Float32Array(dims[1] * dims[2]), dims)

  const fillCells: number[][] = []
  for (let i = 0; i < LOOP_DURATION; i++) {
    let v = [i, i % 9]
    fillCells.push(v)
  }
  // const fillCells = [[1, 2]]

  fillCells.forEach((v) => {
    onsetsPattern.setcell(1., v[0], v[1])
    velocitiesPattern.setcell(v[1] * 0.1, v[0], v[1])
    offsetsPattern.setcell(0.5 - v[1] * 0.1, v[0], v[1])
  })
  const pitchMapping = pitchToIndexMap(
    DRUM_PITCH_CLASSES["pitch"],
    DRUM_PITCH_CLASSES["index"]
  )

  it("writes pattern", async function() {
    await writeMidiFile(onsetsPattern, velocitiesPattern, offsetsPattern, "test.mid")
  })
  it("reads pattern and arraysEqual", async function() {
    const [gotOnsetsPattern, gotVelocitiesPattern, gotOffsetsPattern] = await readMidiFile("test.mid", pitchMapping)
    const expectedOnsets = Array.from(onsetsPattern.data);
    const gotOnsets = Array.from(gotOnsetsPattern.data);
    assert.ok(arraysEqual(expectedOnsets, gotOnsets));

    let gotVelocities = Array.from(gotVelocitiesPattern.data);
    gotVelocities = gotVelocities.map(v => Math.round(v * 127))
    let expectedVelocities = Array.from(velocitiesPattern.data);
    expectedVelocities = expectedVelocities.map(v => Math.round(v * 127))
    assert.ok(arraysEqual(expectedVelocities, gotVelocities));

    // let gotOffsets = Array.from(gotOffsetsPattern.data);
    // gotOffsets = gotOffsets.map(v => Math.round(v * 10))
    // let expectedOffsets = Array.from(offsetsPattern.data);
    // expectedOffsets = expectedOffsets.map(v => Math.round(v * 10))
    // assert.ok(arraysEqual(expectedOffsets, gotOffsets));
  })
})

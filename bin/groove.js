/* eslint-disable @typescript-eslint/no-var-requires */
const path = require("path");
const parseArgs = require("minimist");

const { pitchToIndexMap } = require("../dist/util");
const { DRUM_PITCH_CLASSES, LOCAL_MODEL_DIR } = require("../dist/constants");
const { Pattern } = require("../dist/pattern");
const { readMidiFile, writeMidiFile } = require("../dist/midi");
const ONNXModel = require("../dist/model").default;
const { applyOnsetThreshold } = require("../dist/generate");

const argv = parseArgs(process.argv.slice(2));
const loadDir = "/mnt/c/Users/maxkr/Music/samples/midi/stochastic-groove/"
const filePath = argv.filepath
const threshold = argv.threshold || 0.5
const dropout = argv.dropout || 1.
console.log(`Loading MIDI from ${filePath}`);
console.log(`Using onset threshold value: ${threshold}`)
console.log(`Using note dropout value: ${dropout}`)

const pitchMapping = pitchToIndexMap(
  DRUM_PITCH_CLASSES["pitch"],
  DRUM_PITCH_CLASSES["index"]
)

let inputOnsets;
let inputVelocities;
let inputOffsets;

let outputOnsets;
let outputVelocities;
let outputOffsets;

function predict(model) {
  // zero velocities and offsets as input for groove model
  const velocities = new Pattern(new Float32Array(144), inputOnsets.shape);
  const offsets = new Pattern(new Float32Array(144), inputOnsets.shape);

  let pattern = inputOnsets.concatenate(velocities, 2)
  pattern = pattern.concatenate(offsets, 2)

  model.forward(pattern, inputOnsets.shape, dropout).then(async(output) => {
    [outputOnsets, outputVelocities, outputOffsets] = preprocess(
      output.onsets, output.velocities, output.offsets
    )
    const savePath = filePath.slice(0, -4) + ".groove" + `-${Math.floor(Date.now() / 1000)}`  + ".mid"
    await writeMidiFile(outputOnsets, outputVelocities, outputOffsets, savePath)
    console.log(`Wrote grooved rhythm to ${savePath}.`);
  })
}

function preprocess(onsets, velocities, offsets) {
  outputOnsets = applyOnsetThreshold(onsets, inputOnsets.shape, threshold);
  
  const maskedVelocities = Array.from(velocities.data).map((v, i) => {
    return v * outputOnsets.data[i];
  })
  const sigmVelocities = maskedVelocities.map(v => {
    return (1 / (1 + Math.exp(v * -5)) - 0.5) * 2
  })
  outputVelocities = new Pattern(sigmVelocities, outputOnsets.shape)
  outputOffsets = new Pattern(new Float32Array(outputVelocities.size), inputOnsets.shape);
  return [outputOnsets, outputVelocities, outputOffsets]
}

readMidiFile(filePath, pitchMapping)
  .then((output) => {
    inputOnsets = output[0];
    inputVelocities = output[1];
    inputOffsets = output[2];

    console.log("Loaded file.");
    ONNXModel.load("groove", LOCAL_MODEL_DIR).then((model) => predict(model));
  })
  .catch((e) => {
    console.log("Something went wrong: ", e)
  })



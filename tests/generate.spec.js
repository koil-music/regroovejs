import assert from "assert"
import { performance } from "perf_hooks";
import { describe, it } from 'mocha'

import { CHANNELS, LOCAL_MODEL_DIR, LOOP_DURATION, MAX_ONSET_THRESHOLD, MIN_ONSET_THRESHOLD, NOTE_DROPOUT, NUM_SAMPLES } from "../src/constants";
import { Pattern, PatternSizeError } from "../src/pattern"
import { PatternDataMatrix, applyOnsetThreshold } from "../src/generate"
import { Generator } from "../src/generate";
import { arraysEqual } from "./helpers.ts"
import { linspace } from "../src/util";
import { Tensor } from "onnxruntime";

describe("PatternDataMatrix", function() {

  const expectedShape = [1, LOOP_DURATION, CHANNELS]
  let length = 10
  const dataMatrix = new PatternDataMatrix(expectedShape, length)
  const pattern = Float32Array.from({ length: expectedShape[1] * expectedShape[2] }, _ => {
    return 1
  })

  it("constructed properly", function() {
    
    assert.strictEqual(dataMatrix.length, length)
    assert.strictEqual(dataMatrix.matrixSize, length ** 2)
    assert.strictEqual(dataMatrix.outputShape, expectedShape)
    assert.strictEqual(dataMatrix.outputSize, expectedShape[1] * expectedShape[2])

    length = 20
    dataMatrix.length = length
    assert.strictEqual(dataMatrix.length, length)

    const emptyDataMatrix = dataMatrix.empty()
    assert.strictEqual(emptyDataMatrix.length, length)
    assert.strictEqual(emptyDataMatrix[0].length, length)
    assert.strictEqual(emptyDataMatrix[0][0].length, LOOP_DURATION * CHANNELS)
    assert.ok(arraysEqual(Array.from(emptyDataMatrix[0][0]), Array.from({ length: LOOP_DURATION * CHANNELS }, _ => 0.)))
  }),
  it("appends correct patterns", function() {
    for (let i = 0; i < length; i++) {
      for (let j = 0; j < length; j++) {
        dataMatrix.append(pattern, i, j)
        const gotPattern = dataMatrix.sample(i, j)
        assert.ok(arraysEqual(Array.from(gotPattern), Array.from(pattern)))
      }
    } 
  }),
  it("rejects out of bounds append indices", function() {
    let i = length + 1
    let j = 0
    dataMatrix.append(pattern, i, j)
  }),
  it("throws patternSizeError", function() {
    let i = 1
    let j = 1
    const invalidLength = expectedShape[1] * expectedShape[2] + 1
    const invalidPattern = Float32Array.from({ length: invalidLength }, _ => {
      return 1
    })
    assert.throws(function() {dataMatrix.append(invalidPattern, i, j)}, PatternSizeError)
  }),
  it("fails when indexing out of bounds", function() {
    let i = length + 1
    let j = 1
    assert.throws(function() {dataMatrix.sample(i, j)}, TypeError)
  })
})

describe("Generator", function () {
  const onsetsData = Float32Array.from({ length: LOOP_DURATION * CHANNELS }, _ => 1.)
  const velocitiesData = Float32Array.from(onsetsData).fill(0.5)
  const offsetsData = Float32Array.from(onsetsData).fill(0.)
  const expectedDims = [1, LOOP_DURATION, CHANNELS]

  it("constructs, sets and gets attributes", async function() {

    const generator = await Generator.build(
        onsetsData,
        velocitiesData,
        offsetsData,
        LOCAL_MODEL_DIR
    )

    // assert.strictEqual(typeof generator.model, ONNXModel)
    const got = generator.minOnsetThreshold
    assert.strictEqual(generator.minOnsetThreshold, MIN_ONSET_THRESHOLD)
    const minThreshold = 0.1
    generator.minOnsetThreshold = minThreshold
    assert.strictEqual(generator.minOnsetThreshold, minThreshold)

    assert.strictEqual(generator.maxOnsetThreshold, MAX_ONSET_THRESHOLD)
    const maxThreshold = 0.9
    generator.maxOnsetThreshold = maxThreshold
    assert.strictEqual(generator.maxOnsetThreshold, maxThreshold)

    const expectedRange = linspace(minThreshold, maxThreshold, generator.axisLength)
    assert.ok(arraysEqual(generator.onsetThresholdRange, expectedRange))
    assert.ok(arraysEqual(generator.dims, [generator.axisLength, LOOP_DURATION, CHANNELS]))

    assert.strictEqual(generator.numSamples, NUM_SAMPLES)
    const numSamples = [60, 64, 68]
    for (let i = 0; i < numSamples.length; i++) {
        generator.numSamples = numSamples[i]
        assert.strictEqual(generator.numSamples, numSamples[i])
        assert.strictEqual(generator.axisLength, 8)    
    }

    assert.ok(arraysEqual(generator.outputShape, expectedDims))
    assert.strictEqual(generator.noteDropout, NOTE_DROPOUT)
    assert.strictEqual(generator.minNoteDropout, NOTE_DROPOUT - 0.05)
    assert.strictEqual(generator.maxNoteDropout, NOTE_DROPOUT + 0.05)
    const noteDropout = 0.9
    generator.noteDropout = noteDropout
    assert.strictEqual(generator.noteDropout, noteDropout)

    assert.strictEqual(generator.channels, CHANNELS)
    const channels = 10
    generator.channels = 10
    assert.strictEqual(generator.channels, channels)
    assert.ok(arraysEqual(generator.outputShape, [1, LOOP_DURATION, channels]))

    assert.strictEqual(generator.loopDuration, LOOP_DURATION)
    const loopDuration = 33
    generator.loopDuration = loopDuration
    assert.strictEqual(generator.loopDuration, loopDuration)
    assert.ok(arraysEqual(generator.outputShape, [1, loopDuration, channels]))
  }),
  it("returns empty matrix before running", async function() {
    const generator = await Generator.build(
        onsetsData,
        velocitiesData,
        offsetsData,
        LOCAL_MODEL_DIR
    )
    assert.ok(arraysEqual(generator.onsets.outputShape, [1, LOOP_DURATION, CHANNELS]))
    assert.ok(arraysEqual(generator.velocities.outputShape, [1, LOOP_DURATION, CHANNELS]))
    assert.ok(arraysEqual(generator.offsets.outputShape, [1, LOOP_DURATION, CHANNELS]))

  }),
  it("correct size of batchInput", async function() {
    const generator = await Generator.build(
      onsetsData,
      velocitiesData,
      offsetsData,
      LOCAL_MODEL_DIR
    )
    const dims = [1, LOOP_DURATION, CHANNELS]
    const onsetsPattern = new Pattern(onsetsData, dims)
    const expectedBatchedInputSize = Math.sqrt(NUM_SAMPLES)*LOOP_DURATION*CHANNELS*3
    assert.strictEqual(expectedBatchedInputSize, generator.batchedInput(onsetsPattern, Math.sqrt(NUM_SAMPLES)).data.length)
  })
  it("builds and initializes methods and variables", async function () {
    const generator = await Generator.build(
      onsetsData,
      velocitiesData,
      offsetsData,
      LOCAL_MODEL_DIR
    )
    const st = performance.now();
    await generator.run();
    // console.log("time to populate:", performance.now() - st);

    assert.strictEqual(generator.onsets.matrixSize, generator.numSamples)
    assert.strictEqual(generator.velocities.matrixSize, generator.numSamples)
    assert.strictEqual(generator.offsets.matrixSize, generator.numSamples)

    const onsets = generator.onsets
    for (let i = 0; i < onsets.length; i++) {
      for (let j = 0; j < onsets.length; j++) {
        assert.strictEqual(onsets.sample(i, j).length, onsetsData.length)
      }
    }
  });
});

describe("applyOnsetThreshold", function() {
  it("applies onset properly", async function() {
    const data = new Float32Array(8)
    const expectedDims = [1, 4, 2]
    data.fill(0.5)
    let gotPattern = applyOnsetThreshold(new Tensor("float32", data, expectedDims), expectedDims, 0.4)
    let expected = Array.from({ length: 8 }, _ => 1.)
    assert.ok(arraysEqual(Array.from(gotPattern.data), expected))

    data.fill(0.3)
    gotPattern = applyOnsetThreshold(new Tensor("float32", data, expectedDims), expectedDims, 0.4)
    expected = Array.from({ length: 8 }, _ => 0.)
    assert.ok(arraysEqual(Array.from(gotPattern.data), expected))
  }) })

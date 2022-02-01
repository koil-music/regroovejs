import assert from "assert";
import fs from "fs";
import os from "os";
import path from "path";
import { describe, it } from "mocha";

import {
  CHANNELS,
  LOCAL_MODEL_DIR,
  LOOP_DURATION,
  MAX_ONSET_THRESHOLD,
  MIN_ONSET_THRESHOLD,
  NOTE_DROPOUT,
  NUM_SAMPLES,
} from "../constants";
import { Pattern } from "../pattern";
import Generator from "../generate";
import PatternDataMatrix from "../data-matrix";
import { arraysEqual } from "./helpers.ts";
import { linspace } from "../util";

describe("Generator", function () {
  const onsetsData = Float32Array.from(
    { length: LOOP_DURATION * CHANNELS },
    () => 1
  );
  const velocitiesData = Float32Array.from(onsetsData).fill(0.5);
  const offsetsData = Float32Array.from(onsetsData).fill(0);
  const expectedDims = [1, LOOP_DURATION, CHANNELS];

  it("constructs, sets and gets attributes", async function () {
    const generator = await Generator.build(
      onsetsData,
      velocitiesData,
      offsetsData,
      "./regroove-models/v2/graceful-fire-240/model.onnx",
      "./regroove-models/v2/olive-lion-52/model.onnx",
    );

    // assert.strictEqual(typeof generator.model, ONNXModel)
    const got = generator.minOnsetThreshold;
    assert.strictEqual(got, MIN_ONSET_THRESHOLD);
    const minThreshold = 0.1;
    generator.minOnsetThreshold = minThreshold;
    assert.strictEqual(generator.minOnsetThreshold, minThreshold);

    assert.strictEqual(generator.maxOnsetThreshold, MAX_ONSET_THRESHOLD);
    const maxThreshold = 0.9;
    generator.maxOnsetThreshold = maxThreshold;
    assert.strictEqual(generator.maxOnsetThreshold, maxThreshold);

    const expectedRange = linspace(
      minThreshold,
      maxThreshold,
      generator.axisLength
    );
    assert.ok(arraysEqual(generator.onsetThresholdRange, expectedRange));
    assert.ok(
      arraysEqual(generator.dims, [
        generator.axisLength,
        LOOP_DURATION,
        CHANNELS,
      ])
    );

    assert.strictEqual(generator.numSamples, NUM_SAMPLES);
    const numSamples = [60, 64, 68];
    for (let i = 0; i < numSamples.length; i++) {
      generator.numSamples = numSamples[i];
      assert.strictEqual(generator.numSamples, numSamples[i]);
      assert.strictEqual(generator.axisLength, 8);
    }

    assert.ok(arraysEqual(generator.outputShape, expectedDims));
    assert.strictEqual(generator.noteDropout, NOTE_DROPOUT);
    assert.strictEqual(generator.minNoteDropout, NOTE_DROPOUT - 0.05);
    assert.strictEqual(generator.maxNoteDropout, NOTE_DROPOUT + 0.05);
    const noteDropout = 0.9;
    generator.noteDropout = noteDropout;
    assert.strictEqual(generator.noteDropout, noteDropout);

    assert.strictEqual(generator.channels, CHANNELS);
    const channels = 10;
    generator.channels = 10;
    assert.strictEqual(generator.channels, channels);
    assert.ok(arraysEqual(generator.outputShape, [1, LOOP_DURATION, channels]));

    assert.strictEqual(generator.loopDuration, LOOP_DURATION);
    const loopDuration = 33;
    generator.loopDuration = loopDuration;
    assert.strictEqual(generator.loopDuration, loopDuration);
    assert.ok(arraysEqual(generator.outputShape, [1, loopDuration, channels]));
  }),
    it("returns empty matrix before running", async function () {
      const generator = await Generator.build(
        onsetsData,
        velocitiesData,
        offsetsData,
        "./regroove-models/v2/graceful-fire-240/model.onnx",
        "./regroove-models/v2/olive-lion-52/model.onnx",
      );
      assert.ok(
        arraysEqual(generator.onsets.outputShape, [1, LOOP_DURATION, CHANNELS])
      );
      assert.ok(
        arraysEqual(generator.velocities.outputShape, [
          1,
          LOOP_DURATION,
          CHANNELS,
        ])
      );
      assert.ok(
        arraysEqual(generator.offsets.outputShape, [1, LOOP_DURATION, CHANNELS])
      );
    }),
    it("correct size of batchInput", async function () {
      const generator = await Generator.build(
        onsetsData,
        velocitiesData,
        offsetsData,
        "./regroove-models/v2/graceful-fire-240/model.onnx",
        "./regroove-models/v2/olive-lion-52/model.onnx",
      );
      const dims = [1, LOOP_DURATION, CHANNELS];
      const onsetsPattern = new Pattern(onsetsData, dims);
      const expectedBatchedInputSize =
        Math.sqrt(NUM_SAMPLES) * LOOP_DURATION * CHANNELS * 3;
      assert.strictEqual(
        expectedBatchedInputSize,
        generator.batchedInput(onsetsPattern, Math.sqrt(NUM_SAMPLES)).data
          .length
      );
    }),
    it("builds and initializes methods and variables", async function () {
      const generator = await Generator.build(
        onsetsData,
        velocitiesData,
        offsetsData,
        "./regroove-models/v2/graceful-fire-240/model.onnx",
        "./regroove-models/v2/olive-lion-52/model.onnx",
      );
      await generator.run();

      assert.strictEqual(generator.onsets.matrixSize, generator.numSamples);
      assert.strictEqual(generator.velocities.matrixSize, generator.numSamples);
      assert.strictEqual(generator.offsets.matrixSize, generator.numSamples);

      const onsets = generator.onsets;
      for (let i = 0; i < onsets.length; i++) {
        for (let j = 0; j < onsets.length; j++) {
          assert.strictEqual(onsets.sample(i, j).length, onsetsData.length);
        }
      }
    }),
    it("saves and loads data", async function () {
      const generator = await Generator.build(
        onsetsData,
        velocitiesData,
        offsetsData,
        "./regroove-models/v2/graceful-fire-240/model.onnx",
        "./regroove-models/v2/olive-lion-52/model.onnx",
      );
      await generator.run();

      // save and load a different PatternDataMatrix
      const expectedShape = [1, LOOP_DURATION, CHANNELS];
      let length = 10;
      const dataMatrix = new PatternDataMatrix(expectedShape, length);
      // const testDataMatrix = dataMatrix.empty();
      const ones = Float32Array.from(
        { length: LOOP_DURATION * CHANNELS },
        () => 1
      );
      dataMatrix._T[0][0] = ones;

      // assign different PatternDataMatrix to generator
      generator.onsets = dataMatrix;
      generator.velocities = dataMatrix;
      generator.offsets = dataMatrix;

      const jsonData = await generator.encode();
      const gotGenerator = await Generator.build(
        onsetsData,
        velocitiesData,
        offsetsData,
        "./regroove-models/v2/graceful-fire-240/model.onnx",
        "./regroove-models/v2/olive-lion-52/model.onnx",
      );
      await gotGenerator.load(jsonData);

      assert.ok(
        arraysEqual(
          Array.from(dataMatrix.data[0][0]),
          Array.from(gotGenerator.onsets.data[0][0])
        )
      );
      assert.ok(
        arraysEqual(
          Array.from(dataMatrix.data[1][1]),
          Array.from(gotGenerator.velocities.data[1][1])
        )
      );
      assert.ok(
        arraysEqual(
          Array.from(dataMatrix.data[3][3]),
          Array.from(gotGenerator.offsets.data[3][3])
        )
      );
    });
  it("normalizes velocity", async function () {
    const generator = await Generator.build(
      onsetsData,
      velocitiesData,
      offsetsData,
      "./regroove-models/v2/graceful-fire-240/model.onnx",
      "./regroove-models/v2/olive-lion-52/model.onnx",
    );
    await generator.run();
    await generator.normalizeVelocities();
  });
});
import assert from "assert";
import fs from "fs";
import { describe, it } from "mocha";

import { ONNXModel } from "../model";
import { testPattern } from "./helpers.ts";

describe("ONNXModel", function () {
  it("syncopate model is constructed correctly", async function () {
    const model = await ONNXModel.load(
      "./regroove-models/v2/graceful-fire-240/model.onnx",
      2
    );
    const configData = fs.readFileSync(
      "./regroove-models/v2/syncopate.json",
      "utf-8"
    );
    const config = JSON.parse(configData);
    assert.ok(model.latentSize == config["latentSize"]);
  });
  it("syncopate model properly runs forward function", async function () {
    const model = await ONNXModel.load(
      "./regroove-models/v2/graceful-fire-240/model.onnx",
      2
    );

    // prepare input data
    const [onsets, velocities, offsets] = await testPattern();
    let pattern = onsets.concatenate(velocities, 2);
    pattern = pattern.concatenate(offsets, 2);

    // repeat and flatten
    const batchSize = 2;
    let repeat_pattern = pattern;
    for (let i = 1; i < batchSize; i++) {
      repeat_pattern = repeat_pattern.concatenate(pattern, 0);
    }
    const noteDropout = 0.5;

    const output = await model.forward(repeat_pattern, noteDropout);
    assert.ok(output.onsets.size == onsets.size * batchSize);

    // TODO: More tests on output shape and actual values
  });
  it("groove model is constructed correctly", async function () {
    const model = await ONNXModel.load(
      "./regroove-models/v2/olive-lion-52/model.onnx",
      32
    );
    const configData = fs.readFileSync(
      "./regroove-models/v2/groove.json",
      "utf-8"
    );
    const config = JSON.parse(configData);
    assert.ok(model.latentSize == config["latentSize"]);
  });
  it("groove model properly runs forward function", async function () {
    const model = await ONNXModel.load(
      "./regroove-models/v2/olive-lion-52/model.onnx",
      32
    );

    // prepare input data
    const [onsets, velocities, offsets] = await testPattern();
    let pattern = onsets.concatenate(velocities, 2);
    pattern = pattern.concatenate(offsets, 2);

    // repeat and flatten
    const batchSize = 2;
    let repeat_pattern = pattern;
    for (let i = 1; i < batchSize; i++) {
      repeat_pattern = repeat_pattern.concatenate(pattern, 0);
    }
    const noteDropout = 0.5;

    const output = await model.forward(repeat_pattern, noteDropout);
    assert.ok(output.onsets.size == onsets.size * batchSize);

    // TODO: More tests on output shape and actual values
  });
});

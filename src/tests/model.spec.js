import assert from "assert";
import fs from 'fs';
import { describe, it } from 'mocha'

import { ONNXModel } from "../model";
import { LOCAL_MODEL_DIR } from "../constants";
import { testPattern } from "./helpers.ts";

describe("ONNXModel", function () {
  it("syncopate model is constructed correctly", async function () {
    const model = await ONNXModel.build("syncopate", LOCAL_MODEL_DIR);
    const configData = fs.readFileSync(LOCAL_MODEL_DIR + "/syncopate.json", 'utf-8')
    const config = JSON.parse(configData)
    assert.ok(model.meta.latentSize == config["latentSize"]);
    assert.ok(model.meta.channels == config["channels"]);
    assert.ok(model.meta.loopDuration == config["loopDuration"]);
  });
  it("syncopate model properly runs forward function", async function () {
    // build model
    const model = await ONNXModel.build("syncopate", LOCAL_MODEL_DIR);

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
    assert.ok(output.onsets.size == onsets.size*batchSize)

    // TODO: More tests on output shape and actual values
  });
  it("groove model is constructed correctly", async function () {
    const model = await ONNXModel.build("groove", LOCAL_MODEL_DIR);
    const configData = fs.readFileSync(LOCAL_MODEL_DIR + "/groove.json", 'utf-8')
    const config = JSON.parse(configData)
    assert.ok(model.meta.latentSize == config["latentSize"]);
    assert.ok(model.meta.channels == config["channels"]);
    assert.ok(model.meta.loopDuration == config["loopDuration"]);
  });
  it("groove model properly runs forward function", async function () {
    // build model
    const model = await ONNXModel.build("groove", LOCAL_MODEL_DIR);

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
    assert.ok(output.onsets.size == onsets.size*batchSize)

    // TODO: More tests on output shape and actual values
  });
});

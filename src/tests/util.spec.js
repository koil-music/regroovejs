import assert from "assert";
import { describe, it } from "mocha";
import { Tensor } from "onnxruntime-web";

import { Pattern } from "../pattern";
import { arraysEqual } from "./helpers";
import { linspace, round, normalize, applyOnsetThreshold } from "../util";

describe("linspace", function () {
  it("handles correct input", function () {
    const max = 1;
    const min = 0;
    const length = 5;
    const expected = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
    const got = linspace(min, max, length);
    assert.ok(arraysEqual(got, expected));
  });
});

describe("round", function () {
  it("handles correct input", function () {
    const value = 1.534556;

    const depth = 3;
    const expected = 1.535;
    const got = round(value, depth);
    assert.strictEqual(got, expected);
  });
  it("handles integer rounding", function () {
    const value = 1.784;
    const depth = 0;
    const expected = 2;
    const got = round(value, depth);
    assert.strictEqual(got, expected);
  });
});

describe("applyOnsetThreshold", function () {
  it("applies onset properly", async function () {
    const data = new Float32Array(8);
    const expectedDims = [1, 4, 2];
    data.fill(0.5);
    let gotPattern = applyOnsetThreshold(
      new Tensor("float32", data, expectedDims),
      expectedDims,
      0.4
    );
    let expected = Array.from({ length: 8 }, () => 1);
    assert.ok(arraysEqual(Array.from(gotPattern.data), expected));

    data.fill(0.3);
    gotPattern = applyOnsetThreshold(
      new Tensor("float32", data, expectedDims),
      expectedDims,
      0.4
    );
    expected = Array.from({ length: 8 }, () => 0);
    assert.ok(arraysEqual(Array.from(gotPattern.data), expected));
  });
});

describe("normalize", function () {
  it("runs as expected", function () {
    const dims = [1, 4, 2];
    const data = Float32Array.from({ length: dims[1] * dims[2] }, () => 0.7);
    const inputPattern = new Pattern(data, dims);

    const target = 0.5;
    const gotPattern = normalize(inputPattern, dims, target);
    const expectedData = Array.from(
      { length: dims[1] * dims[2] },
      () => target
    );
    assert.ok(arraysEqual(Array.from(gotPattern.data), expectedData));
  });
});

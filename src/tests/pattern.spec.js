import assert from "assert";
import { describe, it } from "mocha";

import { Pattern } from "../pattern";
import { transpose2d } from "../pattern";
import { arraysEqual, testPattern } from "./helpers.ts";
import { LOOP_DURATION, CHANNELS } from "../constants";

describe("Pattern", function () {
  it("Constructs correctly", function () {
    const data = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let pattern = new Pattern(data, [3, 2, 2]);
    // console.log(pattern)
    assert.ok(pattern.data.length == data.length);
    assert.ok(pattern.dims.length == 3);

    const dims = [1, 4, 3];
    const patternData = pattern.view(dims);
    pattern = new Pattern(patternData, dims);
    assert.ok(pattern.data.length == dims[0] * dims[1] * dims[2]);
  });
  it(".tensor() returns expected output with valid data", function () {
    const data = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const pattern = new Pattern(data, [3, 2, 2]);

    const expected = [
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
      [
        [9, 10],
        [11, 12],
      ],
    ];
    const got = pattern.tensor();
    assert.ok(arraysEqual(expected, got));
  });
  it(".view() handles valid data correctly", function () {
    const data = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const pattern = new Pattern(data, [3, 2, 2]);

    let input = [3, 1, 4];
    let expected = [[[1, 2, 3, 4]], [[5, 6, 7, 8]], [[9, 10, 11, 12]]];
    let got = pattern.view(input);
    assert.ok(arraysEqual(expected, got));

    input = [1, 3, 4];
    expected = [
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ],
    ];
    got = pattern.view(input);
    assert.ok(arraysEqual(expected, got));
  });
  it(".transpose() handles valid data correctly", function () {
    // const data = Float32Array.from([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., ]);
    const dims = [2, 3, 4];
    const data = Float32Array.from(
      { length: dims[0] * dims[1] * dims[2] },
      (_, k) => k + 1
    );
    const pattern = new Pattern(data, dims);

    const got = pattern.transpose();
    const expected = [
      [
        [1, 5, 9],
        [2, 6, 10],
        [3, 7, 11],
        [4, 8, 12],
      ],
      [
        [13, 17, 21],
        [14, 18, 22],
        [15, 19, 23],
        [16, 20, 24],
      ],
    ];
    assert.ok(arraysEqual(expected, got));
  });
  it(".concatenate() handles valid data correctly", function () {
    const dims = [2, 3, 1];
    const data = Float32Array.from(
      { length: dims[0] * dims[1] * dims[2] },
      (_, k) => k + 1
    );
    const pattern = new Pattern(data, dims);
    const pattern2 = new Pattern(data, dims);

    let got = pattern.concatenate(pattern2, 0);
    let expected = [
      [[1], [2], [3]],
      [[4], [5], [6]],
      [[1], [2], [3]],
      [[4], [5], [6]],
    ];
    assert.ok(arraysEqual(got.tensor(), expected));

    got = pattern.concatenate(pattern2, 1);
    expected = [
      [[1], [2], [3], [1], [2], [3]],
      [[4], [5], [6], [4], [5], [6]],
    ];
    assert.ok(arraysEqual(got.tensor(), expected));

    got = pattern.concatenate(pattern2, 2);
    expected = [
      [
        [1, 1],
        [2, 2],
        [3, 3],
      ],
      [
        [4, 4],
        [5, 5],
        [6, 6],
      ],
    ];
    const gotArray = got.tensor();
    assert.ok(arraysEqual(gotArray, expected));
  });
});

describe("Pattern", function () {
  it("loads from midi", async function () {
    const [
      onsetsPattern,
      velocitiesPattern,
      offsetsPattern,
    ] = await testPattern();

    assert.ok(onsetsPattern.tensor()[0][0].length == CHANNELS);
    assert.ok(onsetsPattern.tensor()[0].length == LOOP_DURATION);
    assert.ok(onsetsPattern.data.length == CHANNELS * LOOP_DURATION);

    assert.ok(velocitiesPattern.tensor()[0][0].length == CHANNELS);
    assert.ok(velocitiesPattern.tensor()[0].length == LOOP_DURATION);
    assert.ok(velocitiesPattern.data.length == CHANNELS * LOOP_DURATION);

    assert.ok(offsetsPattern.tensor()[0][0].length == CHANNELS);
    assert.ok(offsetsPattern.tensor()[0].length == LOOP_DURATION);
    assert.ok(offsetsPattern.data.length == CHANNELS * LOOP_DURATION);

    // const isBinary = (v: number) => v == 1.0 || v == 0.0;
    const velocityInRange = (v) => v <= 1.0 && v >= 0.0;
    const offsetsInRange = (v) => v <= 1.0 && v >= -1.0;

    for (let i = 0; i < offsetsPattern.data.length; i++) {
      assert.ok(
        offsetsPattern.data.every(offsetsInRange),
        "offset values not in range [-1, 1]"
      );
      assert.ok(
        velocitiesPattern.data.every(velocityInRange),
        "velocity values not in range [0, 1]"
      );
    }
  });
});

describe("transpose2d", function () {
  it("handles 2d input correctly", function () {
    const input = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ];
    const expected = [
      [1, 5, 9],
      [2, 6, 10],
      [3, 7, 11],
      [4, 8, 12],
    ];
    const got = transpose2d(input);
    assert.ok(arraysEqual(got, expected));
  });
});

describe("setcell", function () {
  it("sets internal data to expected value", function () {
    const dims = [1, 5, 2];
    const data = new Float32Array(dims[0] * dims[1] * dims[2]);
    const pattern = new Pattern(data, dims);

    const c = [3, 2];
    const index = (c[1] - 1) * dims[1] + c[0];
    pattern.setcell(1, c[0], c[1]);
    for (let i = 0; i < pattern.length; i++) {
      if (i == index) {
        assert.strictEqual(pattern.data[i], 1);
      } else {
        assert.strictEqual(pattern.data[i], 0);
      }
    }
  });
  it("doesn't change when batchSize > 1", function () {
    const dims = [2, 5, 2];
    const data = new Float32Array(dims[0] * dims[1] * dims[2]);
    const pattern = new Pattern(data, dims);

    const c = [3, 2];
    const index = (c[1] - 1) * dims[1] + c[0];
    pattern.setcell(1, c[0], c[1]);
    for (let i = 0; i < pattern.length; i++) {
      assert.strictEqual(pattern.data[i], 0);
    }
  });
});

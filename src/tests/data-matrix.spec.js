import assert from "assert";
import { describe, it } from "mocha";

import { LOOP_DURATION, CHANNELS } from "../constants"
import PatternDataMatrix from "../data-matrix"
import { arraysEqual } from "./helpers"
import { PatternSizeError } from "../pattern"


describe("PatternDataMatrix", function () {
  const expectedShape = [1, LOOP_DURATION, CHANNELS];
  let length = 10;
  const dataMatrix = new PatternDataMatrix(expectedShape, length);
  const pattern = Float32Array.from(
    { length: expectedShape[1] * expectedShape[2] },
    () => {
      return 1;
    }
  );

  it("constructed properly", function () {
    assert.strictEqual(dataMatrix.length, length);
    assert.strictEqual(dataMatrix.matrixSize, length ** 2);
    assert.strictEqual(dataMatrix.outputShape, expectedShape);
    assert.strictEqual(
      dataMatrix.outputSize,
      expectedShape[1] * expectedShape[2]
    );

    length = 20;
    dataMatrix.length = length;
    assert.strictEqual(dataMatrix.length, length);

    const emptyDataMatrix = dataMatrix.empty();
    assert.strictEqual(emptyDataMatrix.length, length);
    assert.strictEqual(emptyDataMatrix[0].length, length);
    assert.strictEqual(emptyDataMatrix[0][0].length, LOOP_DURATION * CHANNELS);
    for (let i = 0; i < length; i++) {
      for (let j = 0; j < length; j++) {
        assert.ok(
          arraysEqual(
            Array.from(emptyDataMatrix[i][j]),
            Array.from({ length: LOOP_DURATION * CHANNELS }, () => 0)
          )
        );
      }
    }
  }),
    it("appends correct patterns", function () {
      for (let i = 0; i < length; i++) {
        for (let j = 0; j < length; j++) {
          dataMatrix.append(pattern, i, j);
          const gotPattern = dataMatrix.sample(i, j);
          assert.ok(arraysEqual(Array.from(gotPattern), Array.from(pattern)));
        }
      }
    }),
    it("rejects out of bounds append indices", function () {
      let i = length + 1;
      let j = 0;
      dataMatrix.append(pattern, i, j);
    }),
    it("throws patternSizeError", function () {
      let i = 1;
      let j = 1;
      const invalidLength = expectedShape[1] * expectedShape[2] + 1;
      const invalidPattern = Float32Array.from(
        { length: invalidLength },
        () => {
          return 1;
        }
      );
      assert.throws(function () {
        dataMatrix.append(invalidPattern, i, j);
      }, PatternSizeError);
    }),
    it("fails when indexing out of bounds", function () {
      let i = length + 1;
      let j = 1;
      assert.throws(function () {
        dataMatrix.sample(i, j);
      }, TypeError);
    });
  it("gets and sets data", function () {
    // TODO
    // 1. Actually test PatternDataMatrix here
    // 2. empty() should return a PatternDataMatrix
    const testDataMatrix = dataMatrix.empty();
    const expectedData = Array.from(
      { length: LOOP_DURATION * CHANNELS },
      () => 1
    );
    // assert.ok(arraysEqual(Array.from(testDataMatrix.data), Array.from(expectedData)))
    testDataMatrix[0][0] = Array.from(
      { length: LOOP_DURATION * CHANNELS },
      () => 1
    );
    assert.ok(
      arraysEqual(Array.from(testDataMatrix[0][0]), Array.from(expectedData))
    );
  });
  it("normal", function () {
    const testData = dataMatrix.empty();
    const testDataMatrix = new PatternDataMatrix(expectedShape, length);
    testDataMatrix.data = testData;
    const ones = Array.from({ length: LOOP_DURATION * CHANNELS }, () => 1);
    const fives = Array.from({ length: LOOP_DURATION * CHANNELS }, () => 5);
    for (let i = 0; i < testDataMatrix.length; i++) {
      for (let j = 0; j < testDataMatrix.length; j++) {
        if (j % 2 === 0) {
          testDataMatrix._T[i][j] = Float32Array.from(fives);
        } else {
          testDataMatrix._T[i][j] = Float32Array.from(ones);
        }
      }
    }
    let [gotMean, gotStd] = testDataMatrix.normal(0.5);
    assert.ok(gotMean === 3);
    assert.ok(gotStd === 2);

    [gotMean, gotStd] = testDataMatrix.normal(2);
    assert.ok(gotMean === 5);
    assert.ok(gotStd === 0);
  });
});
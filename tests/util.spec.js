import assert from "assert";
import { arraysEqual } from "./helpers"
import { linspace, round } from "../src/util";


describe("linspace", function() {
  it("handles correct input", function() {
    const max = 1.;
    const min = 0.;
    const length = 5;
    const expected = [0., 0.2, 0.4, 0.6, 0.8, 1.0];
    const got = linspace(min, max, length);
    assert.ok(arraysEqual(got, expected))
  })
})

describe("round", function() {
  it("handles correct input", function() {
    const value = 1.534556;

    const depth = 3;
    const expected = 1.535;
    const got = round(value, depth);
    assert.strictEqual(got, expected);
  })
  it("handles integer rounding", function() {
    const value = 1.784;
    const depth = 0;
    const expected = 2;
    const got = round(value, depth);
    assert.strictEqual(got, expected)
  })
})

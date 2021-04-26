import assert from "assert"
import { describe, it } from 'mocha'

import { PatternHistory } from "../history"
import { Pattern } from "../pattern";
import { arraysEqual } from "./helpers.ts"


describe("PatternHistory", function () {
  const length = 20;
  const history = new PatternHistory(length);
  it("is constructed correctly", async function () {
    assert.strictEqual(history._queue.length, 0)
    assert.strictEqual(history.length, length)
  })
  it("caps the length of the queue", function() {
    const dims = [1, 4, 3]
    const data = new Float32Array(dims[0] * dims[1] * dims[2])
    const pattern = new Pattern(data, dims)
    for (let i = 0; i < 21; i++) {
      history.append(pattern);
    }
    assert.strictEqual(history._queue.length, 20)
  })
  it("acts like a FIFO queue", function() {
    const dims = [1, 4, 3]
    const data = new Float32Array(dims[0] * dims[1] * dims[2])
    const pattern = new Pattern(data, dims)
    history.append(pattern);
    
    data.fill(1.)
    const newPattern = new Pattern(data, dims);
    for (let i = 0; i < newPattern.length; i++) {
        assert.strictEqual(newPattern.data[i], 1.);
    }
    for (let i = 0; i < 19; i++) {
      history.append(newPattern); 
    }
    let gotPattern = Array.from(history.sample(0).data)
    let expectedPattern = Array.from(pattern.data)
    assert.ok(arraysEqual(gotPattern, expectedPattern))

    gotPattern = Array.from(history.sample(1).data)
    expectedPattern = Array.from(newPattern.data)
    assert.ok(arraysEqual(gotPattern, expectedPattern))
  })
})

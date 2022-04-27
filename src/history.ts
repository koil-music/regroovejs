import { Pattern } from "./pattern";

class PatternHistory {
  _length: number;
  _queue: Pattern[];

  /**
   * Contrained list to hold pattern history
   * @param {number} length: Max number of patterns in queue
   */
  constructor(length: number) {
    this._length = length;
    this._queue = [];
  }

  get length(): number {
    return this._length;
  }

  set length(value: number) {
    this._length = value;
  }

  append(pattern: Pattern): void {
    this._queue.unshift(pattern);
    if (this._queue.length > this._length) {
      this._queue.pop();
    }
  }

  sample(idx: number): Pattern {
    return this._queue[idx];
  }
}

export { PatternHistory };

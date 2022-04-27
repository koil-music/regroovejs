import { Pattern, PatternSizeError } from "./pattern";

interface IPatternDataMatrix {
  empty: () => Float32Array[][];
  append: (p: Float32Array, i: number, j: number) => void;
  sample: (i: number, j: number) => Float32Array;
  normal: (threshold: number) => [number, number];
}

class PatternDataMatrix implements IPatternDataMatrix {
  /**
   * 2D data matrix that holds Pattern instances
   */
  outputShape: [number, number, number];
  dims: [number, number, number];
  _length: number;
  _T: Float32Array[][];

  constructor(outputShape: [number, number, number], length: number) {
    this.outputShape = outputShape;
    this.dims = [1, outputShape[1], outputShape[2]];
    this._length = length;
    this._T = this.empty();
  }

  empty(): Float32Array[][] {
    return Array.from({ length: this.length }, () => {
      return Array.from({ length: this.length }, () => {
        return new Float32Array(this.outputSize);
      });
    });
  }

  get length(): number {
    return this._length;
  }

  set length(value: number) {
    this._length = value;
    this._T = this.empty();
  }

  get matrixSize(): number {
    return this.length ** 2;
  }

  get outputSize(): number {
    return this.outputShape[0] * this.outputShape[1] * this.outputShape[2];
  }

  get data(): Float32Array[][] {
    return this._T;
  }

  set data(d: Float32Array[][]) {
    this._T = d;
  }

  append(p: Float32Array, i: number, j: number): void {
    if (i < this.length && j < this.length) {
      if (p.length === this.outputSize) {
        this._T[i][j] = p;
      } else {
        throw new PatternSizeError(this.outputSize, p.length);
      }
    } else {
      console.warn(
        `Index (${i},${j}) is out of bounds for data matrix of size ${this.length}`
      );
    }
  }

  sample(i: number, j: number): Float32Array {
    return this._T[i][j];
  }

  normal(threshold: number): [number, number] {
    const means: number[] = [];
    for (let i = 0; i < this.length; i++) {
      for (let j = 0; j < this.length; j++) {
        const pattern = new Pattern(this.sample(i, j), this.dims);
        const patternMean = pattern.mean(threshold);
        if (!isNaN(patternMean)) {
          means.push(patternMean);
        }
      }
    }

    // get mean
    const sum = (sum: number, value: number) => sum + value;
    const mean = means.reduce(sum) / means.length;

    // get std
    let rms = 0;
    for (const v of means) {
      rms += (v - mean) ** 2;
    }
    const std = Math.sqrt(rms / means.length);

    return [mean, std];
  }
}

export default PatternDataMatrix;

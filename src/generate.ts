import fs from 'fs'
import { promisify } from 'util'

import { Tensor } from 'onnxruntime'
import { ONNXModel } from './model'
import { Pattern, PatternSizeError } from './pattern'
import {
  LOOP_DURATION,
  CHANNELS,
  MIN_ONSET_THRESHOLD,
  MAX_ONSET_THRESHOLD,
  NUM_SAMPLES,
  NOTE_DROPOUT
} from './constants'
import { linspace } from './util'

const asyncReadFile = promisify(fs.readFile)
const asyncWriteFile = promisify(fs.writeFile)

class PatternDataMatrix {
  /**
   * 2D data matrix that holds Pattern instances
  */
  outputShape: [number, number, number]
  _length: number
  _T: Float32Array[][]

  constructor (outputShape: [number, number, number], length: number) {
    this.outputShape = outputShape
    this._length = length
    this._T = this.empty()
  }
  
  empty (): Float32Array[][] {
    return Array.from({ length: this.length }, () => {
      return Array.from({ length: this.length }, () => {
        return new Float32Array(this.outputSize)
      })
    })
  }

  get length (): number {
    return this._length
  }

  set length (value: number) {
    this._length = value
    this._T = this.empty()
  }

  get matrixSize (): number {
    return this.length ** 2
  }

  get outputSize (): number {
    return this.outputShape[0] * this.outputShape[1] * this.outputShape[2]
  }

  get data (): Float32Array[][] {
    return this._T
  }

  set data (d: Float32Array[][]) {
    this._T = d
  }

  append (p: Float32Array, i: number, j: number): void {
    if (i < this.length && j < this.length) {
      if (p.length === this.outputSize) {
        this._T[i][j] = p
      } else {
        throw new PatternSizeError(this.outputSize, p.length)
      }
    } else {
      console.warn(`Index (${i},${j}) is out of bounds for data matrix of size ${this.length}`)
    }
  }

  sample (i: number, j: number): Float32Array {
    return this._T[i][j]
  }
}

class Generator {
  /**
   * This class wraps our loaded ONNX syncopateModel and fills an Object (data)
   * with predicted patterns
   *
   * @syncopateModel          Instance of ONNXModel trained for syncopating patterns
   * @grooveModel             Instance of ONNXModel trained for grooving patterns
   * @onsetsPattern           Instance of Pattern to be used as input onsets
   * @velocitiesPattern       Instance of Pattern to be used as input velocities
   * @offsetsPattern          Instance of Pattern to be used as input offsets
   * @_dims                   Tuple of matrix dimension sizes for row of generator data
   * @_channels               Number of channels in a Pattern
   * @_loopDuration           Number of sequence steps in a Pattern
   * @_outputShape            Tuple of matrix dimensions sizes for ONE instance of Pattern
   * @_numSamples             Number of samples to generate - the sqrt is taken and the
                              result rounded and used as the axis size of the pattern grid
   * @_noteDropout            Mean note dropout probability
   * @_min/maxNoteDropout     Variance around mean note dropout probability
   * @_minOnsetThreshold      Minimum onset threshold value for generated samples
   * @_maxOnsetThreshold      Maximum onset threshold value for generated samples
   * @_onsetsDataMatrix       Instance of PatternSizeError containing onsets data
   * @_velocitiesDataMatrix   Instance of PatternDataMatrix containing velocities data
   * @_offsetsDataMatrix      Instance of PatternDataMatrix containing offsets data
   */
  syncopateModel: ONNXModel
  grooveModel: ONNXModel
  onsetsPattern: Pattern
  velocitiesPattern: Pattern
  offsetsPattern: Pattern
  _dims: number[]
  _channels: number
  _loopDuration: number
  _outputShape: [number, number, number]
  _numSamples: number
  _noteDropout: number
  _minNoteDropout: number
  _maxNoteDropout: number
  _minOnsetThreshold: number
  _maxOnsetThreshold: number
  _onsetsDataMatrix: PatternDataMatrix
  _velocitiesDataMatrix: PatternDataMatrix
  _offsetsDataMatrix: PatternDataMatrix

  constructor (
    syncopateModel: ONNXModel,
    grooveModel: ONNXModel,
    onsets: Float32Array,
    velocities: Float32Array,
    offsets: Float32Array,
    numSamples: number,
    noteDropout: number,
    channels: number,
    loopDuration: number,
    minOnsetThreshold: number,
    maxOnsetThreshold: number
  ) {
    if (typeof syncopateModel === 'undefined') {
      throw new Error(
        'cannot directly access class constructor - use Generator.build(<params>) instead.'
      )
    }
    this.syncopateModel = syncopateModel
    this.grooveModel = grooveModel
    this._numSamples = numSamples
    this._noteDropout = noteDropout
    this._channels = channels
    this._loopDuration = loopDuration
    this._outputShape = [1, this.loopDuration, this.channels]
    this._dims = [this.axisLength, this._loopDuration, this._channels]
    this._minOnsetThreshold = minOnsetThreshold
    this._maxOnsetThreshold = maxOnsetThreshold
    this.onsetsPattern = new Pattern(onsets, [1, LOOP_DURATION, CHANNELS])
    this.velocitiesPattern = new Pattern(velocities, [1, LOOP_DURATION, CHANNELS])
    this.offsetsPattern = new Pattern(offsets, [1, LOOP_DURATION, CHANNELS])
    this._onsetsDataMatrix = new PatternDataMatrix(this.outputShape, this.axisLength)
    this._velocitiesDataMatrix = new PatternDataMatrix(this.outputShape, this.axisLength)
    this._offsetsDataMatrix = new PatternDataMatrix(this.outputShape, this.axisLength)
  }

  get minOnsetThreshold (): number {
    return this._minOnsetThreshold
  }

  set minOnsetThreshold (value: number) {
    this._minOnsetThreshold = value
  }

  get maxOnsetThreshold (): number {
    return this._maxOnsetThreshold
  }

  set maxOnsetThreshold (value: number) {
    this._maxOnsetThreshold = value
  }

  get onsetThresholdRange (): number[] {
    return linspace(this._minOnsetThreshold, this._maxOnsetThreshold, this.axisLength)
  }

  get numSamples (): number {
    return this._numSamples
  }

  set numSamples (value: number) {
    this._numSamples = value
    this._onsetsDataMatrix.length = value
    this._velocitiesDataMatrix.length = value
    this._offsetsDataMatrix.length = value
  }

  get axisLength (): number {
    return Math.round(Math.sqrt(this._numSamples))
  }

  set noteDropout (value: number) {
    this._noteDropout = value
  }

  get noteDropout (): number {
    return this._noteDropout
  }

  get minNoteDropout (): number {
    return this._noteDropout - 0.05
  }

  get maxNoteDropout (): number {
    return this._noteDropout + 0.05
  }

  get channels (): number {
    return this._channels
  }

  set channels (value: number) {
    this._channels = value
    this._outputShape[2] = value
  }

  get loopDuration (): number {
    return this._loopDuration
  }

  set loopDuration (value: number) {
    this._loopDuration = value
    this._outputShape[1] = value
  }

  get outputShape (): [number, number, number] {
    return this._outputShape
  }

  get dims (): number[] {
    return this._dims
  }

  get onsets (): PatternDataMatrix {
    return this._onsetsDataMatrix
  }

  set onsets (matrix: PatternDataMatrix) {
    this._onsetsDataMatrix = matrix
  }

  get velocities (): PatternDataMatrix {
    return this._velocitiesDataMatrix
  }

  set velocities (matrix: PatternDataMatrix) {
    this._velocitiesDataMatrix = matrix
  }

  get offsets (): PatternDataMatrix {
    return this._offsetsDataMatrix
  }

  set offsets (matrix: PatternDataMatrix) {
    this._offsetsDataMatrix = matrix
  }

  static async build (
    onsets: Float32Array,
    velocities: Float32Array,
    offsets: Float32Array,
    modelDir: string,
    minOnsetThreshold: number = MIN_ONSET_THRESHOLD,
    maxOnsetThreshold: number = MAX_ONSET_THRESHOLD,
    numSamples: number = NUM_SAMPLES,
    noteDropout: number = NOTE_DROPOUT,
    instruments: number = CHANNELS,
    sequenceLength: number = LOOP_DURATION
  ): Promise<Generator> {
    try {
      const syncopateModel = await ONNXModel.build('syncopate', modelDir)
      const grooveModel = await ONNXModel.build('groove', modelDir)
      return new Generator(
        syncopateModel,
        grooveModel,
        onsets,
        velocities,
        offsets,
        numSamples,
        noteDropout,
        instruments,
        sequenceLength,
        minOnsetThreshold,
        maxOnsetThreshold
      )
    } catch (e) {
      console.error('failed to load Generator')
      throw new Error(e)
    }
  }
  
  async load(filepath: string): Promise<void> {
    /**
     * Load the generator state, consisting of the onsetsDataMatrix, velocitiesDataMatrix,
     * offsetsDataMatrix from file.
     */
    const d = await asyncReadFile(filepath, "utf-8");
    const data = JSON.parse(d);

    const onsetsDataMatrix = new PatternDataMatrix(
      data["outputShape"],
      data["length"]
    );
    const velocitiesDataMatrix = new PatternDataMatrix(
      data["outputShape"],
      data["length"]
    );
    const offsetsDataMatrix = new PatternDataMatrix(
      data["outputShape"],
      data["length"]
    );

    for (let i = 0; i < data["length"]; i++) {
      for (let j = 0; j < data["length"]; j++) {
        const onsetsMatrix = JSON.parse(data["onsets"])
        const onsetsData = Float32Array.from(Object.values(onsetsMatrix[i][j]))
        onsetsDataMatrix.append(onsetsData, i, j)

        const velocitiesMatrix = JSON.parse(data["onsets"])
        const velocitiesData = Float32Array.from(Object.values(velocitiesMatrix[i][j]))
        velocitiesDataMatrix.append(velocitiesData, i, j)

        const offsetsMatrix = JSON.parse(data["onsets"])
        const offsetsData = Float32Array.from(Object.values(offsetsMatrix[i][j]))
        offsetsDataMatrix.append(offsetsData, i, j)
      }
    }

    this._onsetsDataMatrix = onsetsDataMatrix;
    this._velocitiesDataMatrix = velocitiesDataMatrix;
    this._offsetsDataMatrix = offsetsDataMatrix;
  }

  async save(filepath: string): Promise<void> {
    /*
     * Save the onsetsDataMatrix, velocitiesDataMatrix, offsetsDataMatrix data
     * to file along with necessary ( for PatternDataMatrix construction ) metadata
     */
    const data = {
      outputShape: this.outputShape,
      length: this.onsets.length,
      onsets: JSON.stringify(this.onsets),
      velocities: JSON.stringify(this.velocities),
      offsets: JSON.stringify(this.offsets)
    };
    const stringData = JSON.stringify(data)
    await asyncWriteFile(filepath, stringData);
  }

  batchedInput (onsetsPattern: Pattern, batchSize: number): Pattern {
    /**
     * Repeats input pattern batch_size times and returns a flat typed buffer.
     */
    const dims = onsetsPattern.shape
    const velocities = new Pattern(new Float32Array(onsetsPattern.data.length), dims)
    const offsets = new Pattern(new Float32Array(onsetsPattern.data.length), onsetsPattern.shape)
    let pattern = onsetsPattern.concatenate(velocities, 2)
    pattern = pattern.concatenate(offsets, 2)

    let batch = pattern
    for (let i = 0; i < batchSize - 1; i++) {
      batch = batch.concatenate(pattern, 0)
    }
    return batch
  }

  async run (): Promise<void> {
    /**
     * Generates syncopateModel predictions as parameterized
     * by onsetThresholdRange, noteDropout, and numSamples.
     */
    const syncopateModel = this.syncopateModel
    const grooveModel = this.grooveModel
    const noteDropouts = linspace(this.minNoteDropout, this.maxNoteDropout, this.axisLength)
    const onsetThresholds = linspace(this.minOnsetThreshold, this.maxOnsetThreshold, this.axisLength)

    for (let i = 0; i < this.axisLength; i++) {
      const threshold = onsetThresholds[i]
      const dropout = noteDropouts[i]
      const syncopateInputBatch = this.batchedInput(this.onsetsPattern, this.axisLength)
      const syncopateOutput = await syncopateModel.forward(syncopateInputBatch, dropout)

      const syncopateOnsets = applyOnsetThreshold(syncopateOutput.onsets, this.dims, threshold)

      // TODO: Test performance if we concatenate all onsets and run only a single
      const grooveInputBatch = this.batchedInput(syncopateOnsets, 1)
      const grooveOutput = await grooveModel.forward(grooveInputBatch, 1)

      const onsetsBatch = syncopateOnsets.tensor()
      const velocitiesBatch = new Pattern(grooveOutput.velocities, this.dims).tensor()
      const offsetsBatch = new Pattern(grooveOutput.offsets, this.dims).tensor()

      for (let j = 0; j < this.axisLength; j++) {
        // TODO: Probably need to transpose all vectors
        const onsets = new Pattern([onsetsBatch[j]], this.outputShape)
        this._onsetsDataMatrix.append(onsets.data, i, j)

        const velocities = new Pattern([velocitiesBatch[j]], this.outputShape)
        this._velocitiesDataMatrix.append(velocities.data, i, j)

        const offsets = new Pattern([offsetsBatch[j]], this.outputShape)
        this._offsetsDataMatrix.append(offsets.data, i, j)
      }
    }
  }
}

function applyOnsetThreshold (onsets: Tensor, dims: number[], threshold: number): Pattern {
  const onsetsPattern = new Pattern(onsets, dims)
  const outputArray = onsetsPattern.data.map(v => {
    if (v < threshold) {
      return 0.0
    } else {
      return 1.0
    }
  })
  return new Pattern(outputArray, dims)
}

export { Generator, applyOnsetThreshold, PatternDataMatrix }

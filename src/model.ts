import { InferenceSession, Tensor } from "./onnxruntime";

import { zeroArray } from "./util";
import { Pattern } from "./pattern";

class ONNXModel {
  /**
   * Wraps ONNX model for stateful inference sessions
   */
  session: typeof InferenceSession;
  latentSize: number;

  constructor(session: typeof InferenceSession, latentSize: number) {
    if (typeof session === "undefined") {
      console.error(
        "cannot be called directly - use await Model.build(pattern) instead"
      );
      throw new Error();
    }
    this.session = session;
    this.latentSize = latentSize;
  }

  static async load(modelPath: string, latentSize: number): Promise<ONNXModel> {
    /**
     * @filePath Path to ONNX model
     */
    try {
      const session = await InferenceSession.create(modelPath);
      return new ONNXModel(session, latentSize);
    } catch (e) {
      console.error(`failed to load ONNX model from ${modelPath}`);
      throw new Error();
    }
  }

  async forward(
    input: Pattern,
    noteDropout = 0.5
  ): Promise<Record<string, typeof Tensor>> {
    /**
     * Forward pass of ONNX model.
     *
     * @param {Pattern} input: Input pattern to base predictions on
     * @param {number} noteDropout: Probability of note dropout when generating new pattern
     *
     * @returns output indices
     */
    const feeds = {
      input: input,
      target: input,
      delta_z: new Tensor("float32", zeroArray(this.latentSize), [
        this.latentSize,
      ]),
      note_dropout: new Tensor("float32", [noteDropout], [1]),
    };
    const output = await this.session.run(feeds);
    return output;
  }
}

export { ONNXModel };

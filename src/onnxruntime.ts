const isBrowser = typeof window !== 'undefined';
let InferenceSession, Tensor
if (isBrowser) {
  ({ InferenceSession, Tensor } = require("onnxruntime-web"))
} else {
  ({ InferenceSession, Tensor } = require("onnxruntime"))
}

export { InferenceSession, Tensor }
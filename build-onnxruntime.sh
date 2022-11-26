#!/bin/bash
ONNXRUNTIME_SUBMODULE_PATH=submodules/onnxruntime
ONNXRUNTIME_BINDING_FILE=$ONNXRUNTIME_LIB_PATH/build/Release/onnxruntime_binding.node
OS=`uname -s`-`uname -m`

if [ -f $ONNXRUNTIME_BINDING_FILE ]; then 
  echo "Found onnxruntime_binding.node, continuing to install."
  exit 0;
fi

# build onnxruntime
echo "onnxruntime_binding.node not found, building onnxruntime with Node.js bindings."
git submodule update --init --recursive

# define command depending on OS
BUILD_CMD="./build.sh --config RelWithDebInfo --build_nodejs --parallel"
if [ `uname -s` == "Darwin" ]; then
  BUILD_CMD = $BUILD_CMD + " --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=" + `uname -m`
fi

cd $ONNXRUNTIME_SUBMODULE_PATH && \
  ./build.sh --cnofig RelWithDebInfo --build_nodejs --parallel

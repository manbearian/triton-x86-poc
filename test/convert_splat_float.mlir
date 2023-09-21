// RUN: triton-x86-poc-opt --triton-to-linalg %s | FileCheck %s

// RUN: triton-x86-poc-opt --triton-to-linalg %s > %t
// RUN: sed -i -e "$(grep -n '}' %t | tail -1 | cut -f1 -d':')e cat %s.driver" %t
// RUN: triton-x86-poc-opt --convert-tensor-to-linalg --eliminate-empty-tensors --empty-tensor-to-alloc-tensor \
// RUN:   --one-shot-bufferize --convert-linalg-to-loops --convert-scf-to-cf --convert-linalg-to-llvm --convert-cf-to-llvm \
// RUN:   --convert-arith-to-llvm --convert-math-to-llvm --convert-complex-to-llvm --convert-vector-to-llvm --convert-index-to-llvm \
// RUN:   --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts %t | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils | \
// RUN: FileCheck %t --check-prefix=X86

module {
    tt.func @kernel(%fin : f32,
                    %bin : f16,
                    %save0 : tensor<1024x!tt.ptr<f32>>,
                    %save1 : tensor<128x256x!tt.ptr<f16>>) -> () {
        %0 = tt.splat %fin : (f32) -> (tensor<1024xf32>)
        %1 = tt.splat %bin : (f16) -> (tensor<128x256xf16>)
        tt.store %save0, %0 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
        tt.store %save1, %1 {cache = 1 : i32, evict = 1 : i32} : tensor<128x256xf16>
        tt.return
    }
}

// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f16, %[[VAL_2:.*]]: memref<1024xf32>, %[[VAL_3:.*]]: memref<128x256xf16>, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK:           %[[VAL_8:.*]] = linalg.fill ins(%[[VAL_0]] : f32) outs(%[[VAL_7]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_9:.*]] = tensor.empty() : tensor<128x256xf16>
// CHECK:           %[[VAL_10:.*]] = linalg.fill ins(%[[VAL_1]] : f16) outs(%[[VAL_9]] : tensor<128x256xf16>) -> tensor<128x256xf16>
// CHECK:           memref.tensor_store %[[VAL_8]], %[[VAL_2]] : memref<1024xf32>
// CHECK:           memref.tensor_store %[[VAL_10]], %[[VAL_3]] : memref<128x256xf16>
// CHECK:           return
// CHECK:         }